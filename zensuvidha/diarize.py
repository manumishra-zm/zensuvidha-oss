"""Segment-level speaker gating — keep the caller, drop whoever else spoke.

The speaker gate in speaker.py judges a WHOLE utterance: one voiceprint, one verdict.
That breaks on the commonest real interference, which is not two people talking at once
but a colleague speaking in a GAP. Measured on this codebase:

  caller alone                      similarity 0.867   accepted
  caller, then a colleague          similarity 0.364   REJECTED  <- the caller's own turn
  colleague, then caller            similarity 0.380   REJECTED
  colleague in the middle           similarity 0.340   REJECTED

  both talking AT ONCE, equal loud  similarity 0.594   accepted

So simultaneous overlap is survivable and sequential interference is not: one stray
sentence in a pause throws away everything the caller said, and puts the stranger's
words in the transcript.

This module splits the utterance by speaker, keeps only the parts that match the
enrolled voiceprint, and hands Whisper just those. Everything is capability-gated and
FAILS OPEN — if the models are missing, diarization errors, or nothing matches, the
caller's original audio is passed through untouched. Silencing the real caller is a far
worse failure than letting one stray sentence through.

Models (sherpa-onnx, ONNX runtime — no torch, no GPU):
  segmentation  pyannote-segmentation-3.0   MIT   ~7MB
  embedding     3D-Speaker ERes2Net         Apache-2.0  ~38MB
  bash scripts/download_diarize.sh
"""
import logging
import os
import threading

log = logging.getLogger("zensuvidha.diarize")

# An utterance shorter than this cannot usefully be split — segmentation needs room,
# and a one-word answer is never two people.
MIN_DIARIZE_S = 1.2
# A kept segment shorter than this is a breath or a click, not speech worth sending.
MIN_SEGMENT_S = 0.35
# If we would keep less than this share of the original, something has gone wrong
# (bad enrolment, heavy overlap) — pass the original through instead of a stub.
MIN_KEEP_RATIO = 0.25
# ...unless the sliver is a real utterance in its own right. A caller who answers
# briefly while others talk around them keeps very little of the clip, and that is
# correct rather than a failure.
MIN_KEEP_S = 1.2
# A cluster is only DISCARDED when it is this far below the best-matching one. Clustering
# routinely splits ONE person into several clusters of differing quality, and an absolute
# threshold cannot tell "a different person" from "the same person, noisier segment".
# Measured on a single-speaker recording the clusters scored 0.62 / 0.40 / 0.27 — dropping
# everything under 0.55 cost 11 words. A genuinely different speaker scores 0.04-0.07
# against a 0.87 caller, a gap of ~0.8, so a wide gap separates the two cases cleanly.
DROP_GAP = 0.40
# A cluster kept only because it is CLOSE to the best must still look like the caller
# in absolute terms. Below this it is somebody else, however the rest of the clip
# scored — the same relative-AND-absolute rule the window rescan uses.
KEEP_FLOOR = 0.40


def get_diarizer(cfg: dict):
    """Build the diarizer, or None when unavailable. Never raises."""
    cfg = cfg or {}
    if not cfg.get("diarize", False):
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve(key, default):
        p = cfg.get(key) or default
        return p if os.path.isabs(p) else os.path.join(root, p)

    seg = _resolve("diarize_segmentation", "models/diarize/segmentation.onnx")
    emb = _resolve("diarize_embedding", "models/diarize/embedding.onnx")
    missing = [p for p in (seg, emb) if not os.path.isfile(p)]
    if missing:
        log.info("Diarization models not installed (%s) — segment gating off. "
                 "Run: bash scripts/download_diarize.sh", ", ".join(missing))
        return None
    try:
        return Diarizer(seg, emb, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("Diarization unavailable: %s", e)
        return None


class Diarizer:
    """Wraps sherpa-onnx offline diarization. One model, shared across calls."""

    def __init__(self, seg_model: str, emb_model: str, cfg: dict):
        import sherpa_onnx
        self._lock = threading.Lock()      # sherpa's diarizer is not concurrency-safe
        self.max_speakers = int(cfg.get("diarize_max_speakers", 3))
        c = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=seg_model),
                num_threads=1),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=emb_model, num_threads=1),
            # num_clusters MUST stay -1 (decide from the threshold). Passing
            # max_speakers here does NOT cap the count — sherpa treats a positive value
            # as an EXACT number and would split a single caller into that many clusters,
            # which is precisely the fragmentation DROP_GAP exists to survive.
            # max_speakers is applied after segmentation instead, in segments().
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
            min_duration_on=0.25,
            min_duration_off=0.25,
        )
        if not c.validate():
            raise ValueError("sherpa-onnx rejected the diarization config")
        self._d = sherpa_onnx.OfflineSpeakerDiarization(c)
        self.sample_rate = self._d.sample_rate
        log.info("Diarization ready (sherpa-onnx, %d Hz)", self.sample_rate)

    def segments(self, samples, sr: int = 16000):
        """[(start_s, end_s, speaker_id)] — empty when it can't or shouldn't split."""
        import numpy as np
        if samples is None or len(samples) < int(MIN_DIARIZE_S * sr):
            return []
        try:
            with self._lock:
                res = self._d.process(np.asarray(samples, dtype="float32")).sort_by_start_time()
            segs = [(s.start, s.end, s.speaker) for s in res]
            # Apply the configured max_speakers HERE, where it can act without forcing a
            # count. If the clusterer found more voices than a phone call plausibly has,
            # it has fragmented somebody: keep the speakers who actually hold the floor
            # (most speech time) and fold the long tail into the nearest of them, rather
            # than handing the gate a dozen thin clusters to score.
            if self.max_speakers > 0:
                spoken = {}
                for a, b, spk in segs:
                    spoken[spk] = spoken.get(spk, 0.0) + max(0.0, b - a)
                if len(spoken) > self.max_speakers:
                    keep = {spk for spk, _ in sorted(spoken.items(), key=lambda kv: -kv[1])
                            [: self.max_speakers]}
                    log.info("diarize: %d clusters collapsed to the %d with the most "
                             "speech", len(spoken), self.max_speakers)
                    segs = [(a, b, spk) for a, b, spk in segs if spk in keep]
            return segs
        except Exception as e:  # noqa: BLE001
            log.warning("diarization failed (%s) — treating as a single speaker", e)
            return []


# ---- second opinion on "one speaker" ---------------------------------------
# Only for clips long enough to actually HOLD two turns; a short answer cannot be
# contaminated by a whole sentence from somebody else, and re-embedding every short
# turn would cost far more than it saves.
RESCAN_MIN_S = 4.0
RESCAN_WIN_S = 1.5          # long enough for a usable embedding
RESCAN_HOP_S = 0.75         # half-overlap, so a boundary never falls in a blind spot
# ...but never more windows than this. The hop widens on a long clip instead, because
# the cost is one ECAPA pass per window and it is paid on the slowest turns — measured
# at 1893ms on an 11s clip before the cap, which the caller hears as the line thinking.
# Eight windows still put a boundary inside one hop of wherever it really is.
RESCAN_MAX_WINDOWS = 8
# Judged RELATIVE to the clip's own best window, exactly as DROP_GAP judges clusters.
# An absolute bar cannot work: two female voices scored 0.26-0.46 for the stranger and
# 0.56-0.78 for the caller, with no fixed number that separates them on every call.
RESCAN_GAP = 0.25
# ...and the two groups must be genuinely apart, not the two ends of one spread. One
# speaker's own windows vary, and splitting on that variation would delete half the
# caller's words — much worse than leaving a stray sentence in.
RESCAN_MARGIN = 0.08
# ...and they must be CONSECUTIVE. One low window is the pause between two of the
# caller's own sentences; a run of them is somebody else talking.
RESCAN_MIN_RUN = 2


def _resplit_if_contaminated(segs, gate, voiceprint, samples, sr, info):
    """Return segments relabelled into two speakers when part of a 'single speaker'
    clip is clearly not the caller. Otherwise return `segs` untouched.

    The segmentation model is ORDER-SENSITIVE and the clusterer merges freely, so
    "one label" is not evidence of one person. This scores overlapping windows against
    the caller's own print and looks for a real bimodal split. Windows rather than a
    single cut, because the commonest interruption is A-B-A — somebody speaking in a
    gap — which no single boundary can describe.
    """
    import numpy as np

    total = len(samples) / sr if sr else 0.0
    if total < RESCAN_MIN_S:
        return segs

    # CHEAP GATE FIRST. The overwhelmingly common clip is one person talking, and the
    # window scan spent nine ECAPA passes (182ms) on every clean sentence to confirm
    # what was already true. Three passes settle it instead: split the clip in thirds
    # and, if every third is the caller outright, no stranger's turn can be hiding in
    # it — a real turn is at least MIN_SEGMENT_S and would drag its own third below the
    # threshold. Only an ambiguous profile is worth the full scan.
    third = len(samples) // 3
    quick = []
    for i in range(3):
        chunk = samples[i * third:(i + 1) * third] if i < 2 else samples[2 * third:]
        vec = gate.embed(_wav_bytes(chunk, sr))
        quick.append(None if vec is None else gate.similarity(voiceprint, vec))
    if all(v is not None and v >= gate.threshold for v in quick):
        return segs                                  # one voice, all the way through

    win = int(RESCAN_WIN_S * sr)
    hop_s = max(RESCAN_HOP_S, (total - RESCAN_WIN_S) / max(1, RESCAN_MAX_WINDOWS - 1))
    hop = max(1, int(hop_s * sr))
    mids, sims = [], []
    for start_i in range(0, max(1, len(samples) - win + 1), hop):
        vec = gate.embed(_wav_bytes(samples[start_i:start_i + win], sr))
        if vec is None:
            continue
        mids.append((start_i + win / 2) / sr)
        sims.append(gate.similarity(voiceprint, vec))
    if len(sims) < 3:
        return segs

    # RELATIVE and ABSOLUTE, both. Relative alone falsely split a single caller across
    # two of her own sentences and deleted 0.8s of her words — the failure this whole
    # module is most afraid of. A window the GATE ITSELF would accept as the caller can
    # never be "somebody else", whatever the rest of the clip scores.
    bar = min(max(sims) - RESCAN_GAP, gate.threshold)
    hi = [v for v in sims if v >= bar]
    lo = [v for v in sims if v < bar]
    if not lo or not hi:
        return segs                                  # one thing — believe the clusterer
    if min(hi) - max(lo) < RESCAN_MARGIN:
        return segs                                  # one spread, not two groups
    # A person's turn is a RUN of low windows, not one dip. Measured on the same
    # caller speaking two sentences, the pause between them produces exactly one low
    # window (0.77 0.78 0.74 0.71 [0.50] 0.66 0.67 0.73 0.66) — acting on that deleted
    # 0.8s of her own words. Somebody else's turn produces a sustained run instead
    # (0.77 0.78 0.74 0.71 [0.45 0.52 0.35 0.23]). Contiguity is the difference.
    run = best_run = 0
    for v in sims:
        run = run + 1 if v < bar else 0
        best_run = max(best_run, run)
    if best_run < RESCAN_MIN_RUN:
        return segs                                  # a dip at a pause, not a turn

    log.info("diarize: clustered as one voice, but %d of %d windows sit %.2f below the "
             "best (%.2f vs %.2f) — re-splitting", len(lo), len(sims), RESCAN_GAP,
             max(lo), min(hi))
    info["rescanned"] = True

    def label_at(t):
        j = min(range(len(mids)), key=lambda k: abs(mids[k] - t))
        return 0 if sims[j] >= bar else 1

    out = []
    for a0, b0, _spk in segs:
        t = a0
        while t < b0 - 1e-6:
            end_t = min(b0, t + hop_s)
            out.append((t, end_t, label_at((t + end_t) / 2)))
            t = end_t
    merged = []
    for a0, b0, spk in out:                          # stitch the runs back together
        if merged and merged[-1][2] == spk and abs(merged[-1][1] - a0) < 1e-6:
            merged[-1] = (merged[-1][0], b0, spk)
        else:
            merged.append((a0, b0, spk))
    return [tuple(m) for m in merged]


def keep_matching_speaker(diarizer, gate, voiceprint, samples, sr=16000):
    """Return (audio, info) — audio trimmed to the enrolled speaker's segments.

    `info` explains what happened, for logging and tests:
      speakers   how many distinct voices were found
      kept       seconds retained
      total      seconds in
      reason     why the audio was returned as-is, if it was
      segments   [{s, e, spk, sim, keep}] — the boundaries the diarizer found, in the
                 ORIGINAL clip's timeline, with `keep` meaning "this reached Whisper".
                 None when nobody looked, which is not the same as "one segment".

    FAILS OPEN in every branch: on any doubt the ORIGINAL audio comes back, because
    dropping the caller is worse than keeping a stranger's sentence.
    """
    import numpy as np
    total = len(samples) / sr if samples is not None and sr else 0.0
    info = {"speakers": 1, "kept": total, "total": total, "reason": None,
            "segments": None, "kept_spans": None}

    if diarizer is None or gate is None or voiceprint is None:
        info["reason"] = "not enabled"
        return samples, info

    segs = diarizer.segments(samples, sr)
    if not segs:
        info["reason"] = "too short or diarization declined"
        return samples, info

    speakers = sorted({s for _a, _b, s in segs})
    info["speakers"] = len(speakers)
    _record(info, segs)
    if len(speakers) <= 1:
        # "One label" is not the same as "one person". The segmentation model is
        # ORDER-SENSITIVE and the clusterer merges freely — measured on the same two
        # voices, same gap, only the order reversed:
        #
        #     caller then stranger    2 segments, 2 speakers   trimmed correctly
        #     stranger then caller    1 segment  spanning the whole clip
        #     stranger then caller,   2 segments, BOTH labelled speaker 1
        #       with a wider gap
        #
        # In both failing shapes the stranger's words reached Whisper and the caller
        # was told nothing was removed. So do not treat the clusterer's LABELS as
        # evidence: it is good at finding boundaries and we have a better model for
        # deciding who is speaking. Re-check with ECAPA before believing "one voice".
        segs = _resplit_if_contaminated(segs, gate, voiceprint, samples, sr, info)
        speakers = sorted({s for _a, _b, s in segs})
        info["speakers"] = len(speakers)
        _record(info, segs)
        if len(speakers) <= 1:
            # The overwhelmingly common case — one voice. Nothing to do, and no cost.
            info["reason"] = "single speaker"
            return samples, info

    # Score each speaker ONCE on all of their audio joined, not per segment: a 0.4s
    # fragment gives an unreliable embedding, while a speaker's combined turns give a
    # solid one.
    #
    # Then keep EVERY cluster that matches the caller, not just the best one. Clustering
    # routinely splits ONE person into 2-3 clusters, and taking only the argmax deleted
    # the rest — measured on a single-speaker recording, a cluster scoring 0.565 (ABOVE
    # the gate's own 0.55 accept threshold, i.e. the gate agreed it was the caller) was
    # thrown away because another cluster scored 0.813, losing half the utterance and
    # silently erasing the caller's own name from the transcript.
    # The SAME bar the whole-utterance gate uses, including anything it has learned
    # about this microphone. Reading `gate.threshold` directly meant isolation kept
    # judging against the synthetic 0.55 while the gate itself had adapted to 0.11 —
    # so on a real caller's mic every cluster scored "not the caller", nothing was ever
    # trimmed, and switching Voice Isolation on did precisely nothing.
    bar = gate.effective_threshold() if hasattr(gate, "effective_threshold") \
        else gate.threshold
    sims = {}
    best, best_sim = None, -1.0
    for spk in speakers:
        joined = np.concatenate([samples[int(a*sr):int(b*sr)]
                                 for a, b, s in segs if s == spk] or [np.zeros(1, "float32")])
        if len(joined) < int(MIN_SEGMENT_S * sr):
            continue
        vec = gate.embed(_wav_bytes(joined, sr))
        if vec is None:
            continue
        sim = gate.similarity(voiceprint, vec)
        sims[spk] = sim
        log.debug("diarize: speaker %s similarity %.3f", spk, sim)
        if sim > best_sim:
            best, best_sim = spk, sim
    info["best_sim"] = best_sim
    info["similarities"] = {str(k): round(v, 3) for k, v in sims.items()}
    for seg in info["segments"] or ():
        if seg["spk"] in sims:
            seg["sim"] = round(sims[seg["spk"]], 3)

    if best is None or best_sim < bar:
        # Nobody in this clip looks like the caller. That is the whole-utterance gate's
        # decision to make, not ours — hand back the original and let it judge.
        info["reason"] = "no segment matched the caller"
        return samples, info

    # Keep a cluster unless it is CLEARLY someone else: either the gate would accept it
    # outright, or it is close enough to the best cluster that the difference is more
    # likely segment quality than a different person.
    # ...but a cluster kept ONLY by the gap rule must still be plausibly the caller.
    # The gap exists because clustering used to split one person into 2-3 clusters of
    # very different quality (measured 0.62 / 0.40 / 0.27, none of them a stranger).
    # That fragmentation was an artifact of the old spliced trimming: with it repaired,
    # the same caller across three sentences now comes back as ONE cluster at 0.94.
    # So the gap no longer has to reach down to 0.27 — and reaching that far was
    # keeping real intruders, measured at 0.31 sitting 0.37 below the caller.
    # KEEP_FLOOR and DROP_GAP are both scaled to the bar in use, and BOTH have to be.
    #
    # An absolute 0.40 floor discards the caller's own cluster on a microphone where
    # they score 0.26. But simply lowering the floor is not enough: DROP_GAP keeps any
    # cluster within 0.40 of the best, and on that same microphone the ENTIRE usable
    # range is about 0.2 wide (caller 0.26, background 0.05). A gap twice the width of
    # the range keeps everything, so the trim ran and removed nothing.
    #
    # Measured on one live call: caller 0.21-0.26, a video playing in the room
    # -0.06 to 0.05. Scaling the gap to the bar separates those; leaving it at 0.40
    # does not.
    floor = min(KEEP_FLOOR, bar)
    gap = min(DROP_GAP, max(0.12, bar * 1.5))
    mine = {spk for spk, sim in sims.items()
            if sim >= bar
            or ((best_sim - sim) <= gap and sim >= floor)}
    mine.add(best)
    info["kept_speakers"] = len(mine)
    kept_spans = [(a, b) for a, b, s in segs
                  if s in mine and (b - a) >= MIN_SEGMENT_S]
    keep = [samples[int(a*sr):int(b*sr)] for a, b in kept_spans]
    if not keep:
        info["reason"] = "matched speaker had no segment long enough"
        return samples, info
    _mark_kept(info, kept_spans)

    trimmed = np.concatenate(keep).astype("float32")

    # A cluster can itself be a MERGE of two people — the clusterer is not obliged to
    # separate them just because it found more than one group. Measured: with three
    # voices present it returned two clusters, and the one that scored 0.83 as "the
    # caller" was the caller AND the stranger joined together, so trimming to it kept
    # the stranger's whole sentence. Same shape when the caller only speaks briefly.
    # The window check already knows how to find that; it just never saw these clips,
    # because it only ran when the clusterer claimed a single speaker. Run it once more
    # on what we are about to keep.
    # ONLY when a single cluster survived. If we kept two or more, we have already
    # decided they are all the caller — a same-speaker cluster that the gap rule
    # deliberately rescued must not then be split back out by the window check. Caught
    # by test_a_weaker_cluster_of_the_same_speaker_is_kept: the second pass cut a
    # correctly-kept 6.0s down to 3.75s, deleting the caller's own quieter half.
    if len(mine) == 1 and len(trimmed) / sr >= RESCAN_MIN_S:
        sub = _resplit_if_contaminated([(0.0, len(trimmed) / sr, 0)], gate, voiceprint,
                                       trimmed, sr, info)
        mine_sub = [(a, b) for a, b, spk in sub if spk == 0]
        if len(sub) > 1 and mine_sub:
            again = np.concatenate([trimmed[int(a * sr):int(b * sr)] for a, b in mine_sub])
            if len(again) >= int(MIN_SEGMENT_S * sr):
                log.info("diarize: the kept cluster was itself mixed — trimmed again, "
                         "%.1fs of %.1fs", len(again) / sr, len(trimmed) / sr)
                trimmed = again.astype("float32")
                # The second pass ran in the TRIMMED timeline. The inspector draws the
                # caller's own recording, so map the surviving ranges back through the
                # concatenation before reporting them — otherwise the highlighted region
                # slides left by however much the first pass removed.
                _mark_kept(info, _map_back(kept_spans, mine_sub))

    info["kept"] = len(trimmed) / sr
    if info["kept"] < MIN_KEEP_RATIO * total:
        # We would be throwing away most of the audio. Usually that means OUR
        # segmentation is wrong rather than that the caller spoke for a fifth of their
        # own turn — but not always, and a ratio cannot tell the two apart. A caller who
        # says "yes, tomorrow" while two other people talk around them produces exactly
        # this shape, and refusing it handed Whisper the whole mixture instead: the
        # isolation had found the right 1.5s and the safety rule threw it away.
        #
        # So ask the sliver directly. If it is long enough to be a real utterance AND it
        # sounds like the caller on its own, it is a short turn, not an error.
        keep_it = False
        if len(trimmed) >= int(MIN_KEEP_S * sr):
            vec = gate.embed(_wav_bytes(trimmed, sr))
            if vec is not None:
                sim_kept = gate.similarity(voiceprint, vec)
                info["kept_sim"] = round(sim_kept, 3)
                keep_it = sim_kept >= gate.threshold
        if not keep_it:
            info["reason"] = f"would keep only {info['kept']:.1f}s of {total:.1f}s"
            # Everything goes to Whisper after all, so nothing may still be drawn as
            # removed. `keep` means "reached the recogniser", not "matched the caller" —
            # the similarity is still on every segment for anyone reading the colours.
            _mark_kept(info, None)
            return samples, info
        log.info("diarize: kept only %.1fs of %.1fs, but it scores %.2f on its own — "
                 "a short turn, not a segmentation error", info["kept"], total,
                 info.get("kept_sim", -1))

    info["reason"] = "trimmed to the caller"
    return trimmed, info


def _record(info: dict, segs) -> None:
    """Note the boundaries found, in the original clip's timeline.

    Written for the audio inspector: a caller looking at a turn that was cut short
    needs to see WHERE the cut was, not just that 1.4s went missing. Every segment
    starts as kept — nothing has been removed at the point this first runs, and a
    later fail-open must leave the drawing saying so.
    """
    info["segments"] = [{"s": round(float(a), 3), "e": round(float(b), 3),
                         "spk": int(s), "sim": None, "keep": True}
                        for a, b, s in segs]


def _mark_kept(info: dict, spans) -> None:
    """Record which ranges actually reached the recogniser.

    `spans` of None means all of them — the fail-open paths, where the original audio
    is returned unchanged however the scoring went.

    Two forms, because they answer different questions. `kept_spans` is what the
    waveform shades, and it must be exact: the second-pass rescan cuts INSIDE a
    segment, so a per-segment flag alone would either lose that cut or claim the whole
    segment went. `keep` is the per-segment summary the tests and the row text read,
    and a segment counts as kept when most of it survived.
    """
    segs = info.get("segments") or ()
    if spans is None:
        info["kept_spans"] = None
        for seg in segs:
            seg["keep"] = True
        return
    info["kept_spans"] = [(round(float(a), 3), round(float(b), 3)) for a, b in spans]
    for seg in segs:
        width = seg["e"] - seg["s"]
        covered = sum(max(0.0, min(seg["e"], b) - max(seg["s"], a)) for a, b in spans)
        seg["keep"] = width > 0 and covered >= 0.5 * width


def _map_back(kept_spans, sub_spans):
    """Translate ranges in the CONCATENATED audio back to the original timeline.

    The second-pass rescan sees only what the first pass kept, so its offsets are in
    a timeline with the gaps already closed up. Walking the kept spans in order and
    accumulating their durations is what re-opens them.
    """
    out, cursor = [], 0.0
    for a, b in kept_spans:
        span = b - a
        for sa, sb in sub_spans:
            lo, hi = max(sa, cursor), min(sb, cursor + span)
            if hi - lo > 1e-6:
                out.append((a + (lo - cursor), a + (hi - cursor)))
        cursor += span
    return out


def _wav_bytes(samples, sr: int) -> bytes:
    import io
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, samples, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()
