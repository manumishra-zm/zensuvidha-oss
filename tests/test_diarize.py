"""Segment-level speaker gating — keep the caller, drop whoever else spoke.

The whole-utterance gate returns ONE verdict for a whole clip, which breaks on the
commonest real interference. Measured on this codebase:

    caller alone                 similarity 0.867   accepted
    caller, then a colleague     similarity 0.364   REJECTED   <- the caller's own turn
    colleague in the middle      similarity 0.340   REJECTED
    both talking AT ONCE         similarity 0.594   accepted   <- overlap is survivable

With segment gating those become 0.872 / 0.877 ACCEPT, and the transcript loses the
stranger's words (7.7s -> 3.6s).

These tests use stubs so the suite needs no 45MB download. The real separation is
measured separately (tests/../scratch phase1_rescue).

Run:  pytest -q tests/test_diarize.py
"""
import numpy as np
import pytest

from zensuvidha.diarize import (MIN_KEEP_RATIO, MIN_SEGMENT_S, get_diarizer,
                                keep_matching_speaker)
from zensuvidha.orchestrator import Session
from zensuvidha.packs import load_pack
from zensuvidha.speaker import SpeakerGate

SR = 16000


class StubDiarizer:
    """Returns a scripted segmentation, so the gating logic can be tested exactly."""

    def __init__(self, segs):
        self._segs = segs

    def segments(self, samples, sr=SR):
        return list(self._segs)


class StubGate(SpeakerGate):
    """Embedding is decided by the FIRST sample value, so each 'speaker' is exact."""

    def __init__(self, threshold=0.55):
        self.backend = "stub"
        self.threshold = threshold

    def embed(self, audio, min_seconds=0.6):
        import io
        import soundfile as sf
        data, _sr = sf.read(io.BytesIO(audio)) if isinstance(audio, (bytes, bytearray)) \
            else (np.asarray(audio), SR)
        data = np.asarray(data, dtype="float32").reshape(-1)
        if not data.size:
            return None
        tag = int(round(abs(float(data[np.argmax(np.abs(data))])) * 10))
        v = np.zeros(8, dtype="float32")
        v[min(tag, 7)] = 1.0
        return v


def tone(level, seconds):
    """A clip whose peak encodes a 'speaker' id for StubGate."""
    return np.full(int(seconds * SR), level, dtype="float32")


CALLER, OTHER = 0.5, 0.2          # peak 0.5 -> slot 5, peak 0.2 -> slot 2
gate = StubGate()
print_caller = gate.embed(tone(CALLER, 2.0))


# --------------------------------------------------------------------------- #
# fail-open behaviour — every uncertain branch returns the ORIGINAL audio
# --------------------------------------------------------------------------- #
def test_no_diarizer_returns_the_original():
    a = tone(CALLER, 3.0)
    out, info = keep_matching_speaker(None, gate, print_caller, a, SR)
    assert out is a and info["reason"] == "not enabled"


def test_no_voiceprint_returns_the_original():
    a = tone(CALLER, 3.0)
    out, info = keep_matching_speaker(StubDiarizer([]), gate, None, a, SR)
    assert out is a and info["reason"] == "not enabled"


def test_a_single_speaker_is_passed_through_untouched():
    """The overwhelmingly common case must cost nothing and re-encode nothing."""
    a = tone(CALLER, 3.0)
    d = StubDiarizer([(0.0, 3.0, 0)])
    out, info = keep_matching_speaker(d, gate, print_caller, a, SR)
    assert out is a
    assert info["speakers"] == 1 and info["reason"] == "single speaker"


def test_diarization_declining_returns_the_original():
    a = tone(CALLER, 3.0)
    out, info = keep_matching_speaker(StubDiarizer([]), gate, print_caller, a, SR)
    assert out is a
    assert "declined" in info["reason"]


def test_no_matching_speaker_returns_the_original():
    """If nobody looks like the caller, that is the whole-utterance gate's call."""
    a = np.concatenate([tone(OTHER, 2.0), tone(0.9, 2.0)])
    d = StubDiarizer([(0.0, 2.0, 0), (2.0, 4.0, 1)])
    out, info = keep_matching_speaker(d, gate, print_caller, a, SR)
    assert out is a
    assert info["reason"] == "no segment matched the caller"


def test_trimming_almost_everything_is_refused():
    """Keeping a sliver is likelier a segmentation error than a real turn."""
    a = np.concatenate([tone(CALLER, 0.5), tone(OTHER, 9.5)])
    d = StubDiarizer([(0.0, 0.5, 0), (0.5, 10.0, 1)])
    out, info = keep_matching_speaker(d, gate, print_caller, a, SR)
    assert out is a
    assert "would keep only" in info["reason"]


# --------------------------------------------------------------------------- #
# the rescue this exists for
# --------------------------------------------------------------------------- #
def test_a_colleague_in_a_gap_is_cut_out():
    """The measured failure: caller 0.364 REJECTED -> 0.872 ACCEPTED after trimming."""
    a = np.concatenate([tone(CALLER, 3.0), tone(OTHER, 3.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 6.0, 1)])
    out, info = keep_matching_speaker(d, gate, print_caller, a, SR)
    assert out is not a
    assert info["speakers"] == 2
    assert info["reason"] == "trimmed to the caller"
    assert info["kept"] == pytest.approx(3.0, abs=0.05)
    assert gate.similarity(print_caller, gate.embed(out)) == pytest.approx(1.0, abs=1e-5)


def test_an_interruption_in_the_middle_is_stitched_back_together():
    a = np.concatenate([tone(CALLER, 2.0), tone(OTHER, 2.0), tone(CALLER, 2.0)])
    d = StubDiarizer([(0.0, 2.0, 0), (2.0, 4.0, 1), (4.0, 6.0, 0)])
    out, info = keep_matching_speaker(d, gate, print_caller, a, SR)
    assert info["kept"] == pytest.approx(4.0, abs=0.05), "both caller halves kept"
    assert info["total"] == pytest.approx(6.0, abs=0.05)


def test_segments_too_short_are_dropped():
    tiny = MIN_SEGMENT_S / 2
    a = np.concatenate([tone(CALLER, 3.0), tone(CALLER, tiny), tone(OTHER, 3.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 3.0 + tiny, 0), (3.0 + tiny, 6.0 + tiny, 1)])
    out, info = keep_matching_speaker(d, gate, print_caller, a, SR)
    assert info["kept"] == pytest.approx(3.0, abs=0.05)


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_the_diarizer_is_off_unless_asked_for():
    """It needs a 45MB download, so it must never switch itself on."""
    assert get_diarizer({}) is None
    assert get_diarizer({"diarize": False}) is None


def test_isolate_caller_is_inert_without_the_pieces():
    s = Session(load_pack("clinic"), None)
    raw = b"not audio"
    out, info = s.isolate_caller(raw)
    assert out is raw and info is None


def test_isolate_caller_never_raises_on_broken_audio():
    """A denoiser or diarizer must never be able to drop a turn."""
    s = Session(load_pack("clinic"), None, speaker_gate=gate)
    s.voiceprint = print_caller
    s.diarizer = StubDiarizer([(0.0, 1.0, 0), (1.0, 2.0, 1)])
    for junk in (b"", b"\x00\x01", b"RIFFbroken"):
        out, _info = s.isolate_caller(junk)
        assert out is junk, "must hand back exactly what it was given"


# --------------------------------------------------------------------------- #
# one speaker split across clusters must not lose their own words
# --------------------------------------------------------------------------- #
def test_a_weaker_cluster_of_the_same_speaker_is_kept():
    """Clustering routinely splits ONE person into several clusters of differing quality.
    Keeping only the argmax cost 11 words on a real single-speaker recording and erased
    the caller's own name. Clusters scored 0.62 / 0.40 / 0.27 there — none of the weaker
    ones is a different person, they are just noisier segments."""
    from zensuvidha.diarize import DROP_GAP

    class GapGate(StubGate):
        """Similarity is read straight off the clip's peak, so gaps are exact."""
        def embed(self, audio, min_seconds=0.6):
            import io
            import soundfile as sf
            data, _sr = (sf.read(io.BytesIO(audio)) if isinstance(audio, (bytes, bytearray))
                         else (np.asarray(audio), SR))
            data = np.asarray(data, dtype="float32").reshape(-1)
            if not data.size:
                return None
            peak = float(np.abs(data).max())
            v = np.zeros(2, dtype="float32")
            v[0] = peak; v[1] = np.sqrt(max(0.0, 1 - peak * peak))
            return v

    g = GapGate()
    ref = np.zeros(2, dtype="float32"); ref[0] = 1.0        # similarity == the clip's peak
    strong, weak = 0.90, 0.90 - DROP_GAP + 0.05             # weak sits INSIDE the gap
    a = np.concatenate([tone(strong, 3.0), tone(weak, 3.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 6.0, 1)])
    out, info = keep_matching_speaker(d, g, ref, a, SR)
    assert info["kept_speakers"] == 2, f"the weaker cluster was dropped: {info}"
    assert info["kept"] == pytest.approx(6.0, abs=0.05)


def test_a_clearly_different_speaker_is_still_dropped():
    """The gap rule must not become a licence to keep everyone — a real intruder scores
    ~0.05 against a 0.87 caller, a gap far wider than any same-speaker variation."""
    from zensuvidha.diarize import DROP_GAP

    class GapGate(StubGate):
        def embed(self, audio, min_seconds=0.6):
            import io
            import soundfile as sf
            data, _sr = (sf.read(io.BytesIO(audio)) if isinstance(audio, (bytes, bytearray))
                         else (np.asarray(audio), SR))
            data = np.asarray(data, dtype="float32").reshape(-1)
            if not data.size:
                return None
            peak = float(np.abs(data).max())
            v = np.zeros(2, dtype="float32")
            v[0] = peak; v[1] = np.sqrt(max(0.0, 1 - peak * peak))
            return v

    g = GapGate()
    ref = np.zeros(2, dtype="float32"); ref[0] = 1.0
    caller, intruder = 0.90, 0.90 - DROP_GAP - 0.15         # well OUTSIDE the gap
    a = np.concatenate([tone(caller, 3.0), tone(intruder, 3.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 6.0, 1)])
    out, info = keep_matching_speaker(d, g, ref, a, SR)
    assert info["kept_speakers"] == 1, f"the intruder was kept: {info}"
    assert info["kept"] == pytest.approx(3.0, abs=0.05)


# --------------------------------------------------------------------------- #
# "One speaker" is the clusterer's OPINION, not a fact
#
# Measured on the same two voices, same gap, only the ORDER reversed:
#     caller then stranger   2 segments, 2 speakers   trimmed correctly
#     stranger then caller   1 segment spanning the whole clip
#     stranger then caller,  2 segments, BOTH labelled speaker 1
#       with a wider gap
# In both failing shapes the stranger's words reached Whisper and the caller was told
# nothing had been removed. The segmentation model is order-sensitive and the
# clusterer merges freely, so a second opinion is taken from ECAPA — which is the
# model we actually tuned — before believing "one voice".
# --------------------------------------------------------------------------- #
class WindowGate(SpeakerGate):
    """Similarity is read straight off each chunk's peak, so a clip can be built with
    an exact per-second identity profile."""

    threshold = 0.55

    def __init__(self):
        self.backend = "stub"

    def embed(self, audio, min_seconds=0.6):
        import io
        import soundfile as sf
        data, _sr = (sf.read(io.BytesIO(audio)) if isinstance(audio, (bytes, bytearray))
                     else (np.asarray(audio), SR))
        data = np.asarray(data, dtype="float32").reshape(-1)
        if not data.size:
            return None
        peak = min(1.0, float(np.abs(data).mean()) * 10.0)   # mean → a blend of the parts
        return np.array([peak, np.sqrt(max(0.0, 1 - peak * peak))], dtype="float32")

    def similarity(self, a, b):
        a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if d == 0 else float(a @ b / d)


CALLER_LVL, OTHER_LVL = 0.085, 0.020        # mean*10 → ~0.85 and ~0.20 similarity
WPRINT = np.array([1.0, 0.0], dtype="float32")


def lvl(level, seconds):
    return np.full(int(seconds * SR), level, dtype="float32")


def test_a_stranger_the_clusterer_missed_is_still_removed():
    """The clusterer labels the whole clip speaker 1; ECAPA sees a sustained run that
    is not the caller. The stranger's audio must not reach Whisper."""
    audio = np.concatenate([lvl(OTHER_LVL, 3.5), lvl(CALLER_LVL, 3.5)])
    d = StubDiarizer([(0.0, 7.0, 0)])                  # ONE segment, ONE speaker
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert info.get("rescanned"), f"the second opinion never ran: {info}"
    assert out is not audio, "the stranger's half was kept"
    assert info["kept"] < info["total"] * 0.75


def test_one_caller_speaking_twice_is_never_split():
    """The dangerous direction. Her own pause produces a single low window; acting on
    it deleted 0.8s of her words on real audio. A turn is a RUN, not a dip."""
    audio = np.concatenate([lvl(CALLER_LVL, 3.0), lvl(CALLER_LVL * 0.55, 0.6),
                            lvl(CALLER_LVL, 3.4)])
    d = StubDiarizer([(0.0, 7.0, 0)])
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is audio, f"a single caller was split in two: {info}"
    assert info["reason"] == "single speaker"


def test_a_short_turn_is_never_rescanned():
    """A 3s answer cannot hold somebody else's whole turn, and re-embedding every
    short turn would cost far more than it saves."""
    audio = lvl(CALLER_LVL, 3.0)
    d = StubDiarizer([(0.0, 3.0, 0)])
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is audio and not info.get("rescanned")


def test_the_second_opinion_fails_open():
    """A gate that cannot embed must leave the clip exactly as it found it."""
    class Blind(WindowGate):
        def embed(self, audio, min_seconds=0.6):
            return None

    audio = np.concatenate([lvl(OTHER_LVL, 3.5), lvl(CALLER_LVL, 3.5)])
    out, info = keep_matching_speaker(StubDiarizer([(0.0, 7.0, 0)]), Blind(), WPRINT,
                                      audio, SR)
    assert out is audio


def test_the_rescan_thresholds_stay_coherent():
    from zensuvidha.diarize import (RESCAN_GAP, RESCAN_MARGIN, RESCAN_MIN_RUN,
                                    RESCAN_MIN_S, RESCAN_WIN_S)
    assert RESCAN_MIN_S >= 2 * RESCAN_WIN_S, "the window cannot exceed half the clip"
    assert RESCAN_MIN_RUN >= 2, "one dip is a pause, not a turn"
    assert 0 < RESCAN_MARGIN < RESCAN_GAP


def test_a_cluster_kept_only_by_the_gap_must_still_look_like_the_caller():
    """DROP_GAP exists because clustering once split ONE person into clusters scoring
    0.62 / 0.40 / 0.27. That fragmentation was an artifact of the old spliced trimming
    — repaired, the same caller across three sentences now returns ONE cluster at 0.94.
    So the gap no longer needs to reach 0.27, and reaching that far was keeping real
    intruders (measured at 0.31, sitting 0.37 below the caller)."""
    from zensuvidha.diarize import DROP_GAP, KEEP_FLOOR

    assert KEEP_FLOOR > 0, "the gap rule has no absolute floor"
    assert KEEP_FLOOR < 0.55, "the floor must stay below the accept threshold"

    class GapGate(WindowGate):
        def embed(self, audio, min_seconds=0.6):
            import io
            import soundfile as sf
            data, _sr = (sf.read(io.BytesIO(audio)) if isinstance(audio, (bytes, bytearray))
                         else (np.asarray(audio), SR))
            data = np.asarray(data, dtype="float32").reshape(-1)
            if not data.size:
                return None
            peak = min(1.0, float(np.abs(data).max()))
            return np.array([peak, np.sqrt(max(0.0, 1 - peak * peak))], dtype="float32")

    g = GapGate()
    ref = np.array([1.0, 0.0], dtype="float32")
    intruder = KEEP_FLOOR - 0.05          # inside DROP_GAP of the best, below the floor
    audio = np.concatenate([np.full(int(3.0 * SR), 0.80, dtype="float32"),
                            np.full(int(3.0 * SR), intruder, dtype="float32")])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 6.0, 1)])
    _out, info = keep_matching_speaker(d, g, ref, audio, SR)
    assert info["kept_speakers"] == 1, f"an intruder was kept by the gap rule: {info}"


def test_a_kept_cluster_that_is_itself_a_merge_is_trimmed_again():
    """The clusterer is not obliged to separate two people just because it found more
    than one group. Measured with three voices present: it returned two clusters, and
    the one scoring 0.83 as 'the caller' was the caller AND a stranger joined together,
    so trimming to it kept the stranger's whole sentence."""
    # The merged cluster must still SCORE as the caller — that is what makes it
    # dangerous. Caller-dominant so the blend stays above threshold, with a stranger's
    # turn inside it.
    audio = np.concatenate([lvl(CALLER_LVL, 5.0), lvl(OTHER_LVL, 2.0),
                            lvl(OTHER_LVL * 0.4, 3.0)])
    # the clusterer merges the caller with the FIRST stranger, and separates the second
    d = StubDiarizer([(0.0, 7.0, 0), (7.0, 10.0, 1)])
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is not audio
    assert info["kept"] < 6.0, f"the merged stranger survived the trim: {info}"


def test_a_short_but_genuine_turn_survives_the_ratio_guard():
    """A caller who answers briefly while others talk around them keeps very little of
    the clip. Refusing that handed Whisper the whole mixture instead — the isolation had
    found the right 1.5s and the safety rule threw it away."""
    from zensuvidha.diarize import MIN_KEEP_S

    audio = np.concatenate([lvl(CALLER_LVL, MIN_KEEP_S + 0.5), lvl(OTHER_LVL, 7.0)])
    d = StubDiarizer([(0.0, MIN_KEEP_S + 0.5, 0), (MIN_KEEP_S + 0.5, MIN_KEEP_S + 7.5, 1)])
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is not audio, f"the caller's short turn was discarded: {info}"
    assert info.get("kept_sim", 1.0) >= 0.55


def test_a_true_sliver_is_still_refused():
    """The guard must still catch a segmentation error — a fragment too short to be
    anybody's turn goes back as the original clip."""
    audio = np.concatenate([lvl(CALLER_LVL, 0.5), lvl(OTHER_LVL, 9.0)])
    d = StubDiarizer([(0.0, 0.5, 0), (0.5, 9.5, 1)])
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is audio
    assert "would keep only" in info["reason"]


# --------------------------------------------------------------------------- #
# what the inspector draws — the segments and the ranges that survived them
#
# The waveform in the UI is the caller's OWN recording, so every timestamp reported
# here has to be in THAT timeline. A boundary reported in the trimmed timeline slides
# left by however much was removed before it, and the highlight then sits over the
# wrong words — which reads as the isolation cutting the wrong person.
# --------------------------------------------------------------------------- #
def test_segments_are_reported_with_the_speaker_and_the_score():
    audio = np.concatenate([lvl(CALLER_LVL, 3.0), lvl(OTHER_LVL, 2.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 5.0, 1)])
    _out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    segs = info["segments"]
    assert [(s["s"], s["e"], s["spk"]) for s in segs] == [(0.0, 3.0, 0), (3.0, 5.0, 1)]
    assert segs[0]["keep"] and not segs[1]["keep"], "the stranger is drawn as kept"
    assert segs[0]["sim"] > segs[1]["sim"], "the scores are not attached per speaker"
    assert info["kept_spans"] == [(0.0, 3.0)]


def test_nobody_looked_is_not_the_same_as_one_segment():
    """None means diarization never ran. Drawing that as a single full-width kept
    segment would claim the audio was checked and cleared when it never was."""
    audio = lvl(CALLER_LVL, 3.0)
    _out, info = keep_matching_speaker(None, None, None, audio, SR)
    assert info["segments"] is None and info["kept_spans"] is None


def test_a_fail_open_never_draws_anything_as_removed():
    """The ratio guard hands the WHOLE clip to Whisper. If the drawing still showed the
    stranger's segment greyed out, the operator would be told something was filtered at
    the exact moment nothing was."""
    audio = np.concatenate([lvl(CALLER_LVL, 0.5), lvl(OTHER_LVL, 9.0)])
    d = StubDiarizer([(0.0, 0.5, 0), (0.5, 9.5, 1)])
    out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is audio
    assert info["kept_spans"] is None
    assert all(s["keep"] for s in info["segments"]), info["segments"]


def test_the_second_pass_reports_boundaries_in_the_original_timeline():
    """The rescan runs on the already-trimmed audio, where the gaps have been closed
    up. 1.0s into THAT is not 1.0s into the recording the caller is looking at."""
    from zensuvidha.diarize import _map_back

    # first pass kept 2-5s and 8-10s; the rescan then keeps 1-4s of that 5s result,
    # which is 3-5s and 8-9s of the original.
    assert _map_back([(2.0, 5.0), (8.0, 10.0)], [(1.0, 4.0)]) == [(3.0, 5.0), (8.0, 9.0)]
    # a rescan that keeps everything must map back to exactly what went in
    assert _map_back([(2.0, 5.0), (8.0, 10.0)], [(0.0, 5.0)]) == [(2.0, 5.0), (8.0, 10.0)]


def test_the_merged_cluster_rescan_moves_the_drawn_region_too():
    """End to end through the path the mapping exists for: the second trim must be
    visible in the drawing, and still inside the clip."""
    audio = np.concatenate([lvl(CALLER_LVL, 5.0), lvl(OTHER_LVL, 2.0),
                            lvl(OTHER_LVL * 0.4, 3.0)])
    d = StubDiarizer([(0.0, 7.0, 0), (7.0, 10.0, 1)])
    _out, info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    spans = info["kept_spans"]
    assert spans, info
    assert sum(b - a for a, b in spans) < 6.0, f"the second trim is not drawn: {spans}"
    assert all(0.0 <= a < b <= info["total"] + 1e-6 for a, b in spans), spans


def test_isolation_uses_the_bar_the_gate_actually_learned():
    """Reading `gate.threshold` directly meant isolation kept judging against the
    synthetic 0.55 while the gate itself had adapted to 0.11 for this microphone. Every
    cluster then scored "not the caller", nothing was ever trimmed, and switching Voice
    Isolation on did precisely nothing — the feature was silently inert on exactly the
    hardware it was needed for."""
    # The REAL numbers from one live call, not the stub's synthetic 0.85/0.20: the
    # caller measured 0.21-0.26 and a video playing in the room -0.06 to 0.05. Modelled
    # directly, because the whole point is a microphone where every absolute constant in
    # this file is above where the caller lands.
    class RealMic(WindowGate):
        threshold = 0.55

        def effective_threshold(self):
            return 0.11                      # what the gate learned on this call

        def similarity(self, a, b):
            # first cluster is the caller, second is the room
            mid = (CALLER_LVL + OTHER_LVL) / 2
            return 0.24 if float(np.asarray(b)[0]) > mid else 0.05

        def embed(self, audio, min_seconds=0.6):
            import io

            import soundfile as sf
            data, _sr = sf.read(io.BytesIO(audio)) if isinstance(audio, (bytes, bytearray)) \
                else (np.asarray(audio), SR)
            data = np.asarray(data, dtype="float32").reshape(-1)
            return None if not data.size else np.array([float(np.max(np.abs(data)))])

    audio = np.concatenate([lvl(CALLER_LVL, 3.0), lvl(OTHER_LVL, 2.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 5.0, 1)])
    out, info = keep_matching_speaker(d, RealMic(), np.array([1.0]), audio, SR)
    assert out is not audio, "nothing was trimmed: %s" % info
    assert info["kept_spans"] == [(0.0, 3.0)], (
        "the room was kept alongside the caller: %s" % info)


def test_a_gate_without_the_learned_bar_still_works():
    """speaker.py is pluggable — a custom gate that predates the adaptive threshold must
    keep working on its configured value rather than raising."""
    audio = np.concatenate([lvl(CALLER_LVL, 3.0), lvl(OTHER_LVL, 2.0)])
    d = StubDiarizer([(0.0, 3.0, 0), (3.0, 5.0, 1)])
    out, _info = keep_matching_speaker(d, WindowGate(), WPRINT, audio, SR)
    assert out is not audio
