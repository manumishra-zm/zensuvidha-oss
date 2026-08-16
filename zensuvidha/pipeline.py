"""The audio front-end, as one ordered pipeline.

Three modules touch the caller's audio — DeepFilterNet (denoise), diarization +
ECAPA (isolate the caller), and Whisper (recognise). They used to be independent
switches, which let the caller pick combinations that measurably hurt. This module
owns the order, and the order is not arbitrary — every step below is a measurement
made on this codebase:

    what we measured                              what it implies
    ------------------------------------------    -----------------------------
    instrumental music at EQUAL loudness left      music is not the enemy;
      Whisper's WER unchanged (0.27 -> 0.27)         don't spend 500ms fighting it
    DeepFilter on clean speech: WER 0.10 vs 0.00   denoising a quiet room makes
      raw                                            recognition WORSE
    DeepFilter on the voiceprint: 0.596 vs 0.675   never identify anyone from
      raw                                            denoised audio
    DeepFilter for the VAD: 110.60 vs 1.99         but for DECIDING whether there
      separation (55x)                               is speech, it is superb
    a second voice drags whole-clip similarity     a voice can only be removed by
      0.867 -> 0.34                                  SEPARATING speakers

DeepFilterNet is a speech *enhancer*: it keeps speech and removes everything else.
A singer on a background track IS speech, so it protects the singer too — no amount
of denoising removes a voice. That is why the caller's own observation ("deepfilter
and isolation together worked properly") is exactly right and worth encoding as the
default: the two modules remove different halves of a song, and neither can do the
other's half.

    caller + band + singer
        │
        ├─ isolate  (on RAW — identity is measurably worse on denoised audio)
        │      removes the SINGER: a different voice, clustered and dropped
        │
        └─ denoise  (only when the room warrants it — see should_denoise)
               removes the BAND: broadband non-speech energy

Everything fails open. A missing model, a broken clip, an exception anywhere —
the caller's original audio goes through untouched. Dropping a real turn is a far
worse failure than passing one noisy turn to Whisper.
"""
from __future__ import annotations

import io
import logging
import math

log = logging.getLogger(__name__)

SR = 16000

# ---- the router -----------------------------------------------------------
# Denoising costs ~500ms per turn on CPU and measurably HURTS a clean recording,
# so it has to earn its place turn by turn. The decision is made on a free
# numpy measurement of the recording itself, never by running the denoiser first
# (that would cost the very thing we are trying to avoid).
FRAME_MS = 20
QUIET_PCTL = 15          # the between-words floor
LOUD_PCTL = 85           # the voice itself
DENOISE_BELOW_DB = 10.0  # floor-to-voice gap under this = a room worth cleaning.
                         #   Measured on this codebase (floor-to-voice, dB):
                         #     clean speech            18.0   skip — denoising it
                         #     music at -18dB          16.1     raised WER 0.00 -> 0.10
                         #     music at -12dB          12.8   skip — WER measured unchanged
                         #     music at  -6dB           8.4   clean it
                         #     music at   0dB           4.7   clean it
                         #     continuous hiss          0.7   clean it
                         #   The cut sits in the empty band between 12.8 and 8.4, so it
                         #   errs toward RAW — the version measured best whenever the
                         #   recording is good enough not to need help.
HYSTERESIS_DB = 3.0      # ...and it must swing this far back before we stop again
MIN_ROUTE_S = 0.4        # shorter than this and the percentiles mean nothing


def room_snr_db(data) -> float | None:
    """Gap between the quiet parts of the clip and the loud parts, in dB.

    A clean recording has near-silence between words, so the gap is wide. Anything
    continuous — traffic, a fan, a band — raises the floor and closes it. This is a
    property of the ROOM rather than of any one word, which is what makes it a fair
    basis for deciding whether to clean the audio at all.
    """
    import numpy as np

    x = np.asarray(data, dtype="float32").reshape(-1)
    if x.size < int(MIN_ROUTE_S * SR):
        return None
    hop = max(1, SR * FRAME_MS // 1000)
    n = x.size // hop
    if n < 8:
        return None
    frames = x[: n * hop].reshape(n, hop).astype("float64")
    rms = np.sqrt((frames ** 2).mean(axis=1))
    quiet = float(np.percentile(rms, QUIET_PCTL))
    loud = float(np.percentile(rms, LOUD_PCTL))
    if loud <= 0:
        return None
    if quiet <= 1e-7:                     # digital silence between words: as clean as it gets
        return 90.0
    return float(20.0 * math.log10(loud / quiet))


# AUTO-DENOISE IS OFF, and this is measured rather than cautious. Swept across six
# interference types at five SNRs each — 30 conditions — DeepFilterNet won ONE cell,
# lost EIGHT and tied the rest, for a net word-recall of -400%:
#
#     white hiss    +0  -33  -33  -50  -50      fan/AC       +0  +17   +0  -50  -67
#     traffic        0    0    0    0    0      mains hum     0    0    0    0    0
#     music instr    0    0    0    0    0      music vocals +0   +0  -33 -100   +0
#
# It is also the most expensive stage in the pipeline (226-475ms, scaling with length).
# So on the recognition path it costs time AND accuracy, which is the same conclusion
# `noisereduce` reached before it. Whisper trained on 680k hours of messy audio and does
# not need the help.
#
# The UI toggle still forces it on for A/B, and `stt.auto_denoise: true` re-enables the
# router for anyone whose room genuinely differs from these measurements.
AUTO_DENOISE = False


def should_denoise(data, previous: bool = False) -> tuple[bool, float | None]:
    """Is this recording noisy enough to be worth cleaning?

    `previous` is the last verdict on this call; the threshold moves by
    HYSTERESIS_DB against it so a room sitting near the line does not flip the
    pipeline on and off every turn (which would make latency visibly erratic).
    """
    snr = room_snr_db(data)
    if not AUTO_DENOISE:
        return False, snr          # still MEASURE the room — the inspector reports it
    if snr is None:
        return previous, None
    limit = DENOISE_BELOW_DB + (HYSTERESIS_DB if previous else 0.0)
    return snr < limit, snr


# ---- the pipeline ---------------------------------------------------------
def prepare(raw, *, decode, denoiser=None, diarizer=None, gate=None, voiceprint=None,
            denoise_mode=None, isolate_mode=None, was_denoised=False):
    """Run the caller's audio through isolation and denoising, in that order.

    `decode(raw) -> float32 ndarray @16k` is injected so this module never depends
    on a particular STT provider.

    `denoise_mode` / `isolate_mode`: True forces on, False forces off, None = auto
    (isolation runs whenever it can; denoising runs when the room needs it).

    Returns (audio_for_whisper, info). `audio_for_whisper` is the ORIGINAL `raw`
    bytes whenever nothing was changed, so the common clean single-speaker turn
    costs one decode and no re-encode.
    """
    # `speakers` is None until diarization has actually COUNTED them. "One voice" and
    # "nobody looked" are different answers, and the speaker gate needs to tell them
    # apart before it trusts a clip enough to learn the caller's identity from it.
    info = {"isolated": False, "denoised": False, "snr_db": None,
            "speakers": None, "kept_s": None, "total_s": None, "reason": None,
            "iso_ms": 0.0, "den_ms": 0.0,
            # Where the boundaries were and what survived them, for the inspector's
            # waveform. None means diarization never ran on this turn.
            "segments": None, "kept_spans": None}
    try:
        import time

        import numpy as np
        data = decode(raw)
        if data is None:
            return raw, info
        data = np.asarray(data, dtype="float32").reshape(-1)
        if not data.size:
            return raw, info
        info["total_s"] = round(data.size / SR, 2)
        work, changed = data, False

        # Measure the room on EVERY turn, before anything else touches the signal.
        # It used to be computed inside the denoise branch, so a machine without the
        # DeepFilter binary — or with the toggle off, which is the default — never
        # measured at all: the inspector showed no room reading, and the caller got the
        # same flat "could you say that again?" whether they were in a quiet office or
        # standing next to a fan. Free either way: two numpy percentiles.
        info["snr_db"] = None if (snr0 := room_snr_db(data)) is None else round(snr0, 1)

        # ---- 1. isolate, on RAW ------------------------------------------
        # Identity is measurably better on unprocessed audio (0.675 vs 0.596), and
        # this is the ONLY step that can remove another human voice. It therefore
        # goes first, before anything has been altered.
        if isolate_mode is not False and diarizer is not None and gate is not None \
                and voiceprint is not None:
            t0 = time.time()
            try:
                from .diarize import keep_matching_speaker
                trimmed, dia = keep_matching_speaker(diarizer, gate, voiceprint, work, SR)
                info["speakers"] = dia.get("speakers", 1)   # counted, for real
                info["reason"] = dia.get("reason")
                info["kept_s"] = round(dia.get("kept", 0.0), 2)
                info["segments"] = dia.get("segments")
                info["kept_spans"] = dia.get("kept_spans")
                if trimmed is not work and (info["speakers"] or 1) > 1:
                    work, changed, info["isolated"] = (
                        np.asarray(trimmed, dtype="float32").reshape(-1), True, True)
            except Exception as e:  # noqa: BLE001
                log.warning("isolation failed (%s) — keeping the whole utterance", e)
            info["iso_ms"] = round((time.time() - t0) * 1000, 1)

        # ---- 2. denoise, on what is left ---------------------------------
        # Second, because it must not be allowed to alter the audio the speaker
        # decisions above are made from. Skipped entirely on a clean recording:
        # it costs ~500ms and raised WER from 0.00 to 0.10 there.
        if denoiser is not None and denoise_mode is not False:
            want, snr = (True, info["snr_db"]) if denoise_mode is True \
                else should_denoise(work, was_denoised)
            if snr is not None:
                info["snr_db"] = round(snr, 1)
            if want:
                t0 = time.time()
                try:
                    out = np.asarray(denoiser(work, SR), dtype="float32").reshape(-1)
                    if out.size:
                        work, changed, info["denoised"] = out, True, True
                except Exception as e:  # noqa: BLE001
                    log.warning("denoise failed (%s) — using the audio as recorded", e)
                info["den_ms"] = round((time.time() - t0) * 1000, 1)

        if not changed:
            return raw, info
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, work, SR, format="WAV", subtype="PCM_16")
        return buf.getvalue(), info
    except Exception as e:  # noqa: BLE001
        log.warning("audio pipeline failed (%s) — passing the recording through", e)
        return raw, info
