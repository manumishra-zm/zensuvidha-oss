"""The audio front-end as one ordered pipeline.

Two things are being pinned here:

  1. ORDER. Isolation runs on RAW and denoising runs after it. Both facts are
     measured, both are easy to reverse by accident, and reversing either one
     degrades the system silently — the transcript still arrives, just worse.
  2. FAIL-OPEN. Every branch must hand back the caller's own audio. A denoiser
     that throws, a diarizer that returns nonsense, a clip too short to measure:
     none of them may cost a turn.

Run:  pytest -q tests/test_pipeline.py
"""
import io

import numpy as np
import pytest
import soundfile as sf

from zensuvidha import pipeline
from zensuvidha.pipeline import (DENOISE_BELOW_DB, HYSTERESIS_DB, prepare,
                                 room_snr_db, should_denoise)

SR = 16000


def speech(seconds=2.0, gaps=True):
    """Voice-like: loud bursts with near-silence between them."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = 0.4 * np.sin(2 * np.pi * 160 * t) + 0.2 * np.sin(2 * np.pi * 480 * t)
    if gaps:                                   # word gaps every 250ms
        env = (np.sin(2 * np.pi * 2 * t) > 0).astype("float32")
        x = x * env
    return x.astype("float32")


def noisy(seconds=2.0, level=0.25):
    return (speech(seconds) + np.random.default_rng(0).normal(size=int(seconds * SR))
            * level).astype("float32")


def wav(x):
    b = io.BytesIO()
    sf.write(b, np.asarray(x, dtype="float32"), SR, format="WAV", subtype="PCM_16")
    return b.getvalue()


def decode(raw):
    d, _ = sf.read(io.BytesIO(raw), dtype="float32")
    return d




@pytest.fixture
def router_on():
    """Auto-denoise ships OFF, on measurement: across 6 interference types x 5 SNRs,
    DeepFilter won 1 cell, lost 8 and tied 21 — it costs accuracy as well as 226-475ms.
    Tests that exercise the ROUTER'S JUDGEMENT still need it switched on."""
    old, pipeline.AUTO_DENOISE = pipeline.AUTO_DENOISE, True
    yield
    pipeline.AUTO_DENOISE = old

# --------------------------------------------------------------------------- #
# the router
# --------------------------------------------------------------------------- #
def test_a_clean_recording_is_left_alone():
    """Denoising a clean turn measurably HURT it (WER 0.00 -> 0.10) and costs
    ~500ms. The router's whole job is to not pay that for nothing."""
    want, snr = should_denoise(speech())
    assert not want, f"clean speech routed to the denoiser (snr={snr})"
    assert snr > DENOISE_BELOW_DB


def test_a_noisy_recording_is_cleaned(router_on):
    want, snr = should_denoise(noisy())
    assert want, f"noise left uncleaned (snr={snr})"
    assert snr < DENOISE_BELOW_DB


def test_the_router_does_not_flip_every_turn(router_on):
    """A room sitting near the threshold must not toggle a 500ms stage on and off
    turn after turn — the caller experiences that as random latency."""
    borderline = None
    for level in np.arange(0.02, 0.40, 0.01):
        _, snr = should_denoise(noisy(level=float(level)))
        if snr is not None and abs(snr - DENOISE_BELOW_DB) < 1.0:
            borderline = float(level)
            break
    assert borderline is not None, "no borderline room found to test with"
    clip = noisy(level=borderline)
    on_prev, _ = should_denoise(clip, previous=True)
    off_prev, _ = should_denoise(clip, previous=False)
    assert on_prev, "a clip just under the line stopped being cleaned"
    assert not off_prev, "hysteresis leaked in both directions"


def test_a_clip_too_short_to_judge_keeps_the_last_verdict(router_on):
    tiny = np.zeros(int(0.1 * SR), dtype="float32")
    assert should_denoise(tiny, previous=True) == (True, None)
    assert should_denoise(tiny, previous=False) == (False, None)


def test_digital_silence_between_words_reads_as_clean():
    x = speech(gaps=True)
    x[np.abs(x) < 1e-6] = 0.0
    assert room_snr_db(x) > DENOISE_BELOW_DB


def test_the_thresholds_leave_room_for_hysteresis():
    assert HYSTERESIS_DB > 0
    assert DENOISE_BELOW_DB > HYSTERESIS_DB


# --------------------------------------------------------------------------- #
# order — the measured reason this module exists
# --------------------------------------------------------------------------- #
class Spy:
    """Records what each stage was handed, so ORDER can be asserted."""

    def __init__(self):
        self.seen = []

    def denoiser(self, data, sr):
        self.seen.append(("denoise", np.asarray(data).copy()))
        return np.asarray(data, dtype="float32") * 0.5

    def diarizer_factory(self, segs):
        spy = self

        class D:
            def segments(self, samples, sr=SR):
                spy.seen.append(("isolate", np.asarray(samples).copy()))
                return list(segs)
        return D()


class Gate:
    """Identity is read off the clip's peak, so 'the caller' and 'someone else' are
    exact and a cluster can actually be rejected."""
    threshold = 0.55

    def embed(self, audio, min_seconds=0.6):
        data = (decode(audio) if isinstance(audio, (bytes, bytearray))
                else np.asarray(audio))
        data = np.asarray(data, dtype="float32").reshape(-1)
        if not data.size:
            return None
        peak = min(1.0, float(np.abs(data).max()))
        return np.array([peak, np.sqrt(max(0.0, 1 - peak * peak))], dtype="float32")

    def similarity(self, a, b):
        a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if d == 0 else float(a @ b / d)


CALLER_PRINT = np.array([1.0, 0.0], dtype="float32")


def two_voices(seconds=2.0):
    """The caller (loud) followed by someone else (quiet) — two clusters that a
    peak-reading gate separates cleanly."""
    return np.concatenate([noisy(seconds), noisy(seconds) * 0.25]).astype("float32")


def test_isolation_runs_before_denoising_and_sees_the_raw_audio():
    """Identity is measurably worse on denoised audio (0.596 vs 0.675). If this
    order ever flips, the voiceprint starts judging audio DeepFilter has already
    reshaped and the caller gets locked out of their own call."""
    spy = Spy()
    raw = two_voices()
    # denoise_mode=True: the ORDER is what this pins, so the stage has to run at all.
    out, info = prepare(
        wav(raw), decode=decode, denoiser=spy.denoiser, denoise_mode=True,
        diarizer=spy.diarizer_factory([(0.0, 2.0, 0), (2.0, 4.0, 1)]),
        gate=Gate(), voiceprint=CALLER_PRINT)
    stages = [name for name, _ in spy.seen]
    assert stages == ["isolate", "denoise"], f"wrong order: {stages}"
    assert info["isolated"] and info["denoised"]
    seen_by_isolate = dict(spy.seen)["isolate"]
    assert np.allclose(seen_by_isolate[:100], raw[:100], atol=2e-4), \
        "isolation was handed audio that had already been altered"


def test_denoising_sees_only_what_isolation_kept():
    spy = Spy()
    out, info = prepare(
        wav(two_voices()), decode=decode, denoiser=spy.denoiser, denoise_mode=True,
        diarizer=spy.diarizer_factory([(0.0, 2.0, 0), (2.0, 4.0, 1)]),
        gate=Gate(), voiceprint=CALLER_PRINT)
    by = dict(spy.seen)
    assert len(by["denoise"]) < len(by["isolate"]), \
        "the denoiser was handed the untrimmed clip"


def test_a_clean_single_speaker_turn_costs_nothing():
    """The overwhelmingly common case: no re-encode, no denoise, same bytes back."""
    spy = Spy()
    raw = wav(speech())
    out, info = prepare(raw, decode=decode, denoiser=spy.denoiser,
                        diarizer=spy.diarizer_factory([(0.0, 2.0, 0)]),
                        gate=Gate(), voiceprint=CALLER_PRINT)
    assert out is raw, "a clean single-speaker turn was re-encoded for no reason"
    assert not info["isolated"] and not info["denoised"]
    assert "denoise" not in [n for n, _ in spy.seen]


# --------------------------------------------------------------------------- #
# the toggles
# --------------------------------------------------------------------------- #
def test_forcing_denoise_on_overrides_the_router():
    spy = Spy()
    out, info = prepare(wav(speech()), decode=decode, denoiser=spy.denoiser,
                        denoise_mode=True)
    assert info["denoised"], "the UI toggle did not override a clean-room verdict"


def test_forcing_denoise_off_is_absolute():
    spy = Spy()
    out, info = prepare(wav(noisy()), decode=decode, denoiser=spy.denoiser,
                        denoise_mode=False)
    assert not info["denoised"] and out is not None
    assert not spy.seen


def test_forcing_isolation_off_is_absolute():
    spy = Spy()
    out, info = prepare(wav(two_voices()), decode=decode,
                        diarizer=spy.diarizer_factory([(0.0, 2.0, 0), (2.0, 4.0, 1)]),
                        gate=Gate(), voiceprint=CALLER_PRINT,
                        isolate_mode=False)
    assert not info["isolated"]
    assert not spy.seen


# --------------------------------------------------------------------------- #
# fail-open — none of these may cost a turn
# --------------------------------------------------------------------------- #
def test_a_denoiser_that_throws_never_loses_the_turn():
    def boom(data, sr):
        raise RuntimeError("model file is corrupt")

    raw = wav(noisy())
    out, info = prepare(raw, decode=decode, denoiser=boom)
    assert out is raw
    assert not info["denoised"]


def test_a_diarizer_that_throws_never_loses_the_turn():
    class Boom:
        def segments(self, samples, sr=SR):
            raise RuntimeError("onnx session died")

    raw = wav(speech())
    out, info = prepare(raw, decode=decode, diarizer=Boom(), gate=Gate(),
                        voiceprint=np.ones(4, dtype="float32"))
    assert out is raw
    assert not info["isolated"]


def test_a_denoiser_returning_nothing_never_loses_the_turn():
    raw = wav(noisy())
    out, info = prepare(raw, decode=decode,
                        denoiser=lambda d, sr: np.zeros(0, dtype="float32"))
    assert out is raw and not info["denoised"]


@pytest.mark.parametrize("junk", [b"", b"\x00\x01", b"RIFFbroken", b"not audio at all"])
def test_unreadable_audio_is_handed_straight_back(junk):
    out, info = prepare(junk, decode=decode, denoiser=lambda d, sr: d)
    assert out is junk


def test_a_decoder_returning_none_is_handled():
    raw = wav(speech())
    out, info = prepare(raw, decode=lambda _r: None, denoiser=lambda d, sr: d)
    assert out is raw


def test_with_no_modules_at_all_it_is_a_no_op():
    raw = wav(noisy())
    out, info = prepare(raw, decode=decode)
    assert out is raw
    assert not info["isolated"] and not info["denoised"]


# --------------------------------------------------------------------------- #
# wiring into the session
# --------------------------------------------------------------------------- #
def test_the_session_never_denoises_twice():
    """pipeline.prepare has already denoised by the time transcribe() runs. Letting
    the STT provider's own switch stay live ran DeepFilterNet a second time on its
    own output — ~500ms of pure waste per turn, and a second pass of suppression."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack

    seen = {}

    class STT:
        denoiser = None

        def _decode(self, a):
            return decode(a)

        def transcribe(self, audio, **kw):
            seen.update(kw)
            return "ok", "en", 0.9

    s = Session(load_pack("clinic"), None, stt=STT())
    s.stt_denoise = True                       # UI toggle forced ON
    s.transcribe(wav(speech()))
    assert seen.get("denoise") is False, \
        f"transcribe re-denoised audio the pipeline had already cleaned: {seen}"


def test_clean_audio_survives_a_session_with_no_stt():
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack

    s = Session(load_pack("clinic"), None)
    raw = b"whatever"
    assert s.clean_audio(raw) == (raw, None)


def test_isolate_caller_still_works_as_a_name():
    """Older call sites use the previous name; it must not have become a no-op."""
    from zensuvidha.orchestrator import Session
    assert Session.isolate_caller is not None
    assert Session.clean_audio.__doc__
