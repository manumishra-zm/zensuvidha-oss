"""The seam between a call and the thing carrying it.

The engine's audio stages — isolation, the speaker gate, the guard, the router — are
already transport-agnostic. What was missing was somewhere for a second transport to
plug in without touching them, and a place to put the things a browser supplies for free
and a phone line does not: echo handling, voice activity, and endpointing.
"""
import asyncio

import numpy as np
import pytest

from zensuvidha import transport as T


# ── format conversion, which is where a telephony bug would actually live ──────

def test_ulaw_round_trips_close_enough_for_speech():
    """μ-law is lossy by design — 8 bits, logarithmic. What matters is that a round trip
    does not destroy the signal the recogniser and the voiceprint depend on."""
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    back = T.ulaw_to_pcm(T.pcm_to_ulaw(sig))
    assert back.shape == sig.shape
    err = float(np.abs(back - sig).max())
    assert err < 0.05, f"μ-law round trip lost {err:.3f}"
    corr = float(np.corrcoef(back, sig)[0, 1])
    assert corr > 0.999, f"shape not preserved (r={corr:.4f})"


def test_resampling_8k_to_16k_and_back_preserves_the_speech_band():
    sr_in = 8000
    t = np.linspace(0, 1, sr_in, endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 300 * t)).astype("float32")
    up = T.resample(sig, 8000, 16000)
    assert up.size == pytest.approx(sig.size * 2, rel=0.01)
    down = T.resample(up, 16000, 8000)
    assert down.size == pytest.approx(sig.size, rel=0.01)
    n = min(down.size, sig.size)
    assert float(np.corrcoef(down[:n], sig[:n])[0, 1]) > 0.99


def test_resampling_is_a_no_op_at_the_same_rate():
    x = np.linspace(-1, 1, 100, dtype="float32")
    assert np.array_equal(T.resample(x, 16000, 16000), x)


@pytest.mark.parametrize("empty", [np.zeros(0, dtype="float32"), []])
def test_resampling_survives_an_empty_frame(empty):
    assert T.resample(empty, 8000, 16000).size == 0


# ── the endpointer a carrier does not give you ─────────────────────────────────

def _speech(ms, sr=16000, level=0.3):
    n = int(sr * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    return (level * np.sin(2 * np.pi * 200 * t)).astype("float32")


def _silence(ms, sr=16000):
    return np.zeros(int(sr * ms / 1000), dtype="float32")


def test_a_short_answer_closes_on_the_short_window():
    """A one-word answer is finished the moment it stops. Making it wait for the long
    window is most of what makes a call feel slow — the same rule the browser learned."""
    ep = T.Endpointer()
    out = None
    for frame in [_speech(300)] + [_silence(100)] * 12:
        got = ep.feed(frame)          # `or` would truth-test a numpy array
        if got is not None and out is None:
            out = got
    assert out is not None, "a short turn never closed"
    assert out.size / 16000 < 1.4


def test_a_long_utterance_gets_the_longer_window():
    """An utterance long enough to HAVE a middle waits longer, because that is where
    people stop to find a word."""
    ep = T.Endpointer()
    closed_at = None
    fed = 0.0
    for frame in [_speech(1800)] + [_silence(100)] * 15:
        fed += frame.size / 16000 * 1000
        if ep.feed(frame) is not None and closed_at is None:
            closed_at = fed
    assert closed_at is not None
    # The ladder now comes from zensuvidha/turn.py, shared with the browser — this
    # class used to carry its own copy, which had drifted to a flat 800/1200ms with no
    # learned pause and no per-slot extra.
    from zensuvidha import turn as _turn
    assert closed_at > 1800 + _turn.NORMAL["base_ms"], (
        "a long utterance was cut on the short window")


def test_a_pause_mid_sentence_does_not_end_the_turn():
    ep = T.Endpointer()
    for frame in [_speech(1000), _silence(300), _speech(1000)]:
        assert ep.feed(frame) is None, "the turn ended on a pause they spoke through"


def test_steady_noise_cannot_latch_it_open_forever():
    """The browser needed MAX_UTT_MS for exactly this: background music never gives a
    real silence, so the buffer grew until the frame was discarded and the caller's turn
    vanished with no reply."""
    ep = T.Endpointer()
    out = None
    for _ in range(400):
        got = ep.feed(_speech(100))
        if got is not None:
            out = got
            break
    assert out is not None, "a continuous stream never produced an utterance"
    assert out.size / 16000 <= T.Endpointer.MAX_UTT_MS / 1000 + 0.2


def test_silence_alone_produces_nothing():
    ep = T.Endpointer()
    for _ in range(50):
        assert ep.feed(_silence(100)) is None
    assert ep.flush() is None


def test_a_supplied_vad_is_used_instead_of_the_energy_fallback():
    """A real deployment passes Silero in; the energy gate only has to stop a silent
    line looking like speech."""
    calls = []

    def vad(frame, floor):
        calls.append(frame.size)
        return True
    ep = T.Endpointer(is_speech=vad)
    ep.feed(_speech(100))
    assert calls, "the supplied VAD was never consulted"


# ── the adapter itself ─────────────────────────────────────────────────────────

def test_the_module_imports_without_pipecat_installed():
    """Pipecat is optional. Importing this must never take the browser path down."""
    import importlib
    importlib.reload(T)
    assert hasattr(T, "PipecatTransport")


def test_constructing_the_adapter_without_pipecat_says_what_to_install():
    pytest.importorskip  # noqa: B018
    try:
        import pipecat  # noqa: F401
        pytest.skip("pipecat is installed here")
    except ImportError:
        pass
    with pytest.raises(ImportError) as e:
        T.PipecatTransport(carrier_stream=None)
    assert "pip install pipecat-ai" in str(e.value)


def test_the_transport_contract_is_the_whole_surface():
    """Four methods. Everything above the seam — isolation, the gate, the guard, the
    router — stays independent of where the audio came from. A fifth method here would
    mean something transport-specific had leaked into the engine."""
    required = {"recv_audio", "send_audio", "send_text", "hangup"}
    assert required <= set(dir(T.Transport))
    assert required <= set(dir(T.PipecatTransport))


def test_pipecat_is_used_for_transport_only():
    """It implements no diarization and no voice isolation, so wiring its STT/LLM/TTS
    would trade away the parts of this codebase that took the most work."""
    import inspect
    src = inspect.getsource(T.PipecatTransport)
    for service in ("STTService", "LLMService", "TTSService", "PipelineTask",
                    "OpenAI", "Deepgram", "Cartesia", "ElevenLabs"):
        assert service not in src, f"{service} leaked into the transport adapter"


def test_our_ulaw_codec_matches_the_stdlib_reference():
    """`audioop` was removed in Python 3.13, so this codec is hand-written — which
    means it needs checking against the implementation it replaced, while that still
    exists. A wrong exponent search made the first version lose 0.40 of full scale.
    """
    audioop = pytest.importorskip("audioop")
    import warnings
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 220 * t)).astype("float32")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = audioop.lin2ulaw(
            (np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes(), 2)
    mine = T.pcm_to_ulaw(sig)

    same = sum(a == b for a, b in zip(mine, theirs))
    assert same / len(theirs) > 0.98, (
        f"only {100*same/len(theirs):.1f}% of bytes match the reference encoder")

    # and our decoder must read the reference encoder's bytes correctly
    assert float(np.abs(T.ulaw_to_pcm(theirs) - sig).max()) < 0.02


def test_the_codec_is_accurate_across_full_scale():
    full = np.linspace(-1, 1, 4096).astype("float32")
    back = T.ulaw_to_pcm(T.pcm_to_ulaw(full))
    assert float(np.abs(back - full).max()) < 0.03


def test_no_stdlib_audioop_dependency():
    """It is gone in 3.13. The telephony path is the one part of this aimed squarely at
    the future; it must not be the first thing to break on a modern interpreter."""
    import inspect
    assert "import audioop" not in inspect.getsource(T)


# ── LiveKit ───────────────────────────────────────────────────────────────────
# Adopted for TRANSPORT only, on the same reasoning as Pipecat. What makes LiveKit
# worth having at all is that its server is Apache-2.0 and self-hostable, which the
# offline premise requires — and what makes it only a transport is that the one
# feature overlapping this codebase (background voice cancellation) is a Krisp model
# available exclusively through LiveKit Cloud.

class _FakeFrame:
    def __init__(self, pcm, sr=48000, ch=1):
        import numpy as np
        self.data = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
        self.sample_rate, self.num_channels = sr, ch
        self.samples_per_channel = len(pcm) // ch


class _FakeEvent:
    def __init__(self, frame):
        self.frame = frame


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()


class _FakeSource:
    def __init__(self):
        self.frames = []

    async def capture_frame(self, frame):
        self.frames.append(frame)


def _lk(stream, source=None, room=None):
    return T.LiveKitTransport(stream, audio_source=source, room=room, require_sdk=False)


def test_livekit_constructing_without_the_sdk_says_what_to_install():
    try:
        import livekit.rtc  # noqa: F401
        pytest.skip("the livekit sdk is installed here")
    except ImportError:
        pass
    with pytest.raises(ImportError) as e:
        T.LiveKitTransport(audio_stream=None)
    assert "pip install livekit" in str(e.value)


def test_livekit_meets_the_same_four_method_contract():
    required = {"recv_audio", "send_audio", "send_text", "hangup"}
    assert required <= set(dir(T.LiveKitTransport))


def test_livekit_is_used_for_transport_only():
    """Its turn detector, its noise cancellation and its agent loop are deliberately
    not wired in — the first is SDK-bound and has no Telugu, the second is Cloud-only
    and denoising is already measured to hurt recognition here."""
    import inspect
    # The class docstring NAMES these, because saying why each was rejected is the
    # point of it. So check the code, not the prose — otherwise the only way to pass
    # is to stop explaining the decision.
    src = inspect.getsource(T.LiveKitTransport)
    # __doc__, not getdoc(): getdoc dedents, so it no longer matches the source text
    # and the strip silently does nothing — which is how this test first "passed".
    code = src.replace(T.LiveKitTransport.__doc__ or "", "")
    assert "Krisp" not in code, "the docstring was not actually stripped"
    for service in ("noise_cancellation", "turn_detector", "AgentSession",
                    "STTService", "LLMService", "TTSService", "Krisp"):
        assert service not in code, f"{service} leaked into the transport adapter"


def test_livekit_frames_arrive_as_16k_utterances():
    """WebRTC is 48k and everything above the seam is 16k. Getting this wrong does not
    raise — it hands Whisper audio at three times the speed, which transcribes as
    nonsense rather than as an error."""
    import numpy as np
    sr = 48000
    speech = (np.sin(2 * np.pi * 180 * np.arange(int(1.2 * sr)) / sr) * 0.5).astype("float32")
    silence = np.zeros(int(1.5 * sr), dtype="float32")
    tp = _lk(_FakeStream([_FakeEvent(_FakeFrame(speech, sr)),
                          _FakeEvent(_FakeFrame(silence, sr))]))

    async def go():
        return [u async for u in tp.recv_audio()]
    utts = asyncio.run(go())
    assert utts, "no utterance was produced"
    # 2.7s went in (1.2 speech + 1.5 trailing silence, which the endpointer keeps up to
    # its window). At 16k that is ~43k samples; had the resample been skipped it would
    # be ~130k. The point of the bound is to tell those two apart.
    assert 0.9 * 16000 <= len(utts[0]) <= 3.0 * 16000, len(utts[0])

def test_livekit_stereo_is_mixed_down_rather_than_interleaved():
    """Two channels read as one array is a signal at double speed with a comb filter
    on it. Silent, and fatal to both the transcript and the voiceprint."""
    import numpy as np
    n = 16000
    left = np.full(n, 0.5, dtype="float32")
    stereo = np.empty(n * 2, dtype="float32")
    stereo[0::2], stereo[1::2] = left, -left           # perfectly out of phase
    pcm, sr = T.LiveKitTransport._frame_pcm(_FakeEvent(_FakeFrame(stereo, 48000, ch=2)))
    assert sr == 48000
    assert len(pcm) == n, "channels were not mixed down"
    assert abs(float(np.max(np.abs(pcm)))) < 0.01, "the mix-down is not averaging"


def test_livekit_output_is_resampled_back_up():
    import numpy as np
    src = _FakeSource()
    tp = _lk(_FakeStream([]), source=src)
    asyncio.run(tp.send_audio(np.zeros(16000, dtype="float32")))
    assert src.frames, "nothing was played"
    assert len(src.frames[0]) == 48000 * 2, "output was not resampled 16k -> 48k"

def test_livekit_a_failed_transcript_never_takes_the_call_down():
    class _BadParticipant:
        async def publish_data(self, *a, **k):
            raise RuntimeError("data channel closed")

    class _Room:
        local_participant = _BadParticipant()

    tp = _lk(_FakeStream([]), room=_Room())
    asyncio.run(tp.send_text("your appointment is confirmed"))   # must not raise


def test_livekit_hangup_closes_what_it_can_and_survives_what_it_cannot():
    closed = []

    class _Room:
        async def disconnect(self):
            closed.append("room")

    class _BadStream(_FakeStream):
        async def aclose(self):
            raise RuntimeError("already gone")

    tp = _lk(_BadStream([]), room=_Room())
    asyncio.run(tp.hangup())
    assert closed == ["room"], "one failing close stopped the other"
