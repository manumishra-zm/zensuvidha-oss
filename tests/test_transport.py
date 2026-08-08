"""The seam between a call and the thing carrying it.

The engine's audio stages — isolation, the speaker gate, the guard, the router — are
already transport-agnostic. What was missing was somewhere for a second transport to
plug in without touching them, and a place to put the things a browser supplies for free
and a phone line does not: echo handling, voice activity, and endpointing.
"""
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
    assert closed_at > 1800 + T.Endpointer.SHORT_MS, (
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
