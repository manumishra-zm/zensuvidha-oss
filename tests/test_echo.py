"""Server-side echo suppression — hearing ourselves is not the caller talking.

The failure this prevents is specific and total: on a line with no AEC the agent speaks,
hears itself, treats that as barge-in, stops, then answers its own tail as a new turn.
The failure it must NOT cause is worse — refusing the caller — so almost every test here
is about what it lets through.
"""
import numpy as np
import pytest

from zensuvidha.echo import EchoSuppressor, ECHO_CORRELATION

SR = 16000


def voice(f0, secs=1.0, seed=0):
    """A crude voiced signal: a harmonic stack under a syllable-rate envelope."""
    t = np.linspace(0, secs, int(secs * SR), endpoint=False)
    r = np.random.default_rng(seed)
    env = np.abs(np.sin(2 * np.pi * 3.1 * t)) + 0.15
    stack = sum(np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6)) / k for k in range(1, 9))
    return (0.3 * env * stack).astype("float32")


def delayed(x, ms):
    n = int(ms * SR / 1000)
    return np.concatenate([np.zeros(n, dtype="float32"), x])[:x.size]


AGENT = voice(120, 1.0, seed=1)
CALLER = voice(210, 1.0, seed=2)


def _sup():
    e = EchoSuppressor()
    e.note_output(AGENT)
    return e


# ── it recognises our own voice coming back ────────────────────────────────────

@pytest.mark.parametrize("name,mic", [
    ("full level", AGENT.copy()),
    ("attenuated 12dB", (AGENT * 0.25).astype("float32")),
    ("attenuated 20dB", (AGENT * 0.10).astype("float32")),
    ("delayed 40ms", delayed(AGENT, 40)),
    ("delayed 120ms", delayed(AGENT, 120)),
    ("delayed 300ms", delayed(AGENT, 300)),
    ("delayed 500ms", delayed(AGENT, 500)),
    ("attenuated AND delayed", (delayed(AGENT, 150) * 0.25).astype("float32")),
    ("with room noise", (AGENT * 0.4 + 0.01 * np.random.default_rng(3)
                         .normal(size=AGENT.size)).astype("float32")),
    ("a short 30ms frame", AGENT[:480].copy()),
])
def test_our_own_audio_is_recognised(name, mic):
    got, corr = _sup().is_echo(mic)
    assert got, f"{name}: passed through at correlation {corr:.3f}"


def test_a_delayed_echo_is_the_case_that_matters():
    """The first implementation probed the HEAD of the frame, so a 300ms-delayed echo
    was 300ms of silence and scored 0.451 — straight through. Echo is always delayed,
    so that was the only case that mattered and it was the one that failed."""
    got, corr = _sup().is_echo(delayed(AGENT, 300))
    assert got and corr > 0.9, f"correlation {corr:.3f}"


# ── it must NOT take the caller's turn ─────────────────────────────────────────

@pytest.mark.parametrize("name,mic", [
    ("a different voice", CALLER),
    ("the caller talking OVER our echo", (CALLER + AGENT * 0.25).astype("float32")),
    ("a quiet caller", (CALLER * 0.2).astype("float32")),
    ("silence", np.zeros(SR, dtype="float32")),
    ("white noise", (0.2 * np.random.default_rng(4).normal(size=SR)).astype("float32")),
    ("a fan at 50Hz", (0.1 * np.sin(2 * np.pi * 50 * np.arange(SR) / SR)).astype("float32")),
])
def test_the_caller_is_never_mistaken_for_us(name, mic):
    got, corr = _sup().is_echo(mic)
    assert not got, f"{name}: refused as echo at correlation {corr:.3f}"


def test_the_caller_talking_over_us_still_gets_through():
    """Barge-in is the whole point of the feature this must not break. Their words are
    mixed WITH our echo, and the mixture must still read as them."""
    got, _ = _sup().is_echo((CALLER + AGENT * 0.25).astype("float32"))
    assert not got


def test_the_margin_is_wide_not_marginal():
    """Echo lands near 1.0 and everything else below 0.3, so the threshold sits in a
    gap rather than on a slope. If this narrows, the discrimination is degrading."""
    echo = _sup().is_echo(delayed(AGENT, 120))[1]
    other = _sup().is_echo(CALLER)[1]
    assert echo > 0.9 and other < 0.35, f"echo {echo:.3f} vs caller {other:.3f}"
    assert other < ECHO_CORRELATION < echo


# ── it fails open, everywhere ──────────────────────────────────────────────────

def test_with_no_reference_yet_nothing_is_echo():
    """The first thing on a call is the caller speaking, before we have played a word."""
    assert EchoSuppressor().is_echo(CALLER)[0] is False


def test_disabled_means_disabled():
    e = EchoSuppressor(enabled=False)
    e.note_output(AGENT)
    assert e.is_echo(AGENT.copy())[0] is False


@pytest.mark.parametrize("junk", [
    np.zeros(0, dtype="float32"),
    np.zeros(10, dtype="float32"),
    np.full(SR, np.nan, dtype="float32"),
    np.full(SR, np.inf, dtype="float32"),
])
def test_malformed_audio_is_accepted_not_refused(junk):
    """A false "echo" silences the caller. Anything this cannot judge must pass."""
    got, _ = _sup().is_echo(junk)
    assert got is False


def test_reset_forgets_the_reference():
    """After a barge-in we stop playing, and a stale tail must not be able to explain
    away the words the caller interrupted us with."""
    e = _sup()
    assert e.is_echo(AGENT.copy())[0]
    e.reset()
    assert not e.is_echo(AGENT.copy())[0]


def test_it_is_cheap_enough_for_the_audio_path():
    import time
    e = _sup()
    t = time.perf_counter()
    for _ in range(20):
        e.is_echo(CALLER)
    per = (time.perf_counter() - t) / 20
    assert per < 0.06, f"{per*1000:.1f}ms per frame is too slow for a 1070ms budget"


# ── the wiring ─────────────────────────────────────────────────────────────────

def test_every_frame_we_play_is_recorded_as_reference():
    """A reference with holes is worse than none: the frames it fails to explain are
    exactly the ones that get answered as though the caller had said them."""
    import inspect
    from zensuvidha import server
    src = inspect.getsource(server._send_audio)
    assert "note_played" in src, "_send_audio is the only choke point; it must record"
    calls = [ln for ln in inspect.getsource(server).splitlines()
             if "_send_audio(sock" in ln and "async def" not in ln]
    assert calls, "no call sites found — the check itself is broken"


def test_the_echo_check_runs_on_raw_audio_before_the_pipeline():
    """A denoiser reshapes the echo enough to break the correlation, and isolation would
    trim our own voice into something unrecognisable."""
    import inspect
    from zensuvidha import server
    src = inspect.getsource(server)
    assert src.index("session.is_own_echo") < src.index("session.clean_audio"), (
        "the echo check must see the audio before anything modifies it")
