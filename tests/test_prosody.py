"""Reading the caller's pitch to tell "finished" from "paused mid-thought".

The value of this is entirely in what it does NOT do. It is a third opinion on turn
endings, behind silence and behind the words, and the two ways it could hurt are both
about overreach: deciding a turn on its own, or answering when it cannot actually tell.
So most of what is pinned here is restraint.

The measurements behind the thresholds are in zensuvidha/prosody.py and reproducible
with `python scripts/bench_prosody.py`.

Run:  pytest -q tests/test_prosody.py
"""
import math

import numpy as np
import pytest

from zensuvidha import prosody as P

SR = 16000


def voiced(seconds, f_start, f_end, amp_start=0.4, amp_end=0.4):
    """A synthetic voiced sound whose pitch glides and whose level ramps.

    Not speech — a harmonic buzz. That is deliberate: it isolates the arithmetic from
    everything a real voice also carries, so a failure here is the detector's and not
    the fixture's. Real speech is measured in scripts/bench_prosody.py.
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(f) / SR
    amp = np.linspace(amp_start, amp_end, n)
    # a couple of harmonics, or autocorrelation has an unrealistically easy time
    return (amp * (np.sin(phase) + 0.5 * np.sin(2 * phase)
                   + 0.25 * np.sin(3 * phase))).astype("float32")


# --------------------------------------------------------------------------- #
# it declines rather than guessing
# --------------------------------------------------------------------------- #
def test_silence_is_unreadable_not_finished():
    """Silence must never be read as a finished sentence. It is the commonest input on
    the path this runs on, and 'finished' would shorten the window on nothing at all."""
    assert P.finality(np.zeros(SR, dtype="float32")) is None


def test_noise_is_unreadable():
    rng = np.random.default_rng(0)
    assert P.finality(rng.normal(0, 0.2, SR).astype("float32")) is None


def test_a_clip_too_short_to_read_is_declined():
    assert P.finality(voiced(0.1, 150, 150)) is None


def test_an_unreadable_clip_changes_nothing():
    """The whole fail-open contract in one line: no reading, no effect."""
    assert P.window_scale(None) == 1.0
    assert P.window_scale("unsure") == 1.0


# --------------------------------------------------------------------------- #
# what it does read
# --------------------------------------------------------------------------- #
def test_a_falling_contour_with_fading_energy_reads_as_finished():
    out = P.finality(voiced(1.0, 220, 120, amp_start=0.5, amp_end=0.08))
    assert out and out["verdict"] == "finished", out


def test_a_held_pitch_at_steady_level_reads_as_holding():
    """Someone who means to carry on holds both. This is the case the whole feature
    exists for — silence cannot tell it from the one above."""
    out = P.finality(voiced(1.0, 180, 178, amp_start=0.4, amp_end=0.4))
    assert out and out["verdict"] == "holding", out


def test_the_dead_band_declines_the_ambiguous_middle():
    """Measured on real speech, a finished clip scored 8.1 and an unfinished one 7.0.
    One point apart and opposite answers — so the band between them must produce no
    opinion at all rather than a coin toss."""
    # A gesture halfway between held and decisively falling. Swept to find it rather
    # than guessed: on this buzz fixture the band runs from roughly f_end 175 down to
    # 155 at these levels, which is where a real borderline utterance lands too.
    seen = set()
    for f_end, amp_end in ((160, 0.34), (160, 0.28), (175, 0.22)):
        out = P.finality(voiced(1.0, 220, f_end, amp_start=0.4, amp_end=amp_end))
        assert out, (f_end, amp_end)
        seen.add(out["verdict"])
    assert seen == {"unsure"}, seen


def test_pitch_is_judged_in_semitones_not_hertz():
    """A 40Hz drop is a big gesture for a low voice and a small one for a high voice.
    In Hz the same sentence read by two people gets two different verdicts."""
    low = P.finality(voiced(1.0, 120, 90, amp_start=0.5, amp_end=0.5))
    high = P.finality(voiced(1.0, 240, 180, amp_start=0.5, amp_end=0.5))
    assert low and high
    # the same musical interval, so the slopes must be close
    assert abs(low["slope_st_s"] - high["slope_st_s"]) < 2.0, (low, high)


def test_one_octave_error_cannot_invent_a_verdict():
    """Autocorrelation's real failure mode is halving or doubling the pitch on a frame
    or two. A least-squares fit would let one such frame drag the whole contour, so the
    slope is taken from medians instead."""
    clean = voiced(1.0, 180, 178, amp_start=0.4, amp_end=0.4)
    honest = P.finality(clean)
    # corrupt a short stretch the way an octave error would
    broken = clean.copy()
    broken[int(0.45 * SR):int(0.52 * SR)] *= 0.02
    got = P.finality(broken)
    assert honest and got
    assert got["verdict"] == honest["verdict"], (honest, got)


# --------------------------------------------------------------------------- #
# how it is allowed to act
# --------------------------------------------------------------------------- #
def test_it_can_only_nudge_the_window_never_set_it():
    """Every scale sits close to 1. This is a third opinion on a decision two other
    signals already make; a scale of 0.2 would make it the decider."""
    for verdict in ("finished", "holding", "unsure", None):
        s = P.window_scale(verdict)
        assert 0.7 <= s <= 1.3, (verdict, s)


def test_it_is_more_willing_to_wait_than_to_cut():
    """The two mistakes are not symmetrical. Waiting too long costs a pause; closing
    early chops a sentence in half, which is the failure this endpointer has already
    been retuned twice to avoid."""
    shorten = 1 - P.window_scale("finished")
    lengthen = P.window_scale("holding") - 1
    assert lengthen >= shorten * 0.8, (shorten, lengthen)


def test_the_verdicts_and_the_scale_table_agree():
    """A verdict the scale table does not know would silently become 1.0 — the feature
    switched off by a typo, with nothing to show for it."""
    produced = set()
    for args in ((1.0, 220, 120, 0.5, 0.08), (1.0, 180, 178, 0.4, 0.4),
                 (1.0, 220, 190, 0.4, 0.30)):
        out = P.finality(voiced(*args))
        if out:
            produced.add(out["verdict"])
    assert produced <= {"finished", "holding", "unsure"}, produced
    for v in produced:
        assert P.window_scale(v) != 1.0 or v == "unsure"


def test_it_is_cheap_enough_to_run_on_every_speculative_frame():
    """It runs on the latency-critical path. Measured ~1.6ms on real clips; the bound
    here is loose enough not to be flaky on a loaded machine and tight enough that a
    rewrite into something expensive fails."""
    import time
    clip = voiced(3.0, 200, 140, amp_start=0.5, amp_end=0.1)
    P.finality(clip)                                     # warm numpy
    t0 = time.time()
    for _ in range(5):
        P.finality(clip)
    ms = (time.time() - t0) * 1000 / 5
    assert ms < 40, "%.1fms per call" % ms


def test_only_the_tail_is_read():
    """Finality is a property of the last syllable or two. Reading the whole utterance
    averages the gesture away — and on a 30s turn it would also stop being cheap."""
    tail = voiced(0.7, 220, 120, amp_start=0.5, amp_end=0.08)
    lead = voiced(4.0, 170, 170, amp_start=0.4, amp_end=0.4)
    out = P.finality(np.concatenate([lead, tail]))
    assert out and out["verdict"] == "finished", out
