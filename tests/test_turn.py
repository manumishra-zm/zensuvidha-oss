"""The turn-taking ladder, now that there is only one of it.

Before this module the policy lived in three places that could not see each other — the
browser endpointer, the flags server.py pushed to it, and `transport.Endpointer`. They
had already drifted: the telephony endpointer was a flat 800/1200ms with no learned
pause, no per-slot extra and no prosody, so a caller on a phone line would have been cut
off by exactly the rules two rounds of live testing tuned away from.

So most of what is pinned here is that the three agree, and that the ORDER of the
signals is the one that was arrived at by things going wrong.

Run:  pytest -q tests/test_turn.py
"""
import json
import os
import shutil
import subprocess

import pytest

from zensuvidha import turn as T

NORMAL = T.policy("normal")


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #
def test_a_short_answer_closes_faster_than_a_sentence():
    """"haan" is finished the moment it stops; making it wait for the window sized for
    somebody mid-sentence is most of what makes a call feel slow."""
    short = T.window_ms(NORMAL, utt_ms=500)
    long_ = T.window_ms(NORMAL, utt_ms=3000)
    assert short < long_
    assert short == NORMAL["base_ms"]


def test_this_caller_s_own_pauses_widen_the_window():
    """Learned from pauses they demonstrably spoke THROUGH."""
    plain = T.window_ms(NORMAL, utt_ms=3000)
    slow = T.window_ms(NORMAL, utt_ms=3000, learned_pause_ms=1600)
    assert slow > plain


def test_the_words_outrank_everything_below_them():
    """A caller who says "my mobile number is" with a textbook falling contour has still
    not given us the number."""
    held = T.window_ms(NORMAL, utt_ms=3000, hold=True, tone="finished")
    plain = T.window_ms(NORMAL, utt_ms=3000, hold=True)
    assert held == plain, "the pitch contour overrode the transcript"
    assert held > T.window_ms(NORMAL, utt_ms=3000)


def test_a_filler_waits_less_than_a_dangling_postposition():
    """"…with, um" is weaker evidence than "…मेरा" — it says they are searching for a
    word, not that the sentence is grammatically unfinished."""
    filler = T.window_ms(NORMAL, utt_ms=3000, filler=True)
    hold = T.window_ms(NORMAL, utt_ms=3000, hold=True)
    plain = T.window_ms(NORMAL, utt_ms=3000)
    assert plain < filler < hold


def test_one_hesitation_is_not_charged_twice():
    """A turn that is BOTH grammatically unfinished and trails off on a filler must get
    one extension, not two stacked."""
    both = T.window_ms(NORMAL, utt_ms=3000, hold=True, filler=True)
    hold = T.window_ms(NORMAL, utt_ms=3000, hold=True)
    assert both == hold


def test_a_finished_transcript_closes_early_but_only_if_nothing_holds_it():
    assert T.window_ms(NORMAL, utt_ms=500, settled=True) == NORMAL["settled_ms"]
    # …and every signal that says "not finished" cancels it. "Unfinished" is the
    # conservative answer whenever the two disagree.
    assert T.window_ms(NORMAL, utt_ms=500, settled=True, hold=True) > NORMAL["settled_ms"]
    assert T.window_ms(NORMAL, utt_ms=500, settled=True, filler=True) > NORMAL["settled_ms"]


def test_the_expected_slot_changes_the_window():
    """Reading a phone number aloud has long gaps between digit groups; a name does not."""
    phone = T.window_ms(NORMAL, utt_ms=3000, expect="phone")
    name = T.window_ms(NORMAL, utt_ms=3000, expect="name")
    assert phone > name > T.window_ms(NORMAL, utt_ms=3000)


def test_the_clamp_is_applied_after_the_tone_scale():
    """Scaling an already-clamped window pushed it PAST the ceiling — and the client
    stores the result as the learned pause, so one "holding" verdict ratcheted the
    caller's window up for the rest of the call, above its own maximum."""
    held = T.window_ms(NORMAL, utt_ms=9000, learned_pause_ms=9000, tone="holding")
    assert held <= NORMAL["max_ms"], held


# --------------------------------------------------------------------------- #
# eagerness
# --------------------------------------------------------------------------- #
def test_eagerness_scales_the_ladder_in_the_right_direction():
    e, n, p = (T.window_ms(T.policy(x), utt_ms=3000)
               for x in ("eager", "normal", "patient"))
    assert e < n < p


def test_eagerness_does_not_reorder_the_signals():
    """It changes how patient the agent is, never which evidence wins. If Patient let
    the contour override the words, a caller mid-phone-number would be cut off on the
    setting chosen to avoid exactly that."""
    for name in ("eager", "normal", "patient"):
        pol = T.policy(name)
        plain = T.window_ms(pol, utt_ms=3000)
        assert T.window_ms(pol, utt_ms=3000, filler=True) > plain, name
        assert (T.window_ms(pol, utt_ms=3000, hold=True)
                > T.window_ms(pol, utt_ms=3000, filler=True)), name
        assert T.window_ms(pol, utt_ms=3000, hold=True, tone="finished") == \
            T.window_ms(pol, utt_ms=3000, hold=True), name


def test_a_finished_transcript_is_never_made_to_wait_longer():
    """`settled` means "nothing can follow" — a fact about the transcript, not a
    preference. Scaling it with Patient would just be a slower call for no reason."""
    for name in ("eager", "normal", "patient"):
        assert T.window_ms(T.policy(name), utt_ms=500, settled=True) == \
            T.NORMAL["settled_ms"], name


def test_an_unknown_eagerness_falls_back_rather_than_breaking_turn_taking():
    assert T.policy("nonsense")["eagerness"] == T.DEFAULT_EAGERNESS
    assert T.policy(None)["eagerness"] == T.DEFAULT_EAGERNESS


def test_a_policy_survives_the_json_round_trip_to_the_browser():
    """It is sent on the session frame. A non-serialisable value would take the whole
    handshake down, not just turn-taking."""
    pol = T.policy("patient")
    assert json.loads(json.dumps(pol)) == pol


def test_policies_do_not_share_nested_state():
    """`expect_extra_ms` and `tone_scale` are dicts on the module-level NORMAL. Handing
    out references would let one call's eagerness change another's."""
    a, b = T.policy("eager"), T.policy("patient")
    a["expect_extra_ms"]["phone"] = 99999
    assert b["expect_extra_ms"]["phone"] != 99999
    assert T.NORMAL["expect_extra_ms"]["phone"] != 99999


# --------------------------------------------------------------------------- #
# the three consumers agree
# --------------------------------------------------------------------------- #
def test_the_phone_endpointer_uses_the_shared_ladder():
    from zensuvidha import transport as Tr
    ep = Tr.Endpointer(eagerness="patient")
    assert ep.policy["eagerness"] == "patient"
    assert not hasattr(Tr.Endpointer, "SHORT_MS"), (
        "the telephony endpointer still carries its own copy of the ladder")


NODE = shutil.which("node")
INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "web", "index.html")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_browser_evaluates_the_same_ladder_the_server_sends():
    """The browser holds a fallback copy for the moment before the session frame lands.
    If that copy drifts from turn.py, every caller is endpointed by the wrong numbers
    until the handshake completes — and nothing anywhere would say so."""
    src = open(INDEX, encoding="utf-8").read()
    start = src.index("let TURN = {")
    # Brace-matched, not sliced at the first "}": the literal contains two nested
    # objects, so a naive cut produces something that parses as neither JS nor JSON.
    i, depth = src.index("{", start), 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = src[start:i + 1] + ";"
    out = subprocess.run([NODE, "-e", body + "\nconsole.log(JSON.stringify(TURN));"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    browser = json.loads(out.stdout)
    for key in ("base_ms", "long_ms", "long_utt_ms", "max_ms", "settled_ms",
                "hold_extra_ms", "filler_extra_ms", "pause_factor"):
        assert browser[key] == NORMAL[key], (
            "browser fallback disagrees with turn.py on %s: %s vs %s"
            % (key, browser[key], NORMAL[key]))
    assert browser["tone_scale"] == NORMAL["tone_scale"]
    assert browser["expect_extra_ms"] == NORMAL["expect_extra_ms"]
