"""Speculative reply — answer the guess while the caller may still be pausing.

STT is already off the critical path when the guess is right; this takes the LLM off it
too. The whole design rests on one invariant the codebase learned the hard way: **a
guess never mutates session state.** A speculative voiceprint once locked a caller out
of their own call for its whole duration, so that rule is pinned here structurally.
"""
import asyncio
import inspect

import pytest

from zensuvidha import server


def test_speculation_never_mutates_the_session():
    """`begin_user` appends history AND folds the caller's numbers into the grounding
    set. Neither may happen on a sentence the caller has not finished — if they carry
    on talking, the turn has to look as though nothing happened."""
    import ast
    import textwrap

    src = textwrap.dedent(inspect.getsource(server._speculate_reply))
    fn = ast.parse(src).body[0]
    # Drop the docstring — it NAMES the things it must not do, which is the point of
    # the docstring and would otherwise fail this check for the right reason.
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    code = "\n".join(ast.unparse(n) for n in body)

    for forbidden in ("begin_user", "_append", "finalize", "session.slots",
                      "note_played", "check_speaker", "_caller_numbers"):
        assert forbidden not in code, (
            f"_speculate_reply touches {forbidden} — a guess must not change state")
    assert "call_messages()" in code, "it must READ the prompt, not rebuild it"


def test_it_is_only_started_on_a_transcript_that_looks_finished():
    """Replying to half a sentence is worse than waiting for the whole one, and a
    half-spoken phone number must never reach the guard."""
    src = inspect.getsource(server)
    start = src.index('spec["gen"] = asyncio.create_task(')
    window = src[max(0, start - 700):start]
    assert 'not spec["incomplete"]' in window, (
        "speculation must be gated on the transcript looking finished")


def test_every_path_that_voids_a_guess_calls_the_same_helper():
    """Missing one is how a reply to a half-finished sentence reaches the caller."""
    src = inspect.getsource(server)
    assert src.count("_void_speculation(spec)") >= 4, (
        "the caller resuming, barging in, a newer guess and the commit must all void it")


def test_the_reply_is_only_adopted_when_the_transcript_is_IDENTICAL():
    src = inspect.getsource(server)
    assert 'gen_for == text' in src, (
        "a reply generated for different words must never be spoken")


def test_a_precomputed_reply_replays_through_the_same_pipeline():
    """Not a second code path. The guard, the sentence splitter, the TTS pipeline and
    the history must all behave identically, or the fast case quietly diverges from the
    slow one."""
    src = inspect.getsource(server._stream_turn)
    assert "async def _tokens()" in src
    assert "async for delta in _tokens():" in src, (
        "the replay must go through the same loop the live stream uses")
    # and the guard still runs on it
    assert "check_reply" in src or "gate[" in src


def test_void_speculation_cancels_work_still_running():
    async def go():
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(30)
            return "should never arrive"

        spec = {"gen": asyncio.create_task(slow()), "gen_for": "hello"}
        await started.wait()
        server._void_speculation(spec)
        await asyncio.sleep(0)
        assert spec["gen"] is None and spec["gen_for"] is None
    asyncio.run(go())


def test_void_speculation_is_safe_when_nothing_is_running():
    for spec in ({}, {"gen": None, "gen_for": None}):
        server._void_speculation(spec)          # must not raise
        assert spec.get("gen") is None


def test_a_failed_speculation_falls_back_to_generating_normally():
    """A speculative reply is an optimisation. If it raises, the caller must still get
    an answer — the same rule every other optional stage in this pipeline follows."""
    src = inspect.getsource(server)
    at = src.index("pre = await gen")
    window = src[at:at + 600]
    assert "except Exception" in window and "pre = None" in window


def test_the_feature_can_be_switched_off():
    assert isinstance(server.SPECULATIVE_REPLY, bool)
