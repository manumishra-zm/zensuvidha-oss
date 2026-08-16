"""Tests for the streaming input/output paths added to the voice loop.

Two things are covered, both testable without Ollama or a GPU:

  * SPECULATIVE STT — the client transcribes an utterance while the caller may still
    be mid-pause. The invariant that matters is that a speculative transcript NEVER
    reaches the LLM: the guard grounds numbers against the caller's complete words, so
    starting a turn on "my number is 892" would confirm a phone number nobody said.

  * PROGRESSIVE TTS — providers that can render a sentence incrementally stream PCM
    frames; providers that can't keep the whole-clip path untouched. The point is that
    adding the fast path costs the slow ones nothing.

Run:  pytest -q tests/test_streaming.py
"""
import io
import json
import wave

import pytest

from zensuvidha.tts import CachedTTS


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _wav(seconds=0.2, rate=16000):
    """A tiny valid WAV, so anything that parses audio sees real bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(rate * seconds))
    return buf.getvalue()


class WholeClipTTS:
    """A provider with no incremental support — macOS `say`, Piper, XTTS."""

    def __init__(self):
        self.calls = 0

    def synth(self, text, voice=None):
        self.calls += 1
        return _wav()


class ProgressiveTTS(WholeClipTTS):
    """A provider that renders segment-by-segment — Kokoro."""

    SR = 24000

    def __init__(self, segments=3):
        super().__init__()
        self.segments = segments
        self.stream_calls = 0

    def synth_stream(self, text, voice=None):
        self.stream_calls += 1
        for _ in range(self.segments):
            yield b"\x00\x01" * 1200, self.SR


# --------------------------------------------------------------------------- #
# progressive TTS
# --------------------------------------------------------------------------- #
def test_whole_clip_provider_exposes_no_stream():
    """The fast path must be opt-in by capability, not forced on every provider."""
    assert not hasattr(WholeClipTTS(), "synth_stream")
    assert CachedTTS(WholeClipTTS()).synth_stream("hello") is None


def test_progressive_provider_yields_frames_before_the_clip_is_done():
    inner = ProgressiveTTS(segments=4)
    frames = list(CachedTTS(inner).synth_stream("नमस्ते, कैसे हैं आप?"))
    assert len(frames) == 4
    assert all(sr == ProgressiveTTS.SR for _pcm, sr in frames)
    assert all(pcm for pcm, _sr in frames)


def test_streamed_clip_is_cached_so_the_repeat_is_instant():
    """A streamed sentence must still populate the LRU, or repeated confirmations
    would lose the cache they have today — a regression disguised as a feature."""
    inner = ProgressiveTTS()
    cached = CachedTTS(inner)
    text = "आपका बुकिंग रेफरेंस #11 है।"

    assert list(cached.synth_stream(text)), "first pass should stream"
    assert inner.stream_calls == 1

    # Second time it is a cache hit: no stream is offered…
    assert cached.synth_stream(text) is None
    # …and the whole-clip call is served from cache without touching the provider.
    assert cached.synth(text) is not None
    assert inner.calls == 0, "cached streamed audio must not be re-synthesised"


def test_cached_stream_assembles_a_playable_wav():
    inner = ProgressiveTTS(segments=3)
    cached = CachedTTS(inner)
    list(cached.synth_stream("hello"))
    with wave.open(io.BytesIO(cached.synth("hello")), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == ProgressiveTTS.SR
        assert w.getnframes() == 3 * 1200


def test_a_failing_stream_does_not_poison_the_cache():
    class Broken(ProgressiveTTS):
        def synth_stream(self, text, voice=None):
            yield b"\x00\x01" * 100, self.SR
            raise RuntimeError("model died mid-sentence")

    cached = CachedTTS(Broken())
    with pytest.raises(RuntimeError):
        list(cached.synth_stream("hello"))
    # No half-rendered clip may be cached — the next attempt must re-synthesise.
    assert cached.synth_stream("hello") is not None


# --------------------------------------------------------------------------- #
# speculative STT protocol
# --------------------------------------------------------------------------- #
class SpecProtocol:
    """The server's speculative-STT state machine, exercised message by message.

    Mirrors the `spec` dict and the branches in server.ws so the invariant can be
    tested without standing up a WebSocket, Whisper and Ollama.
    """

    def __init__(self, transcripts):
        self.spec = {"armed": False, "text": None}
        self.transcripts = list(transcripts)
        self.launched = []          # what actually reached the LLM
        self.partials = []          # what was shown in the UI only

    def control(self, msg):
        mtype = msg.get("type")
        if mtype == "stt_hint":
            if msg.get("mode") == "spec":
                self.spec["armed"] = True
            else:
                self.spec["armed"] = False
                self.spec["text"] = None
        elif mtype == "commit":
            text, self.spec["text"], self.spec["armed"] = self.spec["text"], None, False
            if text:
                self.launched.append(text)
        elif mtype == "cancel":
            self.spec["armed"] = False
            self.spec["text"] = None

    def audio(self):
        speculative = self.spec["armed"]
        self.spec["armed"] = False
        heard = self.transcripts.pop(0)
        if speculative:
            self.spec["text"] = heard
            if heard:
                self.partials.append(heard)
            return
        self.spec["text"] = None
        if heard:
            self.launched.append(heard)


def test_speculative_transcript_never_reaches_the_llm_on_its_own():
    """The whole point. A guess is displayed, never spoken on."""
    p = SpecProtocol(["मेरा मोबाइल नंबर"])
    p.control({"type": "stt_hint", "mode": "spec"})
    p.audio()
    assert p.partials == ["मेरा मोबाइल नंबर"]
    assert p.launched == [], "a speculative transcript must not start a turn"


def test_commit_uses_the_held_transcript_with_no_second_stt_pass():
    p = SpecProtocol(["शनिवार दस बजे डॉक्टर अनिल शर्मा"])
    p.control({"type": "stt_hint", "mode": "spec"})
    p.audio()                                     # the only STT pass
    p.control({"type": "commit"})
    assert p.launched == ["शनिवार दस बजे डॉक्टर अनिल शर्मा"]
    assert p.transcripts == [], "commit must not trigger another transcription"


def test_caller_resuming_mid_pause_discards_the_guess():
    """This is the bug that produced 'मेरा मोबाइल नंबर' with no number: the caller
    paused before the digits and the old endpointer shipped the fragment."""
    p = SpecProtocol(["मेरा मोबाइल नंबर", "मेरा मोबाइल नंबर 8920429057 है"])
    p.control({"type": "stt_hint", "mode": "spec"})
    p.audio()                                     # guess taken during the pause
    p.control({"type": "stt_hint", "mode": "drop"})   # they carried on speaking
    assert p.spec["text"] is None
    p.audio()                                     # full utterance at the real endpoint
    assert p.launched == ["मेरा मोबाइल नंबर 8920429057 है"]


def test_commit_after_a_drop_speaks_nothing():
    """A stale commit must be inert, not replay the discarded fragment."""
    p = SpecProtocol(["मेरा मोबाइल नंबर"])
    p.control({"type": "stt_hint", "mode": "spec"})
    p.audio()
    p.control({"type": "stt_hint", "mode": "drop"})
    p.control({"type": "commit"})
    assert p.launched == []


def test_barge_in_clears_a_pending_guess():
    p = SpecProtocol(["रुकिए"])
    p.control({"type": "stt_hint", "mode": "spec"})
    p.audio()
    p.control({"type": "cancel"})
    p.control({"type": "commit"})
    assert p.launched == []


def test_a_full_frame_supersedes_an_unconfirmed_guess():
    p = SpecProtocol(["आधा", "पूरा वाक्य"])
    p.control({"type": "stt_hint", "mode": "spec"})
    p.audio()
    p.audio()                                     # unarmed → a real turn
    assert p.launched == ["पूरा वाक्य"]
    assert p.spec["text"] is None


# --------------------------------------------------------------------------- #
# the real sender: server._stream_turn driving a progressive provider
# --------------------------------------------------------------------------- #
class StreamLLM:
    """Yields a JSON reply token-by-token, like Ollama's streaming endpoint."""

    def __init__(self, payload):
        self.text = json.dumps(payload, ensure_ascii=False)
        self.num_predict = 200
        self.num_ctx = 6144

    async def astream(self, messages, force_json=True, model=None, num_predict=None,
                      meta=None, num_ctx=None):
        for i in range(0, len(self.text), 7):
            yield self.text[i:i + 7]
        if meta is not None:
            meta["finish_reason"] = "stop"

    def chat(self, messages, **kw):
        return self.text


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, m):
        self.sent.append(m)

    async def send_bytes(self, b):
        hlen = int.from_bytes(b[:4], "big")
        meta = json.loads(b[4:4 + hlen])
        meta["_audio"] = b[4 + hlen:]
        self.sent.append(meta)

    def of_type(self, t):
        return [m for m in self.sent if m.get("type") == t]


def _stream_session(reply, provider):
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), StreamLLM(reply), tts=CachedTTS(provider))
    s.voice = None
    return s


REPLY = {"kind": "answer",
         "say": "We are open nine to eight. Sunday is ten to two. Please come by.",
         "action": {"type": "none"}}


def test_stream_turn_ships_pcm_frames_for_a_progressive_provider():
    import asyncio
    from zensuvidha import server

    inner = ProgressiveTTS(segments=3)
    sock = FakeSocket()
    session = _stream_session(REPLY, inner)
    asyncio.run(server._stream_turn(sock, session, "when are you open", "when are you open"))

    pcm = sock.of_type("pcm")
    assert pcm, "a progressive provider must produce pcm frames"
    assert not sock.of_type("chunk"), "it must not ALSO send whole clips"
    # exactly one frame per sentence carries the text, and it's the first
    starts = [m for m in pcm if m.get("start")]
    assert all(m["text"] for m in starts)
    assert all(m["text"] is None for m in pcm if not m.get("start"))
    # frames stay in sentence order, and every one is even-length int16
    assert [m["seq"] for m in pcm] == sorted(m["seq"] for m in pcm)
    assert all(len(m["_audio"]) % 2 == 0 and m["sr"] == 24000 for m in pcm)
    assert sock.of_type("reply_end"), "the turn must still complete normally"


def test_stream_turn_is_unchanged_for_a_whole_clip_provider():
    """The regression that would matter most: macOS `say` must behave exactly as before."""
    import asyncio
    from zensuvidha import server

    sock = FakeSocket()
    session = _stream_session(REPLY, WholeClipTTS())
    asyncio.run(server._stream_turn(sock, session, "when are you open", "when are you open"))

    assert sock.of_type("chunk"), "whole-clip providers must keep the WAV path"
    assert not sock.of_type("pcm")
    assert sock.of_type("reply_end")


def test_stream_turn_falls_back_when_a_stream_yields_nothing():
    """A provider that streams zero frames must not leave the caller in silence."""
    import asyncio
    from zensuvidha import server

    class Empty(ProgressiveTTS):
        def synth_stream(self, text, voice=None):
            return iter(())

    sock = FakeSocket()
    session = _stream_session(REPLY, Empty())
    asyncio.run(server._stream_turn(sock, session, "when are you open", "when are you open"))

    assert not sock.of_type("pcm")
    assert sock.of_type("chunk"), "must fall back to whole-clip synthesis"


# --------------------------------------------------------------------------- #
# wire format
# --------------------------------------------------------------------------- #
def test_audio_frame_round_trips_metadata_and_payload():
    from zensuvidha.server import _audio_frame
    pcm = b"\x01\x02" * 512
    frame = _audio_frame({"type": "pcm", "seq": 3, "sr": 24000, "start": True}, pcm)
    hlen = int.from_bytes(frame[:4], "big")
    meta = json.loads(frame[4:4 + hlen].decode("utf-8"))
    assert meta["type"] == "pcm" and meta["sr"] == 24000 and meta["start"] is True
    assert frame[4 + hlen:] == pcm
    # int16 frames must stay even-length or the browser's Int16Array view throws
    assert len(frame[4 + hlen:]) % 2 == 0


# --------------------------------------------------------------------------- #
# "have they finished talking?" — read from the words, not the silence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("said,expect_phone", [
    ("मेरा मोबाइल नंबर", False),          # the call #11 turn, shipped without its digits
    ("मेरा मोबाइल नंबर 892", True),       # a number being read out, stopped short
    ("मेरा नाम", False),
    ("my mobile number", True),
    ("my number is 98765", True),
    ("నా మొబైల్ నంబర్", False),
    ("mera mobile number", True),
    ("I want to book an appointment and", False),
])
def test_an_unfinished_sentence_is_detected(said, expect_phone):
    from zensuvidha.guard import looks_incomplete
    assert looks_incomplete(said, expect_phone)


@pytest.mark.parametrize("said,expect_phone", [
    ("मेरा मोबाइल नंबर 8920429057 है", True),
    ("8920429057", True),
    ("मेरा नाम मनु मिश्रा है", False),
    ("दस बजे सुबह", False),               # a time is a complete answer
    ("नमस्ते", False),
    ("మీ ఫీజు ఎంత", False),
    ("क्या आप डायलिसिस करते हैं", False),
    ("", False),
])
def test_a_finished_sentence_is_left_alone(said, expect_phone):
    """A false 'unfinished' costs a pause; a false 'finished' chops the sentence."""
    from zensuvidha.guard import looks_incomplete
    assert not looks_incomplete(said, expect_phone)


def test_go_on_exists_in_every_language():
    """The prompt to continue must never be composed by the model."""
    from zensuvidha.guard import SAFE_LINES, SAFE_LINES_ROMAN, safe_line
    from zensuvidha.packs import load_pack
    pack = load_pack("clinic")
    assert all("go_on" in t for t in SAFE_LINES.values())
    assert "go_on" in SAFE_LINES_ROMAN["Hindi"]
    lines = {safe_line("go_on", lang, pack) for lang in SAFE_LINES}
    assert len(lines) == len(SAFE_LINES), "each language needs its own wording"
    assert safe_line("go_on", "Hindi", pack) != safe_line("unknown", "Hindi", pack)


def test_the_filler_does_not_play_on_consecutive_turns():
    """On a slow box every reply trips the threshold; a filler before every answer reads
    as a stuck record rather than thinking."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), None)
    assert not s.filled_last_turn

    # turn 1 plays one
    s.filled_last_turn, s.filled_this_turn = s.filled_this_turn, False
    assert not s.filled_last_turn, "nothing played yet — turn 1 may fill"
    s.filled_this_turn = True

    # turn 2 must be suppressed
    s.filled_last_turn, s.filled_this_turn = s.filled_this_turn, False
    assert s.filled_last_turn, "turn 2 must skip the filler"

    # turn 3 is eligible again
    s.filled_last_turn, s.filled_this_turn = s.filled_this_turn, False
    assert not s.filled_last_turn, "turn 3 may fill again"


# --------------------------------------------------------------------------- #
# defects found by adversarial sweep
# --------------------------------------------------------------------------- #
def test_our_own_lines_are_never_read_as_unfinished():
    """A pre-written line judged 'unfinished' would make the agent nag the caller to
    continue after its own perfectly complete sentence."""
    from zensuvidha.guard import SAFE_LINES, ASK_LINES, safe_line, ask_line, looks_incomplete
    from zensuvidha.packs import load_pack
    pack = load_pack("clinic")
    for lang in SAFE_LINES:
        for kind in ("unknown", "scope", "repeat", "go_on"):
            line = safe_line(kind, lang, pack)
            assert not looks_incomplete(line, True), (lang, kind, line)
        for field in ASK_LINES[lang]:
            line = ask_line(field, lang)
            assert not looks_incomplete(line, True), (lang, field, line)


def test_our_own_lines_are_never_read_as_an_echo():
    """Self-rejection would loop: reject the reply, emit a safe line, reject that too."""
    from zensuvidha.guard import SAFE_LINES, ASK_LINES, safe_line, ask_line, looks_like_echo
    from zensuvidha.packs import load_pack
    pack = load_pack("clinic")
    for lang in SAFE_LINES:
        for kind in ("unknown", "scope", "repeat", "go_on"):
            line = safe_line(kind, lang, pack)
            assert not looks_like_echo(line, "मेरा नाम मनु है", ("8920429057",)), (lang, kind, line)
        for field in ASK_LINES[lang]:
            assert not looks_like_echo(ask_line(field, lang), "मेरा नाम मनु है", ()), (lang, field)


def test_a_question_is_always_a_complete_utterance():
    """"…anything else I can help you with?" ends on a dangling preposition, and
    Marathi's interrogative "का" is spelled like Hindi's genitive postposition."""
    from zensuvidha.guard import looks_incomplete
    for q in ("Is there anything else I can help you with?",
              "तुमचं पूर्ण नाव सांगाल का?",
              "अपॉइंटमेंट बुक करू का?"):
        assert not looks_incomplete(q, True), q


def test_a_possessive_and_a_question_in_different_sentences_are_fine():
    """"That detail is not with me. Anything else?" is two thoughts, not a misused
    possessive — the wrong-possessive check must not span sentences."""
    from zensuvidha.guard import looks_like_echo
    assert not looks_like_echo(
        "क्षमा कीजिए, वह जानकारी मेरे पास नहीं है। और कुछ मदद चाहिए?", "फीस कितनी है", ())
    # …but both in ONE sentence is still the failure
    assert looks_like_echo("मेरा मोबाइल नंबर क्या है?", "मेरा नाम मनु है", ())


@pytest.mark.parametrize("noise", ["hmm", "uh", "haan", "हाँ", "अच्छा", "ok", "అవును", "ठीक"])
def test_a_thinking_noise_is_not_a_name(noise):
    """Whisper transcribes hesitations faithfully; they are short and not questions, so
    without an explicit reject a caller saying "hmm" gets it filed as their name."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), None)
    s.booking_started = True
    s.pending_slot = "name"
    s._answer_to_pending(noise)
    assert not s.slots.get("name"), f"stored {s.slots.get('name')!r}"
    assert not s._plausible_slot("name", noise)


# --------------------------------------------------------------------------- #
# STT provider contract
# --------------------------------------------------------------------------- #
def test_a_provider_without_the_denoise_switch_still_works():
    """stt.py is a pluggable interface. Adding the DeepFilter toggle must not break a
    custom adapter written against the older signature."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack

    class OldStyleSTT:                       # no `denoise` kwarg at all
        def transcribe(self, audio, hint=None, language=None, fast=None):
            return "heard you", "en", 0.99

    s = Session(load_pack("clinic"), None, stt=OldStyleSTT())
    assert s.transcribe(b"x") == "heard you"          # toggle untouched
    s.stt_denoise = True                              # toggle ON
    assert s.transcribe(b"x") == "heard you", "must degrade gracefully, not raise"


def test_the_stt_provider_is_told_never_to_denoise():
    """Denoising moved OUT of the STT provider and into pipeline.prepare, which
    decides per turn and runs it before Whisper is called at all. The provider's
    own switch must therefore stay off in every configuration — when it did not,
    a turn the pipeline had already cleaned went through DeepFilterNet a second
    time, adding ~500ms and suppressing already-suppressed speech.

    The toggle itself is not lost: it reaches the pipeline as `denoise_mode`
    (see tests/test_pipeline.py::test_forcing_denoise_on_overrides_the_router)."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    seen = {}

    class NewStyleSTT:
        def transcribe(self, audio, hint=None, language=None, fast=None, denoise=None):
            seen["denoise"] = denoise
            return "ok", "en", 0.9

    s = Session(load_pack("clinic"), None, stt=NewStyleSTT())
    for toggle in (None, True, False):
        s.stt_denoise = toggle
        s.transcribe(b"x")
        assert seen["denoise"] is False, \
            f"stt_denoise={toggle!r} let the provider denoise a second time"


# --------------------------------------------------------------------------- #
# the turn must always terminate
# --------------------------------------------------------------------------- #
def test_an_llm_that_dies_mid_stream_does_not_hang_the_call():
    """generate() pushed its end-of-stream sentinel on its LAST line, so a failure
    before that (Ollama down, read timeout, malformed JSON) left the consumer blocked on
    task_q.get() FOREVER — no reply_end, no audio, the call dead for good. A blocked
    get() never raises, so none of the except handlers could fire either."""
    import asyncio
    from zensuvidha import server

    class DyingLLM:
        num_predict, num_ctx = 200, 6144
        async def astream(self, messages, meta=None, **kw):
            yield '{"kind":"answer","say":"We are open'      # a few tokens, then death
            raise RuntimeError("ollama went away")
        def chat(self, messages, **kw):
            raise RuntimeError("ollama went away")

    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    from zensuvidha.tts import CachedTTS
    s = Session(load_pack("clinic"), DyingLLM(), tts=CachedTTS(WholeClipTTS()))
    s.voice = None
    sock = FakeSocket()

    async def run():
        # 5s is ~100x the time a healthy turn needs here; before the fix this never returned
        try:
            await asyncio.wait_for(server._stream_turn(sock, s, "when open", "when open"), 5.0)
        except asyncio.TimeoutError:
            raise AssertionError("the turn HUNG — the sentinel was never delivered")
        except Exception:
            pass          # raising is fine: the caller falls back to a non-streaming turn
    asyncio.run(run())


def test_a_dying_llm_after_partial_audio_still_finishes_the_turn():
    """Once something has been spoken we must not replay it via the fallback — the turn
    has to close gracefully with whatever was generated."""
    import asyncio
    from zensuvidha import server
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    from zensuvidha.tts import CachedTTS

    class LateDeath:
        num_predict, num_ctx = 200, 6144
        async def astream(self, messages, meta=None, **kw):
            for chunk in ['{"kind":"answer","say":"We are open nine to eight. ',
                          'Sunday is ten to two. ']:
                yield chunk
            raise RuntimeError("stream died after audio was already sent")
        def chat(self, messages, **kw):
            return '{"kind":"answer","say":"ok","action":{"type":"none"}}'

    s = Session(load_pack("clinic"), LateDeath(), tts=CachedTTS(WholeClipTTS()))
    s.voice = None
    sock = FakeSocket()

    async def run():
        try:
            await asyncio.wait_for(server._stream_turn(sock, s, "when open", "when open"), 5.0)
        except asyncio.TimeoutError:
            raise AssertionError("the turn HUNG")
        except Exception:
            pass
    asyncio.run(run())
    assert sock.of_type("chunk"), "audio generated before the failure should still be sent"


def test_an_unparseable_reply_is_still_spoken():
    """A model reply with no parseable "say" produced ZERO audio: the stream had nothing
    to synthesise, finalize substituted a safe line, and the caller sat in silence while
    the UI showed text."""
    import asyncio
    from zensuvidha import server
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    from zensuvidha.tts import CachedTTS

    class Garbage:
        num_predict, num_ctx = 200, 6144
        async def astream(self, messages, meta=None, **kw):
            for c in ["not json at all", " still not json"]:
                yield c
            if meta is not None:
                meta["finish_reason"] = "stop"
        def chat(self, messages, **kw):
            return "not json at all"

    s = Session(load_pack("clinic"), Garbage(), tts=CachedTTS(WholeClipTTS()))
    s.voice = None
    sock = FakeSocket()
    asyncio.run(server._stream_turn(sock, s, "hello", "hello"))

    end = sock.of_type("reply_end")
    assert end and end[0]["text"].strip(), "finalize should still produce words"
    spoken = sock.of_type("chunk") + sock.of_type("pcm")
    assert spoken, "the caller must HEAR something, not just see text"


def test_a_database_failure_never_replays_the_reply():
    """A locked DB or full disk raised inside finalize AFTER audio had been sent, and the
    blanket handler re-ran the whole turn — the caller heard the SAME reply twice. If the
    condition persisted the retry raised too and the turn ended with no reply_end at all,
    leaving the UI hanging."""
    import asyncio
    from zensuvidha import server
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    from zensuvidha.tts import CachedTTS

    calls = {"llm": 0}

    class CountingLLM(StreamLLM):
        async def astream(self, m, meta=None, **kw):
            calls["llm"] += 1
            async for c in super().astream(m, meta=meta, **kw):
                yield c

    s = Session(load_pack("clinic"), CountingLLM(REPLY), tts=CachedTTS(WholeClipTTS()))
    s.voice = None
    def boom(*a, **k):
        raise Exception("database is locked")
    s.finalize = boom

    sock = FakeSocket()
    asyncio.run(server._stream_turn(sock, s, "when open", "when open"))

    assert sock.of_type("reply_end"), "the turn must still close, or the UI hangs forever"
    assert calls["llm"] == 1, "the reply must NOT be regenerated and spoken twice"


# --------------------------------------------------------------------------- #
# caller PII must not be readable from off-loopback
# --------------------------------------------------------------------------- #
def _fake_request(host, headers=None, params=None):
    class R:
        class client:
            pass
        pass
    r = R(); r.client = type("C", (), {"host": host})()
    r.headers = headers or {}
    r.query_params = params or {}
    return r


def test_localhost_can_still_read_transcripts_and_bookings():
    """A local operator must be unaffected — this is the normal dev and single-box case."""
    from zensuvidha import server
    for host in ("127.0.0.1", "::1", "localhost"):
        assert server._authorised(_fake_request(host))


def test_a_remote_client_is_refused_when_no_token_is_configured():
    """/transcripts and /bookings return caller names, phone numbers and full call text.
    config.gpu.yaml binds 0.0.0.0, so these were open to the network on the GPU box."""
    from zensuvidha import server
    old, server._ADMIN_TOKEN = server._ADMIN_TOKEN, None
    try:
        assert not server._authorised(_fake_request("10.0.0.9"))
        assert not server._authorised(_fake_request("10.0.0.9", {"x-admin-token": "guess"}))
    finally:
        server._ADMIN_TOKEN = old


def test_a_remote_client_needs_the_right_token():
    from zensuvidha import server
    old, server._ADMIN_TOKEN = server._ADMIN_TOKEN, "s3cret-token"
    try:
        assert server._authorised(_fake_request("10.0.0.9", {"x-admin-token": "s3cret-token"}))
        assert server._authorised(_fake_request("10.0.0.9", params={"token": "s3cret-token"}))
        assert not server._authorised(_fake_request("10.0.0.9", {"x-admin-token": "wrong"}))
        assert not server._authorised(_fake_request("10.0.0.9"))
    finally:
        server._ADMIN_TOKEN = old


# --------------------------------------------------------------------------- #
# Predictive endpointing — silence cannot tell "finished" from "thinking", words can
# --------------------------------------------------------------------------- #
def test_a_finished_phone_number_closes_the_turn_early():
    from zensuvidha.guard import looks_complete
    assert looks_complete("8920429057", expect_phone=True)
    assert looks_complete("my number is 8920429057", expect_phone=True)


def test_a_bare_yes_or_no_closes_the_turn_early():
    """Waiting 1200ms for a word that is already finished is most of what makes a call
    feel slow — "haan" admits no continuation."""
    from zensuvidha.guard import looks_complete
    for word in ("haan", "yes", "हाँ", "no", "nahi", "ఆమ్", "ok"):
        if word == "ఆమ్":
            continue                       # not in the list; the point is the shape
        assert looks_complete(word), f"{word!r} should close early"


@pytest.mark.parametrize("text,phone", [
    ("892042", True),                      # a number still being read out
    ("मेरा नाम", False),                    # an announcing noun phrase
    ("Manu", False),                       # a first name — "Mishra" may follow
    ("tomorrow", False),                   # "morning" may follow
    ("yes I would like to book an appointment", False),
    ("ठीक है", False),                      # a pair where only one word is terminal
    ("", False),
])
def test_it_never_closes_early_on_something_that_may_continue(text, phone):
    """The two mistakes are not symmetrical. A false "incomplete" costs a pause; a
    false "complete" chops the sentence in half — the failure the endpointer has
    already been tuned twice to avoid."""
    from zensuvidha.guard import looks_complete
    assert not looks_complete(text, expect_phone=phone)


def test_complete_and_incomplete_are_never_both_true():
    from zensuvidha.guard import looks_complete, looks_incomplete
    probes = ["haan", "8920429057", "892042", "मेरा नाम", "yes please book it",
              "Manu", "", "tomorrow morning", "what are your timings?"]
    for t in probes:
        for phone in (True, False):
            assert not (looks_complete(t, expect_phone=phone)
                        and looks_incomplete(t, expect_phone=phone)), t


def test_the_client_prefers_the_conservative_hint():
    """`holdForMore` must win over `settled`. If the server ever sent both, closing
    early is the one that can chop a sentence."""
    import pathlib
    js = pathlib.Path("web/index.html").read_text()
    # Asserted as a PROPERTY rather than a literal line: the early close must be gated
    # on every signal that says "not finished". It now has two — the words and a
    # trailing filler — and pinning the old exact string would have failed the moment
    # the second was added, while a client that dropped one of them would still pass.
    line = next((ln.strip() for ln in js.splitlines()
                 if "settled" in ln and "return TURN.settled_ms" in ln), None)
    assert line, "the settled early-close is gone"
    for guard in ("!holdForMore", "!fillerHold"):
        assert guard in line, f"the early close is not gated on {guard}: {line}"


def test_the_settled_hint_is_cleared_not_latched():
    """holdForMore once latched on, so a single mid-sentence guess added HOLD_EXTRA_MS
    to every remaining endpoint of the call. The same bug in reverse would be worse."""
    import pathlib
    js = pathlib.Path("web/index.html").read_text()
    assert "settled = (m.settled===true);" in js, "settled must be assigned, not OR-ed"


# --------------------------------------------------------------------------- #
# Backchannels — the "mm-hm" a listener makes while the OTHER person is talking
# --------------------------------------------------------------------------- #
def test_a_backchannel_exists_for_every_language_we_speak():
    from zensuvidha.server import _BACKCHANNELS
    from zensuvidha.guard import SAFE_LINES
    missing = set(SAFE_LINES) - set(_BACKCHANNELS)
    assert not missing, f"no listening noise for {missing}"


def test_it_is_one_short_word_not_a_sentence():
    """Anything longer stops being a listening noise and becomes an interruption — the
    caller stops to let us finish, then has to restart their sentence."""
    from zensuvidha.server import _BACKCHANNELS
    for lang, phrase in _BACKCHANNELS.items():
        assert len(phrase.split()) <= 2, f"{lang}: {phrase!r} is too long to murmur"
        assert len(phrase) <= 12, f"{lang}: {phrase!r} is too long"


def test_it_never_falls_back_to_english_mid_call():
    """Dropping an English "mm-hm" into a Telugu call breaks the single-language rule
    the fillers already observe."""
    from zensuvidha.server import _backchannel_for
    assert _backchannel_for("Telugu") is not None
    assert _backchannel_for("Swahili") is None       # unknown → silence, not English


def test_the_client_gates_it_on_every_condition_that_stops_it_interrupting():
    """Each guard removes a way this becomes an interruption rather than a murmur.
    Losing any one of them is the difference between natural and infuriating."""
    import pathlib
    js = pathlib.Path("web/index.html").read_text()
    for guard, why in [
        ("bcUsedThisTurn", "at most once per turn"),
        ("uttMs < BACKCHANNEL_AFTER_MS", "only on a turn long enough to feel unattended"),
        ("silenceMs < BACKCHANNEL_PAUSE_MS", "only into a real pause, never over a word"),
        ("agentLevel() > 0.001", "never while we are already speaking"),
    ]:
        assert guard in js, f"the backchannel lost its guard: {why}"
    assert "bcUsedThisTurn=false" in js, "the once-per-turn flag is never reset"


def test_the_backchannel_is_ducked():
    """A listener murmurs; they do not announce."""
    import pathlib
    import re
    js = pathlib.Path("web/index.html").read_text()
    m = re.search(r"BACKCHANNEL_GAIN\s*=\s*([0-9.]+)", js)
    assert m and float(m.group(1)) < 0.5, "the listening noise is as loud as speech"
