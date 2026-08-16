"""Saying the fixed lines once, in the owner's voice.

The idea this rests on is countable: most of what the agent says on a call is already
written down. The clinic pack yields 318 fixed sentences — the greeting, every slot
question, every safe line in twelve languages, and the knowledge answers the semantic
fast path quotes verbatim. Cloning is far too slow for a turn (seconds per sentence on
CPU) and perfectly fine for a one-off render.

What can go wrong is quieter than it sounds:

  * the cache evicting the render as fast as it is made — 318 lines against a default
    maxsize of 256 means a fifth of the owner's voice is discarded on arrival;
  * a language the voice cannot speak being counted as a failure, so a healthy render
    on a twelve-language pack looks broken;
  * the render blocking the operator, or a failure costing more than the feature.

Run:  pytest -q tests/test_prerender.py
"""
import pytest

from zensuvidha.packs import load_pack
from zensuvidha.prerender import fixed_lines, prerender
from zensuvidha.tts import CachedTTS


class FakeVoice:
    """A cloner that is slow and cannot speak every script — like the real ones."""

    def __init__(self, cannot=(), fail_on=()):
        self.cannot = set(cannot)
        self.fail_on = set(fail_on)
        self.calls = []
        self.last_skipped_script = False

    def synth(self, text, voice=None):
        self.calls.append(text)
        if text in self.fail_on:
            raise RuntimeError("synth exploded")
        # The documented contract: a provider signals "not my script" by returning None
        # with the flag set, rather than producing confident nonsense.
        if any(c in text for c in self.cannot):
            self.last_skipped_script = True
            return None
        self.last_skipped_script = False
        return b"RIFF" + text.encode()[:8].ljust(60, b"\0")


@pytest.fixture()
def pack():
    return load_pack("clinic")


# --------------------------------------------------------------------------- #
# what counts as a fixed line
# --------------------------------------------------------------------------- #
def test_it_finds_the_lines_a_caller_actually_hears(pack):
    lines = fixed_lines(pack)
    texts = [t for _lang, t in lines]
    assert len(lines) > 200, len(lines)
    assert pack["greeting"] in texts, "the one line heard on every single call"
    slot_qs = list((pack["booking"]["slots"]).values())
    assert any(q in texts for q in slot_qs), "the booking questions are missing"
    answers = [e.get("a") for e in pack["knowledge"] if e.get("a")]
    assert sum(a in texts for a in answers) > 20, "the quoted knowledge answers are missing"


def test_the_greeting_comes_first(pack):
    """A render that is interrupted half way should have covered the lines that matter
    most. The greeting is heard on every call; a safe line only when something breaks."""
    assert fixed_lines(pack)[0][1] == pack["greeting"]


def test_every_language_the_agent_can_speak_is_covered(pack):
    langs = {lang for lang, _ in fixed_lines(pack)}
    for expected in ("English", "Hindi", "Telugu", "Tamil", "Marathi"):
        assert expected in langs, expected


def test_it_can_be_narrowed_to_the_languages_a_business_uses(pack):
    """A clinic that only ever hears English and Hindi should not spend twenty minutes
    rendering Odia."""
    lines = fixed_lines(pack, languages={"English", "Hindi"})
    assert {lang for lang, _ in lines} == {"English", "Hindi"}
    assert 0 < len(lines) < len(fixed_lines(pack))


def test_the_same_sentence_is_never_rendered_twice(pack):
    lines = fixed_lines(pack)
    assert len(lines) == len(set(lines))


def test_generated_replies_are_not_pre_rendered(pack):
    """Only FIXED text. A cache of generated sentences would be a cache of one, and
    would grow without bound across calls."""
    texts = [t for _l, t in fixed_lines(pack)]
    assert all(isinstance(t, str) and t.strip() for t in texts)
    assert not any("{" in t and "}" in t for t in texts), (
        "a template with an unfilled placeholder would be rendered as literal braces")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_a_rendered_line_is_a_cache_hit_afterwards(pack):
    voice = FakeVoice()
    tts = CachedTTS(voice)
    prerender(tts, pack, languages={"English"})
    before = len(voice.calls)
    tts.synth(pack["greeting"], None)
    assert len(voice.calls) == before, "the greeting was synthesised again on the hot path"


def test_the_render_is_not_evicted_by_ordinary_traffic(pack):
    """THE ONE THAT MAKES THE FEATURE REAL. 318 fixed lines against a default maxsize
    of 256 means an LRU throws the owner's voice away as fast as it is made — and then
    live traffic grinds down whatever survived."""
    voice = FakeVoice()
    tts = CachedTTS(voice, maxsize=8)
    stats = prerender(tts, pack, languages={"English"})
    assert stats["rendered"] > 20
    for i in range(200):                       # a long call, evicting hard
        tts.synth("a generated sentence %d" % i, None)
    before = len(voice.calls)
    tts.synth(pack["greeting"], None)
    assert len(voice.calls) == before, "the pre-rendered greeting was evicted"


def test_a_script_the_voice_cannot_speak_is_not_a_failure(pack):
    """Kokoro declines Telugu by design rather than producing confident nonsense.
    Counting that as failure makes a healthy render look broken on any multilingual
    pack — and would hide a real failure among the noise."""
    voice = FakeVoice(cannot=("ఫ", "మ", "క"))
    stats = prerender(CachedTTS(voice), pack, languages={"English", "Telugu"})
    assert stats["skipped"] > 0
    assert stats["failed"] == 0
    assert stats["rendered"] > 0


def test_a_declined_line_is_not_pinned_as_silence(pack):
    """Pinning a None would make the cache permanently answer that line with nothing —
    and unlike an LRU entry, it would never age out."""
    voice = FakeVoice(cannot=("Namaste",))
    tts = CachedTTS(voice)
    prerender(tts, pack, languages={"English"})
    before = len(voice.calls)
    tts.synth(pack["greeting"], None)
    assert len(voice.calls) == before + 1, "a line the voice declined was pinned anyway"


def test_one_exploding_line_does_not_stop_the_render(pack):
    lines = fixed_lines(pack, languages={"English"})
    voice = FakeVoice(fail_on={lines[1][1], lines[3][1]})
    stats = prerender(CachedTTS(voice), pack, languages={"English"})
    assert stats["failed"] == 2
    assert stats["rendered"] >= len(lines) - 3, stats


def test_the_render_is_bounded_so_it_cannot_appear_to_hang(pack):
    """It runs while somebody waits for a 'done'. A cloner that turns out to take four
    seconds a line must stop, not run for twenty minutes."""
    import time

    class Slow(FakeVoice):
        def synth(self, text, voice=None):
            time.sleep(0.02)
            return super().synth(text, voice)

    voice = Slow()
    stats = prerender(CachedTTS(voice), pack, budget_s=0.15)
    assert stats["rendered"] < stats["total"], "the budget was ignored"
    assert stats["rendered"] > 0, "nothing was kept from a partial render"


def test_a_partial_render_is_a_partial_improvement(pack):
    """Whatever finished must still be a cache hit — the rest simply falls through to
    the live synthesiser, which is exactly the behaviour that shipped before."""
    voice = FakeVoice()
    tts = CachedTTS(voice)
    prerender(tts, pack, budget_s=0.05)
    before = len(voice.calls)
    tts.synth(pack["greeting"], None)
    assert len(voice.calls) == before, "the first line was not kept"


def test_it_reports_what_it_is_holding(pack):
    """Pinned audio is never evicted, so its size has to be observable rather than
    discovered as memory growth."""
    tts = CachedTTS(FakeVoice())
    assert tts.pinned_bytes() == 0
    prerender(tts, pack, languages={"English"})
    assert tts.pinned_bytes() > 0


def test_a_provider_without_the_pin_kwarg_still_renders(pack):
    """tts.py is a pluggable interface — a custom provider predating `pin` must degrade
    to the ordinary cache, not break the feature."""
    class Old:
        last_skipped_script = False

        def __init__(self):
            self.calls = []

        def synth(self, text, voice=None):     # no `pin`
            self.calls.append(text)
            return b"RIFF" + b"\0" * 60

    old = Old()
    stats = prerender(old, pack, languages={"English"})
    assert stats["rendered"] > 20 and stats["failed"] == 0


# --------------------------------------------------------------------------- #
# verify before pinning — the safety valve for the whole feature
# --------------------------------------------------------------------------- #
class FakeSTT:
    """Stands in for the recogniser. `mangle` makes it hear something else, which is
    what a cloner failing at a language actually looks like from here."""

    def __init__(self, mangle=None, heard=None):
        self.mangle, self.heard = mangle or set(), heard

    def transcribe(self, path):
        text = open(path, "rb").read()[60:].decode("utf-8", "ignore")
        if self.heard is not None:
            return self.heard, "en", 0.9
        if any(m in text for m in self.mangle):
            return "and was myself a lack of your civly infelicit lack", "en", 0.9
        return text, "en", 0.9


class EchoVoice(FakeVoice):
    """Writes the text into the clip so FakeSTT can read it back."""

    def synth(self, text, voice=None, pin=False):
        self.calls.append(text)
        self.last_skipped_script = False
        return b"RIFF" + b"\0" * 56 + text.encode()


def test_a_mangled_line_is_never_pinned(pack):
    """MEASURED, and the reason this exists: VoxCPM asked for Hindi produced fluent
    audio that this project's own recogniser read as English gibberish. A pinned line
    is played on EVERY call and never evicted — a bad render is a permanent defect in
    the greeting with no fallback and nothing to explain it."""
    voice = EchoVoice()
    tts = CachedTTS(voice)
    stats = prerender(tts, pack, languages={"English"},
                      stt=FakeSTT(mangle={pack["greeting"][:20]}))
    assert stats["rejected"] >= 1, stats
    before = len(voice.calls)
    tts.synth(pack["greeting"], None)
    assert len(voice.calls) == before + 1, "the mangled greeting was pinned anyway"


def test_a_good_line_survives_verification(pack):
    voice = EchoVoice()
    tts = CachedTTS(voice)
    stats = prerender(tts, pack, languages={"English"}, stt=FakeSTT())
    assert stats["rejected"] == 0 and stats["rendered"] > 20, stats
    before = len(voice.calls)
    tts.synth(pack["greeting"], None)
    assert len(voice.calls) == before, "a verified line was not pinned"


def test_a_mispronounced_proper_noun_is_not_a_rejection():
    """Measured: the clone said "Suvita" for "Suvidha". That is intelligible and must
    pass — this checks the words are right, it does not score a WER."""
    from zensuvidha.prerender import verified
    ok, why = verified(b"RIFF" + b"\0" * 56,
                       "Thank you for calling Suvidha Clinic, how may I help you today?",
                       FakeSTT(heard="Thank you for calling Suvita Clinic, how may I "
                                     "help you today?"))
    assert ok, why


def test_gibberish_is_a_rejection():
    from zensuvidha.prerender import verified
    ok, _why = verified(b"RIFF" + b"\0" * 56, "नमस्ते, सुविधा क्लीनिक में आपका स्वागत है",
                        FakeSTT(heard="and was myself a lack of your civly infelicit"))
    assert not ok


def test_silence_is_a_rejection():
    from zensuvidha.prerender import verified
    ok, why = verified(b"RIFF" + b"\0" * 56, "hello there", FakeSTT(heard=""))
    assert not ok and "nothing recognisable" in why


def test_no_recogniser_means_unchecked_not_rejected():
    """Refusing to pin anything because the recogniser is unavailable would silently
    disable the whole feature on a text-only install."""
    from zensuvidha.prerender import verified
    ok, why = verified(b"RIFF" + b"\0" * 56, "hello", None)
    assert ok and why == "unchecked"


def test_indic_words_are_split_without_losing_combining_marks():
    """`\\w` drops Devanagari matras — this codebase has been bitten by that twice, and
    verification is exactly the alphabet where it would matter most."""
    from zensuvidha.prerender import _words
    assert _words("मेरा नाम मनु मिश्रा है") == ["मेरा", "नाम", "मनु", "मिश्रा", "है"]


# --------------------------------------------------------------------------- #
# the cloner that runs out of process, because the in-process one cannot load
# --------------------------------------------------------------------------- #
def _stub_worker(tmp_path):
    """A worker satisfying the four-flag contract, in the same interpreter."""
    w = tmp_path / "w.py"
    w.write_text(
        "import argparse,sys,math,struct,wave\n"
        "ap=argparse.ArgumentParser()\n"
        "[ap.add_argument(a,default='') for a in "
        "('--ref','--text','--out','--language','--ref-text')]\n"
        "a=ap.parse_args()\n"
        "sys.exit(4) if 'explode' in a.text else None\n"
        "open(a.out,'wb').write(b'ERROR: no voice') if 'garbage' in a.text else None\n"
        "sys.exit(0) if 'garbage' in a.text else None\n"
        "w=wave.open(a.out,'wb'); w.setnchannels(1); w.setsampwidth(2)\n"
        "w.setframerate(16000)\n"
        "w.writeframes(b''.join(struct.pack('<h',int(8000*math.sin(i/12))) "
        "for i in range(16000))); w.close()\n")
    return w


def _clone(tmp_path, **over):
    import sys

    from zensuvidha.tts import SubprocessCloneTTS
    cfg = {"clone_command": [sys.executable, str(_stub_worker(tmp_path))],
           "reference": "models/.bench_clips/clip0.wav", "clone_timeout": 60}
    cfg.update(over)
    return SubprocessCloneTTS(cfg)


def test_the_subprocess_cloner_produces_audio(tmp_path):
    """The in-process cloner returns None on this install — coqui-tts needs a newer
    transformers than parler-tts allows — so this is the path that actually works."""
    assert len(_clone(tmp_path).synth("hello there")) > 1000


def test_a_worker_that_writes_a_message_instead_of_audio_is_not_forwarded(tmp_path):
    """Forwarding an error string AS audio makes the client fail to decode it, and the
    turn is silent with no reason recorded anywhere."""
    assert _clone(tmp_path).synth("garbage please") is None


def test_a_crashing_worker_is_a_decline_not_an_exception(tmp_path):
    assert _clone(tmp_path).synth("explode now") is None


def test_an_empty_line_never_spawns_a_process(tmp_path):
    assert _clone(tmp_path).synth("   ") is None


def test_a_wedged_worker_gives_up(tmp_path):
    """Same rule as the `say` and whisper.cpp timeouts: a child that never returns holds
    a threadpool worker forever, and enough of them stop the server answering anybody."""
    import sys
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(30)")
    c = _clone(tmp_path, clone_command=[sys.executable, str(slow)], clone_timeout=0.5)
    assert c.synth("anything") is None


def test_it_refuses_to_start_without_a_reference_or_a_worker(tmp_path):
    import pytest as _pt

    from zensuvidha.tts import SubprocessCloneTTS
    with _pt.raises(RuntimeError, match="clone_command"):
        SubprocessCloneTTS({"reference": "models/.bench_clips/clip0.wav"})
    with _pt.raises(RuntimeError, match="reference"):
        _clone(tmp_path, reference="/nope/missing.wav")
