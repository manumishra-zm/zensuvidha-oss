"""The expectation signal — a second opinion on identity, alongside the voiceprint.

The tests that matter most here are the NEGATIVE ones. This module sits on the speaker
gate's refusal path, so the danger is not that it fails to rescue a turn; it is that it
rescues a stranger, or that some future edit lets it reject a caller. Both are pinned.
"""
import pytest

from zensuvidha import expectation as X
from zensuvidha.expectation import expectation_score, should_rescue, RESCUE_AT


PACK = {
    "services": [{"name": "General Physician consultation"},
                 {"name": "Dermatology"},
                 {"name": "Ultrasound scan"}],
    "knowledge": [{"q": "What are your consultation timings?", "a": "9am to 2pm"},
                  {"q": "Do you do vaccinations?", "a": "Yes, routine only"}],
    "common": [{"q": "Where are you located?", "a": "Pune"}],
}
ALIASES = {"doctor": {"Dr Anil Sharma": ["anil", "sharma", "dr sharma"],
                      "Dr Priya Mehta": ["priya", "mehta"]}}


# ── it rescues the cases the voiceprint cannot see ──────────────────────────────

@pytest.mark.parametrize("said", [
    "8920429057",
    "8-9-2-0-4-2-9-0-5-7",
    "my number is 892 042 9057",
    "it's 8920429057 please",
])
def test_the_number_we_asked_for_rescues_the_turn(said):
    """The documented unsolved limit: loud audio drives the caller's score against
    their OWN voice to 0.07. Ten digits arriving right after we asked for a mobile
    number is the caller, whatever the acoustics say."""
    ok, why = should_rescue(said, "phone", PACK, ALIASES)
    assert ok, f"{said!r} should rescue — got {why!r}"
    assert "digit" in why


def test_naming_our_own_doctor_rescues_the_turn():
    ok, why = should_rescue("I want to see Dr Anil Sharma", "doctor", PACK, ALIASES)
    assert ok and "Dr Anil Sharma" in why


def test_two_medium_signals_combine_to_rescue():
    """Scores accumulate, so independent corroboration can reach the bar that no
    single medium signal reaches alone."""
    alone, _ = expectation_score("tomorrow morning", "datetime", None, None)
    assert alone < RESCUE_AT
    both, why = expectation_score(
        "tomorrow morning for the dermatology consultation", "datetime", PACK, None)
    assert both >= RESCUE_AT, why


# ── it must NOT rescue a stranger ───────────────────────────────────────────────

@pytest.mark.parametrize("said", [
    "and then the weather tomorrow will be clear across the region",  # a television
    "yeah so I told him that already",                                # a colleague
    "ooh baby baby, don't you know I love you so",                    # a song
    "please hold the line for the next available agent",              # another device
])
def test_a_bystander_is_not_rescued(said):
    for slot in (None, "phone", "name", "datetime", "doctor"):
        ok, why = should_rescue(said, slot, PACK, ALIASES)
        assert not ok, f"{said!r} rescued at slot={slot}: {why!r}"


def test_our_own_question_echoed_back_is_not_a_name():
    """'What is your name?' said back at us is an ECHO, not an answer."""
    ok, _ = should_rescue("what is your name?", "name", PACK, ALIASES)
    assert not ok


def test_two_candidate_numbers_do_not_rescue():
    """The same rule the phone slot itself applies: two candidates means we cannot
    tell which was meant, so neither is evidence."""
    ok, _ = should_rescue("is it 8920429057 or 9876543210", "phone", PACK, ALIASES)
    assert not ok


def test_a_number_when_we_asked_for_something_else_does_not_rescue():
    ok, _ = should_rescue("8920429057", "name", PACK, ALIASES)
    assert not ok


# ── the safety property: it can never cause a rejection ─────────────────────────

CALLER_UTTERANCES = [
    "hello?", "can you hear me?", "haan", "yes", "ok", "हाँ", "అవును",
    "my son has a fever since last night",
    "I need an appointment",
    "sorry, could you repeat that?",
    "",
    "   ",
    "8920429057",
    "Dr Anil Sharma",
    "a" * 5000,                       # a degenerate transcript that survived trimming
    "!@#$%^&*()",
    "\n\t\r",
]


@pytest.mark.parametrize("said", CALLER_UTTERANCES)
def test_score_is_never_negative_and_never_raises(said):
    """There is no path from this module to a rejection. A caller describing a symptom
    in their own words may score zero — that must mean 'no opinion', never 'refuse'.

    This is the same failure the 0.55 speaker threshold made, in a domain where the
    caller has no way to try harder, so it is pinned rather than assumed.
    """
    for slot in (None, "phone", "name", "datetime", "doctor", "unknown_slot"):
        score, why = expectation_score(said, slot, PACK, ALIASES)
        assert score >= 0.0
        assert isinstance(why, str)
        ok, _ = should_rescue(said, slot, PACK, ALIASES)
        assert ok in (True, False)


def test_should_rescue_is_the_only_verdict_and_it_is_one_directional():
    """`should_rescue` returning False must be indistinguishable from not calling it.
    Pinned structurally: the module exposes no 'reject' verdict to call by mistake."""
    verdicts = [n for n in dir(X)
                if n.startswith(("reject", "should_reject", "deny", "refuse"))]
    assert not verdicts, f"a rejection path appeared: {verdicts}"


# ── robustness of the inputs it will actually be handed ─────────────────────────

def test_survives_a_missing_or_malformed_pack():
    for pack in (None, {}, {"services": None, "knowledge": None},
                 {"services": [{}], "knowledge": [{}]}):
        score, _ = expectation_score("8920429057", "phone", pack, None)
        assert score >= X.STRONG          # the slot rule works without a pack at all


def test_survives_missing_or_malformed_aliases():
    for al in (None, {}, {"doctor": None}, {"doctor": {"Dr X": None}}):
        ok, _ = should_rescue("Dr X", "doctor", PACK, al)
        assert ok in (True, False)


def test_alias_matching_is_whole_word():
    """A pack entry like 'Rao' must not be satisfied by the 'rao' inside another word."""
    al = {"doctor": {"Dr Rao": ["rao"]}}
    ok, _ = should_rescue("the tarot reading was accurate", "doctor", PACK, al)
    assert not ok


def test_pack_vocab_is_cached_on_the_pack_not_by_id():
    """An id()-keyed cache can hand one pack another pack's vocabulary after a free.
    Caching on the dict makes that impossible and lets it die with the pack."""
    p = dict(PACK)
    first = X._pack_vocab(p)
    assert X._VOCAB_KEY in p
    assert X._pack_vocab(p) is first
    other = X._pack_vocab({"services": [{"name": "Haircut and colour"}]})
    assert other != first


def test_cached_vocab_key_cannot_reach_the_prompt():
    """The cache lives on the pack dict, which is also what builds the system prompt.
    It is only safe because the prompt reads named keys — so pin that the key is
    private and that nothing enumerates the pack."""
    p = dict(PACK)
    X._pack_vocab(p)
    assert X._VOCAB_KEY.startswith("_")
    from zensuvidha import orchestrator
    src = open(orchestrator.__file__, encoding="utf-8").read()
    for pattern in ("pack.items()", "pack.keys()", "**pack"):
        assert pattern not in src, f"{pattern} would leak the cache key into the prompt"


def test_a_read_only_pack_does_not_break_it():
    class Frozen(dict):
        def __setitem__(self, k, v):
            raise TypeError("read-only")
    p = Frozen(PACK)
    assert X._pack_vocab(p) == X._pack_vocab(dict(PACK))


# ── the ordering this whole feature depends on ─────────────────────────────────

def test_pending_slot_is_still_live_when_the_gate_runs():
    """The rescue reads `pending_slot`, which we set while generating the PREVIOUS
    turn's reply. It is cleared by `_answer_to_pending`, which is reached from
    `begin_user` — i.e. from `_stream_turn`, which runs AFTER `check_speaker`.

    If a refactor ever moves the gate after slot capture, `pending_slot` would be None
    at gate time and the phone rescue would silently stop firing — no error, no failing
    assertion anywhere else, just a feature that quietly does nothing. Hence this pin.
    """
    import inspect
    from zensuvidha import server

    src = inspect.getsource(server)
    # Anchored on the DISPATCH, not on a signature. The first version matched the exact
    # argument list of `_stream_turn`, so adding a parameter to it broke this test for
    # a reason that had nothing to do with the ordering it exists to protect.
    gate_at = src.index("session.check_speaker")
    turn_at = src.index("await launch(heard, heard")
    assert gate_at < turn_at, (
        "the speaker gate must run BEFORE the turn is launched, or pending_slot is "
        "already cleared and the expectation rescue can never fire")


def test_nothing_unexpected_clears_pending_slot():
    """Only three places may touch it, and each is safe for a different reason:

      __init__            – a new call starts with nothing pending.
      finalize            – ends turn N and decides what turn N+1 is waiting for.
                            Both the clear and the set live here, which is why it is
                            the SOURCE of the value the gate reads, not a threat to it.
      _answer_to_pending  – consumes a slot the caller just filled. Reached from
                            begin_user, i.e. AFTER check_speaker on that same turn.

    A fourth clearer appearing is the failure mode this pins: it would most likely run
    before the gate, leaving pending_slot None at rescue time, and the feature would
    silently do nothing — no error, no other failing test.
    """
    import inspect
    from zensuvidha.orchestrator import Session

    clearing = {name for name, fn in inspect.getmembers(Session, inspect.isfunction)
                if "self.pending_slot = None" in (inspect.getsource(fn) or "")}
    assert clearing == {"__init__", "finalize", "_answer_to_pending"}, (
        f"pending_slot is cleared somewhere new: {clearing - {chr(0)}} — check it "
        f"cannot run before check_speaker")


# ── bounded cost ───────────────────────────────────────────────────────────────

def test_input_is_bounded():
    """This runs on the refusal path of every turn, so its cost must not depend on how
    degenerate a transcript the STT trimmer let through."""
    import time
    huge = "8920429057 " + ("x" * 200_000)
    t = time.perf_counter()
    expectation_score(huge, "phone", PACK, ALIASES)
    assert (time.perf_counter() - t) < 0.05, "cost grew with a degenerate transcript"


def test_the_bound_does_not_change_a_normal_turn():
    normal = "my number is 8920429057"
    assert len(normal) < X.MAX_SCAN
    assert expectation_score(normal, "phone", PACK, ALIASES)[0] >= X.STRONG


# ── what an end-to-end probe found: pending_slot alone is not enough ────────────

def _clinic_session():
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    return Session(load_pack("clinic"), None)


def test_pending_slot_is_none_for_most_of_a_booking():
    """The reason `last_asked_slot` exists at all.

    `finalize` deliberately leaves `pending_slot` None whenever more than one field is
    outstanding — filing an answer against the wrong slot is worse than not filing it.
    Correct for collection, and it meant the rescue would almost never have fired,
    because a booking spends most of its turns with several fields missing.
    """
    s = _clinic_session()
    s.booking_started, s.slots = True, {"name": "Manu Mishra"}
    s.finalize('{"kind":"answer","action":{"type":"collect"},'
               '"say":"What mobile number should we use for confirmation?"}')
    assert s.pending_slot is None, "if this ever holds a value, re-read the rescue wiring"
    assert s.last_asked_slot == "phone", "the rescue would have had nothing to work with"


def test_asked_which_slot_matches_the_packs_own_wording_exactly():
    s = _clinic_session()
    q = (s.pack.get("booking", {}).get("slots") or {})["phone"]
    assert s.asked_which_slot(q) == "phone"


def test_asked_which_slot_tolerates_the_model_rewording_it():
    s = _clinic_session()
    assert s.asked_which_slot("May I have your mobile number for confirmation?") == "phone"


@pytest.mark.parametrize("line", [
    "We have several doctors available today across many specialities",
    "Thank you, your appointment is confirmed",
    "Hello, how can I help you today?",
    "I'm sorry, I don't have that detail",
    "There's continuous audio in the background, could you say that again?",
    "We are open Monday to Saturday from 9am to 2pm",
    "The consultation fee is 500 rupees",
])
def test_asked_which_slot_returns_nothing_when_we_did_not_ask(line):
    """A WRONG slot is the only way this can hurt: it would have the rescue look for
    the wrong shape, and a bystander reciting digits could then be let through. Failing
    to match costs nothing, so the tie-break is deliberately strict."""
    assert _clinic_session().asked_which_slot(line) is None


def test_a_tie_between_two_slots_yields_no_answer():
    """Two fields scoring alike means we cannot tell which was asked."""
    s = _clinic_session()
    s.pack = dict(s.pack)
    s.pack["booking"] = dict(s.pack["booking"])
    s.pack["booking"]["slots"] = {"phone": "please tell me the detail",
                                  "name": "please tell me the detail"}
    s.pack["booking"]["required"] = ["phone", "name"]
    assert s.asked_which_slot("please tell me the detail now") is None



# a stub encoder, so this file needs no 80MB download and no sibling test module
class _Stub:
    backend, threshold = "stub", 0.55

    def __init__(self):
        self._v = {}

    def embed(self, audio, min_seconds=0.6):
        import numpy as np
        k = audio if isinstance(audio, str) else bytes(audio).decode("utf8", "ignore")
        if k not in self._v:
            r = np.random.default_rng(abs(hash(k)) % 2 ** 31)
            v = r.normal(size=192).astype("float32")
            self._v[k] = v / np.linalg.norm(v)
        return self._v[k]

    def similarity(self, a, b):
        import numpy as np
        return float(np.dot(a, b))

    def matches(self, print_, audio, **kw):
        v = self.embed(audio)
        if v is None:
            return True, None
        sim = self.similarity(print_, v)
        return sim >= self.threshold, sim

    def judge(self, print_, audio):
        ok, sim = self.matches(print_, audio)
        return ok, sim, self.embed(audio)


def _corroborated_session(who="caller"):
    """A session whose voiceprint the caller has actually confirmed — the gate refuses
    nobody until then, so a test about refusal must establish one first."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), None, speaker_gate=_Stub())
    for _ in range(s.VOICEPRINT_TRUST_N + 1):
        s.check_speaker(who, speakers=1)
    assert s._gate_proven
    return s


# ══════════════════════════════════════════════════════════════════════════════
# CONTAINMENT — the two things this must never do:
#   (a) cause the caller's real voice or words to be filtered/removed
#   (b) let a TEXT judgement feed back into the AUDIO subsystem
#
# Both are pinned behaviourally, not by reading the code, because the danger is a
# future edit that looks harmless.
# ══════════════════════════════════════════════════════════════════════════════

def test_the_module_cannot_touch_audio_or_a_model_at_all():
    """Its only import is `re`. No audio buffer, no encoder, no LLM, no numpy.

    This is the structural guarantee behind everything below: a module that cannot
    reach the audio path cannot degrade it, whatever its scoring does.
    """
    import ast, pathlib
    tree = ast.parse(pathlib.Path(X.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"re", "__future__"}, f"expectation.py grew an import: {imported}"

    banned = ("numpy", "soundfile", "torch", "speechbrain", "sherpa", "librosa",
              "llm", "ollama", "diarize", "speaker", "pipeline", "stt")
    src = pathlib.Path(X.__file__).read_text().lower()
    for name in banned:
        assert f"import {name}" not in src, f"expectation.py reached for {name}"


def test_it_never_receives_audio_so_it_cannot_alter_it():
    """No parameter carries a signal. The rescue is handed a transcript, which STT has
    already produced — it is downstream of every audio decision and cannot revisit one."""
    import inspect
    params = set(inspect.signature(X.should_rescue).parameters)
    assert params == {"text", "pending_slot", "pack", "aliases"}
    assert not (params & {"audio", "pcm", "wav", "samples", "signal"})


def test_a_rescue_does_not_move_the_voiceprint():
    """THE feedback loop to avoid. Accepting a turn on TEXT evidence and then folding
    its AUDIO into the voiceprint would let a wrong-but-well-worded turn drag the
    caller's acoustic identity — and the voiceprint is what isolation trims against, so
    the damage would spread into what future turns keep."""
    import numpy as np

    s = _corroborated_session()
    s.pending_slot = "phone"
    before = np.array(s.voiceprint, copy=True)
    n_before = s._voiceprint_n

    ok, _ = s.check_speaker("a voice that is NOT the caller", heard="8920429057")
    assert ok, "precondition: the rescue fired"
    assert np.array_equal(s.voiceprint, before), "the rescue moved the voiceprint"
    assert s._voiceprint_n == n_before, "the rescue corroborated the print"


def test_a_run_of_rescues_may_REPAIR_the_print_but_only_under_the_existing_guard():
    """The one indirect path from text back to audio — found by this test, not by
    reading the code, and worth stating precisely rather than pretending it is absent.

    A single rescue never moves the voiceprint (above). But rescued turns deliberately
    still count toward `_gate_rejects`, so after REENROL_AFTER of them the PRE-EXISTING
    repair fires and re-enrols. That is the intended outcome for the case this feature
    exists for: the caller's print was destroyed by loud audio, they keep answering our
    questions, and the print gets rebuilt from how they actually sound now. Suppressing
    it would trade a permanent fix for a per-turn rescue.

    It is not a new hole. The repair has always required the SAME voice refused
    repeatedly, and it still refuses to learn from a clip known to hold more than one
    voice — a blend of two people must never become the caller's identity. Both halves
    are pinned here, because a future edit relaxing either would matter.
    """
    import numpy as np

    # (a) with one voice in the clip, a run of rescues repairs the print
    s = _corroborated_session()
    s.pending_slot = "phone"
    before = np.array(s.voiceprint, copy=True)
    for _ in range(s.REENROL_AFTER + 1):
        s.check_speaker("one consistent voice", speakers=1, heard="8920429057")
    assert not np.array_equal(s.voiceprint, before), (
        "the repair never fired — a caller whose print was destroyed would be stuck "
        "depending on every future turn happening to match an expectation")

    # (b) with TWO voices in the clip it must refuse to learn, rescued or not
    s2 = _corroborated_session()
    s2.pending_slot = "phone"
    before2 = np.array(s2.voiceprint, copy=True)
    for _ in range(s2.REENROL_AFTER + 3):
        s2.check_speaker("a mixture of two people", speakers=2, heard="8920429057")
    assert np.array_equal(s2.voiceprint, before2), (
        "a clip holding two voices was enrolled as the caller — the rescue must not "
        "weaken the guard that stops a blend becoming somebody's identity")


def test_a_rescue_never_causes_more_audio_to_be_removed():
    """Isolation trims against the voiceprint, so the only way this feature could cost
    the caller their own words is by moving that print on a single turn. It does not."""
    import numpy as np

    s = _corroborated_session()
    s.pending_slot = "phone"
    before = np.array(s.voiceprint, copy=True)
    ok, _ = s.check_speaker("some other voice", speakers=1, heard="8920429057")
    assert ok
    assert np.array_equal(s.voiceprint, before)


def test_adding_a_transcript_can_only_turn_a_NO_into_a_YES():
    """Monotonicity, over the whole corpus. `heard` may flip False→True and must never
    flip True→False, so no caller can lose a turn they would otherwise have kept."""
    voices = ["caller", "stranger", "another stranger"]
    texts = [None, "", "8920429057", "Dr Anil Sharma", "haan", "my son has a fever",
             "and now the weather", "!@#$", "a" * 3000]
    for voice in voices:
        for slot in (None, "phone", "name", "datetime", "doctor"):
            base = _corroborated_session()
            base.pending_slot = slot
            without, _ = base.check_speaker(voice)
            for t in texts:
                s = _corroborated_session()
                s.pending_slot = slot
                with_text, _ = s.check_speaker(voice, heard=t)
                assert with_text or not without, (
                    f"heard={t!r} turned an ACCEPT into a REFUSAL "
                    f"(voice={voice!r}, slot={slot!r})")


def test_the_rescue_runs_after_stt_so_it_cannot_affect_what_whisper_saw():
    """Ordering, pinned in the server source. Isolation and denoise both run inside
    `pipeline.prepare`, before transcription; the gate — and therefore the rescue —
    only sees the transcript afterwards. There is no path back."""
    import inspect
    from zensuvidha import server
    src = inspect.getsource(server)
    prepare_at = src.index("session.clean_audio")     # isolation + denoise
    gate_at = src.index("session.check_speaker")      # …then, and only then, identity
    assert prepare_at < gate_at, (
        "the speaker gate must run AFTER the audio pipeline, or a text judgement "
        "could reach back into filtering")
