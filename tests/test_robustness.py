"""The audit's remaining findings, each pinned by the failure it caused.

Grouped by what the caller experiences, because that is what decides severity:

  * THE CALL WEDGES  — the caller talks and nothing ever comes back
  * THE CALL LIES    — the transcript/history/DB disagree with what was said
  * THE CALL LEAKS   — PII or the filesystem escapes the process
  * THE CALL DRAGS   — avoidable latency on the critical path

Run:  pytest -q tests/test_robustness.py
"""
import io

import numpy as np
import pytest
import soundfile as sf

from zensuvidha.orchestrator import Session
from zensuvidha.packs import load_pack

SR = 16000


def wav(x):
    b = io.BytesIO()
    sf.write(b, np.asarray(x, dtype="float32"), SR, format="WAV", subtype="PCM_16")
    return b.getvalue()


# --------------------------------------------------------------------------- #
# THE CALL WEDGES
# --------------------------------------------------------------------------- #
def test_a_wedged_say_daemon_cannot_hang_a_turn_forever():
    """macOS `say` talks to a system daemon that can wedge. With no timeout the call
    never returns: the caller hears nothing for the REST of the call and the
    threadpool worker is consumed permanently — enough of those and the server stops
    answering everyone, not just this caller."""
    import inspect

    from zensuvidha import tts
    src = inspect.getsource(tts.SystemTTS._mac_say)
    assert "timeout=" in src, "`say` is invoked with no timeout"
    assert tts.SAY_TIMEOUT_S > 0


def test_a_decompression_bomb_cannot_stall_every_call():
    """The 8MB frame cap counts COMPRESSED bytes. A small FLAC decodes to hundreds of
    megabytes of float32 and stalls STT for every concurrent call on the box."""
    from zensuvidha.stt import MAX_DECODE_S, FasterWhisperSTT

    b = io.BytesIO()
    sf.write(b, np.zeros(SR * (MAX_DECODE_S + 120), dtype="float32"), SR, format="FLAC")
    raw = b.getvalue()
    assert len(raw) < 20_000_000, "test fixture is not a compression bomb"

    class Bare(FasterWhisperSTT):
        def __init__(self):
            pass

    out = Bare()._decode(raw)
    assert len(out) / SR <= MAX_DECODE_S + 1, \
        f"decoded {len(out)/SR:.0f}s — the cap did not apply"


def test_a_phone_number_with_repeated_digits_is_not_thrown_away():
    """The degeneracy guard condemned the WHOLE turn on any 6-character run, anywhere.
    "my number is 8888884321" has one. So does a drawn-out "haaaaaan"."""
    from zensuvidha.stt import _looks_degenerate

    for real in ("my number is 8888884321", "मेरा नंबर 9999998888 है",
                 "haaaaaan theek hai", "call me on 7777776543"):
        assert not _looks_degenerate(real), f"a real utterance was condemned: {real!r}"


def test_genuine_whisper_degeneracy_is_still_caught():
    from zensuvidha.stt import _looks_degenerate

    for junk in ("5 5 5 5 5 5 5", "तो अचाएज़" + "़" * 120, "अ" + "ऽ" * 40):
        assert _looks_degenerate(junk), f"degenerate output slipped through: {junk[:30]!r}"


def test_a_transcript_with_a_runaway_tail_keeps_the_caller_words():
    """Whisper often produces a real sentence and THEN latches onto a combining mark.
    Dropping the turn threw the phone number away along with the artifact."""
    from zensuvidha.stt import _looks_degenerate, _trim_degenerate

    raw = "मेरा नंबर है 8920429057" + "ऽ" * 40
    fixed = _trim_degenerate(raw)
    assert "8920429057" in fixed, "the caller's number was lost with the artifact"
    assert not _looks_degenerate(fixed)


def test_history_cannot_grow_without_bound_on_paths_that_never_finalize():
    """Trimming lived only in finalize(), so escalations, unoffered services and
    barge-ins grew the prompt forever. Eventually it overflows num_ctx, Ollama drops
    the OLDEST messages — the system prompt — and the agent loses its instructions
    mid-call and starts looping."""
    s = Session(load_pack("clinic"), None)
    s.messages = [{"role": "system", "content": "sys"}]
    for i in range(200):
        s._append("user", f"turn {i}")
        s._append("assistant", f"reply {i}")
    assert len(s.messages) <= 1 + s.MAX_TURNS * 2, f"history grew to {len(s.messages)}"
    assert s.messages[0]["role"] == "system", "the system prompt was trimmed away"


def test_note_interrupted_also_stays_bounded():
    s = Session(load_pack("clinic"), None)
    s.messages = [{"role": "system", "content": "sys"}]
    for i in range(200):
        s._append("user", f"turn {i}")
        s.note_interrupted()
    assert len(s.messages) <= 1 + s.MAX_TURNS * 2


# --------------------------------------------------------------------------- #
# THE CALL LIES
# --------------------------------------------------------------------------- #
def test_a_blocked_reply_is_not_recorded_as_if_it_had_been_spoken():
    """The guard's whole purpose is that the caller never hears the rejected sentence.
    Recording it in the transcript, the history and the turns table hands it to
    everyone who reads the call afterwards — and to the model on the next turn."""
    s = Session(load_pack("clinic"), None)
    s.messages = [{"role": "system", "content": "sys"}]
    blocked = '{"kind":"answer","say":"The fee is 9999 rupees","action":{"type":"none"}}'
    spoken = "May I have the patient's full name?"

    res = s.finalize(blocked, None, spoken)
    assert res["say"] == spoken, "the transcript kept the words the guard rejected"
    assert "9999" not in res["say"]
    assert "9999" not in s.messages[-1]["content"], \
        "the rejected reply went into the history the model sees next turn"


@pytest.mark.parametrize("override", [None, ""])
def test_an_unblocked_reply_is_recorded_exactly_as_generated(override):
    # A fresh session per case: finalizing the same words twice on ONE session trips the
    # repetition guard, which is correct behaviour and not what this test is about.
    s = Session(load_pack("clinic"), None)
    s.messages = [{"role": "system", "content": "sys"}]
    raw = '{"kind":"answer","say":"We are open until 8pm.","action":{"type":"none"}}'
    assert s.finalize(raw, None, override)["say"] == "We are open until 8pm.", \
        "an absent override must not disturb a reply the guard allowed"


def test_a_cached_line_does_not_claim_the_voice_could_not_speak_it():
    from zensuvidha.tts import CachedTTS

    class Inner:
        last_skipped_script = False

        def synth(self, text, voice=None):
            return b"RIFFfake"

    c = CachedTTS(Inner())
    c.synth("hello")
    c.last_skipped_script = True                 # stale from an earlier failure
    c.synth("hello")                             # cache hit
    assert c.last_skipped_script is False


def test_the_mute_reason_reaches_the_session_through_the_cache():
    """CachedTTS swallowed the flag, so mute_reason was ALWAYS None and a caller whose
    script the voice cannot pronounce got silence with no explanation anywhere."""
    from zensuvidha.tts import CachedTTS

    class Inner:
        last_skipped_script = False

        def synth(self, text, voice=None):
            self.last_skipped_script = True
            return None

    s = Session(load_pack("clinic"), None, tts=CachedTTS(Inner()))
    s.voice = None
    s.tts_bytes("नमस्ते")
    assert s.mute_reason == "no_voice_for_script"


# --------------------------------------------------------------------------- #
# THE CALL LEAKS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["../../etc/passwd", "../config", "..", "_base",
                                  "clinic/../../secrets", "", "a" * 200,
                                  "clinic\x00.yaml", "/etc/hosts"])
def test_a_pack_name_cannot_reach_outside_the_pack_directory(name):
    """`pack` is a URL query param and a WS message, so it is caller-controlled. Any
    .yaml on disk could be loaded — and whatever sits under its `greeting:` key was
    then SPOKEN down the phone."""
    with pytest.raises(FileNotFoundError):
        load_pack(name)


def test_real_packs_still_load():
    assert load_pack("clinic")["id"] == "clinic"


def test_a_negative_limit_cannot_dump_the_whole_transcript_table():
    """min(limit, 1000) does not bound a negative number, and SQLite reads LIMIT -1 as
    'no limit' — ?limit=-1 returned every caller's transcript."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.transcripts)
    assert "max(1, min(" in src, "the limit is still not bounded from below"


def test_a_resumed_call_still_counts_against_capacity():
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    resume_branch = src.split("if resumed:")[1].split("else:")[0]
    assert "MAX_SESSIONS" in resume_branch, \
        "the resume path skips the capacity check, so the cap is walkable"
    assert server.RESUME_HEADROOM >= 0


def test_caller_words_stay_out_of_the_log_when_transcripts_are_off():
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    for marker in ('log.info("mic turn: %s bytes', 'log.info("spec STT: %s bytes'):
        line = src.split(marker)[1][:260]
        assert "LOG_TRANSCRIPTS" in line, \
            f"{marker!r} writes the caller's words regardless of the setting"


# --------------------------------------------------------------------------- #
# THE CALL DRAGS
# --------------------------------------------------------------------------- #
def test_the_llm_is_warmed_at_the_context_it_will_actually_serve():
    """Ollama sizes the KV cache from num_ctx at LOAD time. Warming at the default and
    serving at 12288 makes the first real turn reload the model — measured 1.99s vs
    0.26s, so the warmup cost an extra load instead of saving one."""
    import inspect

    from zensuvidha.llm import OllamaLLM
    src = inspect.getsource(OllamaLLM.warmup)
    assert "num_ctx" in src, "warmup does not pin num_ctx"


def test_the_speaker_gate_embeds_once_per_turn_not_twice():
    """The ECAPA encoder is the most expensive thing in the gate, and the rejection
    path needed the same vector again."""
    calls = {"n": 0}

    class Gate:
        threshold = 0.55

        def embed(self, audio, min_seconds=0.6):
            calls["n"] += 1
            return np.array([0.0, 1.0], dtype="float32")

        def similarity(self, a, b):
            return 0.01

        def judge(self, print_, audio):
            vec = self.embed(audio)
            return False, 0.01, vec

        def matches(self, print_, audio):
            ok, sim, _ = self.judge(print_, audio)
            return ok, sim

    s = Session(load_pack("clinic"), None, speaker_gate=Gate())
    s.voiceprint = np.array([1.0, 0.0], dtype="float32")
    s.check_speaker(wav(np.zeros(SR, dtype="float32")))
    assert calls["n"] == 1, f"the encoder ran {calls['n']} times for one verdict"


def test_an_older_gate_without_judge_still_works():
    """The gate is swappable; adding judge() must not become a hard requirement."""
    class OldGate:
        threshold = 0.55

        def embed(self, audio, min_seconds=0.6):
            return np.array([0.0, 1.0], dtype="float32")

        def similarity(self, a, b):
            return 0.01

        def matches(self, print_, audio):
            return False, 0.01

    s = Session(load_pack("clinic"), None, speaker_gate=OldGate())
    s.voiceprint = np.array([1.0, 0.0], dtype="float32")
    # Both gates opened by hand: it refuses nobody until it has MATCHED the caller once
    # and the print has been CORROBORATED. An old gate cannot report a rival either (no
    # judge(), so no embedding comes back), which makes both phases pure fail-open for it.
    s._voiceprint_n, s._gate_proven = s.VOICEPRINT_TRUST_N, True
    ok, sim = s.check_speaker(wav(np.zeros(SR, dtype="float32")))
    assert ok is False and sim == pytest.approx(0.01)


def test_the_degeneracy_retry_does_not_run_while_the_stream_is_open():
    """Ollama serialises requests to one model, so a retry issued from inside the
    still-open stream queues behind the generation it has already given up on — the
    fast recovery could not start until the doomed reply had finished looping."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server._stream_turn)
    body = src.split("reply degenerated into repetition")[1].split("async def")[0]
    retry_line = body.split("retry_short")[0]
    assert "break" in retry_line, \
        "the retry still fires from inside the async-for over the live stream"


# --------------------------------------------------------------------------- #
# TTS routing — the dominant latency, and a way to speak confident nonsense
# --------------------------------------------------------------------------- #
def test_kokoro_declines_scripts_it_cannot_speak():
    """Fed Telugu, an English Kokoro pipeline produced 2.1MB and 6.5 SECONDS of audio
    for one short sentence. The caller hears fluent nonsense, which is worse than
    silence and worse than a fallback. Only the languages it was trained on."""
    from zensuvidha.tts import KokoroTTS

    route = KokoroTTS._route
    fake = type("F", (), {"_CAN_SPEAK": KokoroTTS._CAN_SPEAK, "lang": "a",
                          "_LANG_VOICE": KokoroTTS._LANG_VOICE})()
    assert route(fake, "We are open until eight.") == ("a", None)
    assert route(fake, "हम नौ बजे से खुले हैं।") == ("h", "hf_alpha")
    for unsupported in ("మేము తెరిచి ఉంటాము.", "நாங்கள் திறந்திருக்கிறோம்.",
                        "ನಾವು ತೆರೆದಿದ್ದೇವೆ.", "আমরা খোলা আছি।"):
        assert route(fake, unsupported) is None, \
            f"kokoro would have attempted {unsupported[:20]!r}"


def test_a_declined_script_falls_through_to_the_other_provider():
    from zensuvidha.tts import FallbackTTS

    class Primary:
        last_skipped_script = False

        def synth(self, text, voice=None):
            self.last_skipped_script = not text.isascii()
            return None if self.last_skipped_script else b"PRIMARY"

    class Secondary:
        last_skipped_script = False

        def synth(self, text, voice=None):
            return b"SECONDARY"

    t = FallbackTTS(Primary(), Secondary())
    assert t.synth("hello") == b"PRIMARY", "the fast provider was skipped"
    assert t.synth("మేము") == b"SECONDARY", "a declined script was left silent"
    assert t.last_skipped_script is False


def test_a_primary_returning_none_for_a_REAL_failure_is_not_routed_away():
    """Only an explicit 'not my script' may fall through. A provider that simply
    failed must not silently hand every line to the fallback for the rest of the
    call — that hides a broken voice behind a working one."""
    from zensuvidha.tts import FallbackTTS

    class Broken:
        last_skipped_script = False        # never claims the script was the problem

        def synth(self, text, voice=None):
            return None

    class Secondary:
        last_skipped_script = False

        def synth(self, text, voice=None):
            return b"SECONDARY"

    assert FallbackTTS(Broken(), Secondary()).synth("hello") is None


def test_both_providers_declining_sets_the_flag_the_ui_reads():
    from zensuvidha.tts import FallbackTTS

    class Nope:
        last_skipped_script = False

        def synth(self, text, voice=None):
            self.last_skipped_script = True
            return None

    t = FallbackTTS(Nope(), Nope())
    assert t.synth("മലയാളം") is None
    assert t.last_skipped_script is True, "the UI cannot explain the silence"


# --------------------------------------------------------------------------- #
# KNOWLEDGE & GROUNDING — the agent must answer, and only from what it was given
# --------------------------------------------------------------------------- #
def test_numbers_the_caller_SPOKE_count_as_grounded():
    """People say numbers as words on the phone. The guard only collected digits, so
    "my baby is six months old" left "6" ungrounded — the model's correct answer
    ("vaccinations at 6 months") was rejected and the caller got a slot question
    instead. Observed live."""
    from zensuvidha.orchestrator import spoken_numbers

    assert "6" in spoken_numbers("my baby is six months old and needs shots")
    assert "10" in spoken_numbers("मुझे दस बजे सुबह चाहिए")
    assert "9" in spoken_numbers("తొమ్మిది గంటలకు")
    assert spoken_numbers("no numbers here at all") == set()


def test_a_spoken_number_reaches_the_grounding_check():
    s = Session(load_pack("clinic"), None)
    s.begin_user("my baby is six months old")
    assert "6" in s.allowed_numbers(), "the caller said it, so the reply may repeat it"


@pytest.mark.parametrize("value", [
    "I need a medical fitness certificate",
    "do you have a female staff for the examination",
    "मुझे अपॉइंटमेंट चाहिए",
    "how much is an x-ray",
    "can I get an appointment on Sunday",
])
def test_a_question_is_never_filed_as_the_patients_name(value):
    """Observed live: a caller's own question was stored as their name, and the rest of
    the call was conducted as if that were who was ringing."""
    assert not Session(load_pack("clinic"), None)._plausible_slot("name", value)


@pytest.mark.parametrize("value", ["Manu Mishra", "मनु मिश्रा", "Ram Iyer",
                                   "Priya", "మను మిశ్రా", "Anil Kumar Sharma"])
def test_real_names_are_still_accepted(value):
    assert Session(load_pack("clinic"), None)._plausible_slot("name", value)


def test_the_clinic_pack_can_answer_what_callers_actually_ask():
    """Probing with 30 real questions found the agent INVENTING answers — SMS
    reminders, fitness certificates, travel vaccinations, an ECG price borrowed from
    home collection — none of which were in the pack. A model with no fact produces a
    plausible one, so the fix is to have the fact."""
    import yaml

    pack = yaml.safe_load(open("packs/clinic.yaml"))
    blob = " ".join(f"{e.get('q','')} {e.get('a','')} {' '.join(e.get('tags',[]))}"
                    for e in pack["knowledge"]).lower()
    for topic in ["ecg", "x-ray", "ultrasound", "consultation usually takes",
                  "reminder", "certificate", "travel", "old report", "someone else",
                  "one patient", "bus stop", "follow-up"]:
        assert topic in blob, f"callers ask about {topic!r} and the pack cannot answer"


def test_every_knowledge_entry_is_written_in_all_three_languages():
    """A missing a_hi/a_te means the model TRANSLATES instead of quoting — measured at
    27s vs 5s for a Hindi fee question, and it invents while translating."""
    import yaml

    pack = yaml.safe_load(open("packs/clinic.yaml"))
    missing = [e["q"] for e in pack["knowledge"]
               if not (e.get("a_hi") and e.get("a_te"))]
    assert len(missing) <= 4, f"entries with no native-script answer: {missing}"


def test_no_knowledge_answer_mixes_scripts():
    """I have shipped Devanagari 'न' inside a Telugu string before. It looks right and
    is unreadable to the caller."""
    import yaml

    pack = yaml.safe_load(open("packs/clinic.yaml"))
    bad = []
    for e in pack["knowledge"]:
        for field, lo, hi in (("a_hi", 0x900, 0x97F), ("a_te", 0xC00, 0xC7F)):
            text = e.get(field) or ""
            wrong = sum(1 for c in text
                        if c.isalpha() and ord(c) > 0x7F and not lo <= ord(c) <= hi)
            if wrong > 2:
                bad.append(f"{e['q'][:40]} [{field}] {wrong} foreign chars")
    assert not bad, bad


def test_the_base_pack_does_not_invite_the_model_to_guess():
    """_base.yaml used to say "give price/time ranges when unsure", which is an
    instruction to invent — a range is a made-up number with a second made-up number
    beside it, and the caller hears it as the answer."""
    import yaml

    base = yaml.safe_load(open("packs/_base.yaml"))
    joined = " ".join(base.get("policies", [])).lower()
    assert "ranges when unsure" not in joined
    assert "never" in joined or "do not" in joined


# --------------------------------------------------------------------------- #
# COLD START — what the FIRST caller pays
# --------------------------------------------------------------------------- #
def test_the_llm_is_warmed_with_the_real_prompt_not_a_stub():
    """Loading the weights is the cheap half. The clinic pack's system prompt is ~6,000
    tokens (the whole knowledge base sits in a deliberately cache-stable prefix), and
    Ollama must EVALUATE it. Warming with "hi" cached a two-token prefix no real turn
    shares, so the first caller paid the lot. Measured: first turn 27.3s -> 5.5s."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.lifespan)
    assert "_prime_prompt" in src, "startup no longer builds the real warmup prompt"
    assert "call_messages" in src


def test_warmup_accepts_a_real_message_list():
    import inspect

    from zensuvidha.llm import OllamaLLM
    sig = inspect.signature(OllamaLLM.warmup)
    assert "messages" in sig.parameters


def test_a_redundant_model_switch_does_not_re_warm():
    """A browser sends `model` on every connect with whatever is in the dropdown —
    usually the default. Re-warming it burned CPU against the caller's own first turn,
    and before the prompt fix it EVICTED the cached prefix as well."""
    import inspect

    from zensuvidha import server
    branch = inspect.getsource(server.ws).split('elif mtype == "model"')[1][:1400]
    guard = branch.split("if ")[1][:200] if "if " in branch else ""
    assert "session.model != was" in guard, f"it re-warms on a no-op switch: {guard!r}"
    assert 'getattr(_llm, "model"' in guard, \
        f"it re-warms the model the server already has loaded: {guard!r}"


def test_the_startup_precache_is_bounded():
    """Warming every filler phrase against every pack's voice was 209 synth calls,
    ~84s of CPU, competing with the first real callers for exactly that long — and six
    of every seven were identical, because a provider ignores voice ids that are not
    its own. One phrase per language, in the voice this server actually greets with."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.lifespan)
    body = src.split("_FILLERS.values()")[1][:400]
    assert "phrases[0]" in body, "the precache is fanning out over every phrase again"
    assert "default_voice" in body, "it warms voices this server will never speak with"


def test_the_first_filler_is_the_one_that_was_precached():
    """_pick_filler chose at random while only phrases[0] was warmed, so the line whose
    entire purpose is covering a slow reply was itself synthesised cold."""
    from zensuvidha.server import _FILLERS, _filler_used, _pick_filler

    _filler_used.clear()
    for lang, phrases in _FILLERS.items():
        assert _pick_filler(lang) == phrases[0], f"{lang} first filler is not cached"
    _filler_used.clear()


def test_an_unsupported_language_still_gets_no_english_filler():
    from zensuvidha.server import _filler_used, _pick_filler

    _filler_used.clear()
    assert _pick_filler("Basque") is None, "an English filler leaked into another language"
    assert _pick_filler("English") is not None
    _filler_used.clear()


# --------------------------------------------------------------------------- #
# THE VOICE — a caller cannot see the screen, so silence IS the failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("junk", [b"not a wav at all", b"<html>500 error</html>",
                                  b"RIFF", b"", b"\x00" * 10])
def test_a_provider_returning_non_audio_is_treated_as_a_failure(junk):
    """A provider is free to hand back anything on failure, and one that returned a
    short error string was forwarded to the caller AS IF it were sound — the client
    then failed to decode it and the turn was silent with nothing recorded anywhere.
    Unplayable output must become no output, which the pipeline already handles."""
    from zensuvidha.orchestrator import _playable

    assert _playable(junk) is None


def test_real_audio_still_passes_through():
    from zensuvidha.orchestrator import _playable

    b = io.BytesIO()
    sf.write(b, np.zeros(SR, dtype="float32"), SR, format="WAV", subtype="PCM_16")
    real = b.getvalue()
    assert _playable(real) is real


def test_the_session_never_hands_the_caller_unplayable_bytes():
    class Junk:
        last_skipped_script = False

        def synth(self, text, voice=None):
            return b"upstream returned an error"

    s = Session(load_pack("clinic"), None, tts=Junk())
    s.voice = None
    assert s.tts_bytes("We are open until eight.") is None


def test_a_provider_that_raises_never_reaches_the_turn():
    class Boom:
        last_skipped_script = False

        def synth(self, text, voice=None):
            raise RuntimeError("model file corrupt")

    s = Session(load_pack("clinic"), None, tts=Boom())
    s.voice = None
    assert s.tts_bytes("hello") is None          # handled, not propagated


def test_the_deployment_reports_which_languages_it_cannot_speak():
    """A script no configured voice can pronounce produces a silent reply. That is
    correct behaviour — the UI explains it — but a bad thing to discover from a real
    caller. Measured on macOS: Malayalam, Gujarati, Punjabi, Odia and Urdu are mute."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.lifespan)
    assert "_voice_coverage" in src, "startup no longer probes voice coverage"
    assert "NO VOICE" in src


# --------------------------------------------------------------------------- #
# LOUD CONTINUOUS AUDIO — reported live, speaker at full volume beside the mic
# --------------------------------------------------------------------------- #
def test_a_long_unrecognisable_turn_asks_the_caller_to_repeat():
    """Observed live with music playing at volume: 12s and 15s clips both transcribed
    to '' and the caller got NOTHING back — they repeated themselves into a machine
    that never reacted. Silence is right for a cough; it is wrong for somebody who
    clearly tried to speak."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    # Assert on the whole branch rather than a fixed-width window — a window is brittle
    # against any edit above it, which is how this test broke twice.
    branch = src.split("Nothing recognisable in it")[1].split("except WebSocketDisconnect")[0]
    assert "REPEAT_ASK_S" in branch, "a long empty turn is still dropped in silence"
    assert "safe_say" in branch, "no spoken prompt on the long-clip path"
    assert "_dropped" in branch, "the short-clip path must stay silent"
    assert server.REPEAT_ASK_S > 0


def test_the_repeat_prompt_cannot_become_a_stuck_record():
    """A noisy room produces a run of empty turns. Asking every time would be worse
    than saying nothing — it must fire once and re-arm only after a turn gets through."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    assert 'spec["asked_repeat"] = True' in src, "the prompt is not rate limited"
    assert 'spec["asked_repeat"] = False' in src, "it never re-arms after a good turn"


def test_the_client_cuts_a_latched_vad():
    """Loud continuous audio never gives the detector a silence, so the turn never ends
    and the buffer grows — measured, to 12s and 15s, both of which Whisper returned
    nothing for and which scored the caller 0.07 on their own voiceprint.

    People breathe; playback does not. Recording past LATCH_MS having never seen a pause
    means this is not somebody talking."""
    html = open("web/index.html", encoding="utf-8").read()
    assert "LATCH_MS" in html and "LATCH_GAP_MS" in html
    assert "maxGapMs" in html, "the longest pause is not tracked"
    assert "askToRepeat" in html, "the caller is not told why their turn vanished"

    # Ordering, not a fixed-width window. Slicing N characters after an anchor is how
    # several tests in this suite have broken on unrelated edits — the property is the
    # SEQUENCE of what happens, so assert that.
    at = html.index("maxGapMs<LATCH_GAP_MS")
    closed = html.index("vad='silence'", at)
    offered = html.index("mode:'latched'", at)
    fallback = html.index("askToRepeat", at)

    # The turn is always CLOSED locally — the state machine must not stay in 'speech'.
    assert "uttPCM=[]" in html[at:closed + 200], "the latched turn never closed"
    # …the audio is then OFFERED for salvage. A caller talking OVER continuous noise
    # never gives a clean pause either, so they hit this guard too and used to be
    # discarded and asked to repeat into the same noise. Isolation exists to pull one
    # voice out of exactly that.
    assert closed < offered < fallback, (
        "the latched turn must close, then be offered for isolation, and only then "
        "fall back to asking for a repeat")


def test_a_salvaged_latched_turn_can_never_teach_the_voiceprint():
    """The thing that made the turn latch — continuous sound with no breaths — is
    exactly the thing that would poison the print. Answering it is worth attempting;
    learning identity from it is not.

    Measured before this guard existed: a latched clip scored the real caller 0.07
    against their own voice.
    """
    import inspect
    from zensuvidha.orchestrator import Session
    from zensuvidha import server

    src = inspect.getsource(Session.check_speaker)
    # every path that mutates the print is gated
    assert src.count("may_learn") >= 4, "a learning path is not gated on may_learn"
    for path in ("self.voiceprint = vec", "_widen_voiceprint"):
        assert path in src

    # and the server passes False for a salvaged frame
    ssrc = inspect.getsource(server)
    assert "heard, not salvage)" in ssrc, "salvaged turns are still allowed to enrol"


def test_a_salvage_with_no_voiceprint_is_refused_not_transcribed():
    """With nothing to trim against, isolation cannot run — so passing it on would hand
    Whisper the noise the guard exists for, and enrol it as the caller."""
    import inspect
    from zensuvidha import server
    src = inspect.getsource(server)
    at = src.index("salvage, spec[")
    guard = src.index("session.voiceprint is None", at)
    told = src.index("repair_kind()", guard)
    skipped = src.index("continue", told)
    reaches_pipeline = src.index("session.clean_audio", at)
    assert guard < told < skipped < reaches_pipeline, (
        "a salvage with no voiceprint must be answered and skipped BEFORE the pipeline "
        "— otherwise Whisper gets the noise the latch guard exists for, and it is "
        "enrolled as the caller")


def test_the_repeat_prompt_is_rate_limited_client_side_too():
    html = open("web/index.html", encoding="utf-8").read()
    assert "REPEAT_GAP_MS" in html
    body = html.split("function askToRepeat")[1][:500]
    assert "lastRepeatAt" in body


def test_a_turn_with_no_transcript_still_appears_in_the_inspector():
    """The inspector only ever reported turns that HAD a transcript, so the one case a
    caller most needs explained — "I spoke and nothing happened" — produced no row at
    all and the panel sat blank. Reported live with loud audio playing."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    empty = src.split("Nothing recognisable in it")[1][:1200]
    assert "_insight(" in empty, "an empty turn is still invisible in the inspector"


def test_the_client_labels_an_empty_turn():
    html = open("web/index.html", encoding="utf-8").read()
    assert "nothing recognisable in it" in html, \
        "an empty turn renders as a blank row with no explanation"


def test_a_locally_discarded_turn_is_shown_too():
    """A latched VAD is cut on the client, so the server never sees the clip and cannot
    report it. Without a local row the inspector goes silent at exactly the moment the
    caller is trying to work out what is drowning them out."""
    html = open("web/index.html", encoding="utf-8").read()
    assert "discarded here" in html
    assert "never sent" in html, "no tag distinguishes a local discard"
    # The BAR branch specifically — "if(m.local)" also appears in the tag section above,
    # so splitting on it alone reads the wrong block.
    bar = html.split("const bar=document.createElement")[1].split("row.appendChild(bar)")[0]
    assert "if(m.local){" in bar and "b-drop" in bar, \
        "a discarded clip still draws a full 'kept' bar"


def test_the_analyser_has_headroom_for_loud_audio():
    """Defaults are -100..-30dB. Measured, a full-scale signal pinned only a couple of
    bins — so this is headroom rather than the fix it first looked like — but a loud
    room should not be reading off the top of the scale at all."""
    html = open("web/index.html", encoding="utf-8").read()
    assert "maxDecibels" in html and "minDecibels" in html


# --------------------------------------------------------------------------- #
# THE CALLER WHO SAYS NOTHING
# --------------------------------------------------------------------------- #
def test_every_safe_line_key_is_reachable():
    """safe_line() whitelists the keys it will serve, and an unlisted one degrades to
    "I don't have that detail with me". That is a silent failure: "are you still there?"
    first went out as a refusal because the key had been added to SAFE_LINES but not to
    the whitelist. Any key present in the table must be reachable."""
    from zensuvidha.guard import SAFE_LINES, safe_line

    pack = {"business": {"name": "Test Clinic"}}
    fallback = safe_line("unknown", "English", pack)
    for key in SAFE_LINES["English"]:
        out = safe_line(key, "English", pack, ref="1")
        if key != "unknown":
            assert out != fallback, f"safe_line({key!r}) silently fell back to 'unknown'"


def test_the_idle_lines_exist_in_every_language():
    from zensuvidha.guard import SAFE_LINES

    for lang, table in SAFE_LINES.items():
        assert "still_there" in table, f"{lang} cannot ask if the caller is there"
        assert "goodbye" in table, f"{lang} cannot end the call politely"


def test_a_silent_call_is_prompted_then_closed():
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    assert "_watch_idle" in src, "a caller who says nothing is never re-prompted"
    assert "still_there" in src and "goodbye" in src
    assert "idle_task.cancel()" in src, "the watcher leaks one task per call"
    assert server.IDLE_PROMPT_S < server.IDLE_HANGUP_S, "it would hang up before asking"


def test_max_sessions_is_honest_about_the_cpu():
    """Measured: first audio stays flat at 2.0s under load (streaming works) but the
    TOTAL turn doubles from 4.3s to 9.6s at 6 concurrent, because generation serialises.
    24 would mean 30-40s turns — a number that promises capacity the box has not got."""
    import yaml

    cpu = yaml.safe_load(open("config.yaml"))["server"]["max_sessions"]
    gpu = yaml.safe_load(open("config.gpu.yaml"))["server"]["max_sessions"]
    assert cpu <= 8, f"max_sessions={cpu} is more than this CPU can serve"
    assert gpu > cpu, "the GPU preset should allow more, not the same"


def test_barge_in_does_not_depend_on_browser_echo_cancellation():
    """All AEC today comes from getUserMedia({echoCancellation:true}) — a BROWSER
    feature. A telephony transport (Exotel/Plivo) has no getUserMedia, so the agent will
    hear itself through the caller's speakerphone, Silero will call it speech, and
    BARGE_MS will make it interrupt itself on a loop.

    Barge-in therefore needs a condition that survives without AEC: real speech is
    LOUDER than a room-attenuated echo of our own output."""
    html = open("web/index.html", encoding="utf-8").read()
    assert "ECHO_MARGIN" in html, "barge-in has no self-echo guard"
    assert "agentLevel()" in html, "nothing measures our own output level"
    guard = html.split("SELF-ECHO GUARD")[1].split("if(speaking){ bargeMs+=ms")[0]
    assert "agentLevel()*ECHO_MARGIN" in guard, "the bar is not relative to our own output"
    assert "return;" in guard, "a suspected echo still reaches the barge-in counter"


# --------------------------------------------------------------------------- #
# SNR-aware repair — say WHY you could not hear, not just "say that again"
# --------------------------------------------------------------------------- #
def test_the_repair_line_reflects_how_the_room_actually_sounded():
    """A person tells you why they cannot hear you. Repeating into a fan does not
    help, so "could you move somewhere quieter" is the useful answer — and the number
    needed to say it is already measured on every turn."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), None)

    s.last_snr_db = 18.0
    assert s.repair_kind() == "repeat"        # clean room — nothing to diagnose
    s.last_snr_db = 8.0
    assert s.repair_kind() == "faint"         # poor line, a repeat may work
    s.last_snr_db = 2.0
    assert s.repair_kind() == "noisy"         # repeating will not help; move


def test_an_unmeasured_room_does_not_invent_a_diagnosis():
    """`snr_db` is None when the audio could not be decoded. Telling a caller in a
    quiet office that they are in a noisy one is worse than the generic line."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), None)
    s.last_snr_db = None
    assert s.repair_kind() == "repeat"


def test_every_language_has_the_new_repair_lines():
    """A missing key silently becomes "I don't have that detail" — which is how
    "are you still there?" once went out as a refusal."""
    from zensuvidha.guard import SAFE_LINES, SAFE_LINES_ROMAN, safe_line
    from zensuvidha.packs import load_pack
    pack = load_pack("clinic")
    for lang, table in SAFE_LINES.items():
        for key in ("noisy", "faint"):
            assert key in table, f"{lang} has no {key} line"
            line = safe_line(key, lang, pack)
            assert line and "don't have that detail" not in line
    for key in ("noisy", "faint"):
        assert key in SAFE_LINES_ROMAN["Hindi"]


def test_the_room_is_measured_even_with_no_denoiser_installed():
    """It used to be computed inside the denoise branch, so with the toggle off — the
    default — nothing measured at all and the inspector showed no room reading."""
    import io
    import numpy as np
    import soundfile as sf
    from zensuvidha import pipeline

    sr = 16000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    voice = (0.3 * np.sign(np.sin(2 * np.pi * 130 * t))
             * np.abs(np.sin(2 * np.pi * 3 * t))).astype("float32")
    noisy = voice + 0.25 * np.random.default_rng(0).normal(size=voice.size).astype("float32")

    def wav(x):
        b = io.BytesIO()
        sf.write(b, x, sr, format="WAV", subtype="PCM_16")
        return b.getvalue()

    def dec(raw):
        d, _ = sf.read(io.BytesIO(raw), dtype="float32")
        return d

    _, clean_info = pipeline.prepare(wav(voice), gate=None, voiceprint=None, decode=dec)
    _, noisy_info = pipeline.prepare(wav(noisy), gate=None, voiceprint=None, decode=dec)
    assert clean_info["snr_db"] is not None, "the room was not measured at all"
    assert noisy_info["snr_db"] is not None
    assert noisy_info["snr_db"] < clean_info["snr_db"], "the reading is not responsive"
    assert not clean_info["denoised"] and not noisy_info["denoised"]


# --------------------------------------------------------------------------- #
# who may read caller PII
# --------------------------------------------------------------------------- #
def test_a_relayed_request_is_not_treated_as_local():
    """`client.host == 127.0.0.1` means "is the operator" only while nothing forwards
    to us. Put nginx, Caddy, ngrok or `kubectl port-forward` on the same box and EVERY
    request on earth arrives from 127.0.0.1 — and /transcripts starts serving callers'
    names and phone numbers to the internet.

    A forwarding header is trivially forgeable, which is exactly why it is used this
    way round: its PRESENCE is taken as proof the request was relayed, never as proof
    of who sent it."""
    from zensuvidha import server

    class _Req:
        def __init__(self, host="127.0.0.1", headers=None):
            self.client = type("C", (), {"host": host})()
            self.headers = headers or {}
            self.query_params = {}

    assert server._authorised(_Req()), "a genuinely local operator was locked out"
    for header in ("x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host"):
        req = _Req(headers={header: "203.0.113.9"})
        assert not server._authorised(req), (
            "a request relayed via %s was accepted as local" % header)
    assert not server._authorised(_Req(host="10.0.0.5"))
