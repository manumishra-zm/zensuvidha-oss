"""Fast, dependency-light tests for the engine core (no Ollama/STT/TTS needed).

Run:  pip install pytest && pytest -q
"""
import json

from zensuvidha.orchestrator import (
    Session, build_system_prompt, extract_say, next_chunk, _parse_json,
)
from zensuvidha.packs import list_packs, load_pack


# ---- packs / prompt --------------------------------------------------------
def test_all_packs_load_and_merge():
    packs = list_packs()
    assert {"clinic", "restaurant", "salon", "laundry"} <= set(packs)
    p = load_pack("restaurant")
    assert p["business"]["name"] == "Spice Garden"
    assert p["booking"]["required"] == ["name", "phone", "party_size", "datetime"]
    # inherited from _base.yaml
    assert any(i["name"] == "answer_faq" for i in p["intents"])


def test_system_prompt_contains_contract():
    sp = build_system_prompt(load_pack("clinic"))
    assert "Suvidha Clinic" in sp and "book_appointment" in sp and '"say"' in sp


# ---- streaming helpers -----------------------------------------------------
def test_extract_say_partial_and_complete():
    say, done = extract_say('{"say":"Hello ther')
    assert say == "Hello ther" and done is False
    say, done = extract_say('{"say":"Hello there","action":{}}')
    assert say == "Hello there" and done is True


def test_extract_say_handles_escapes():
    say, done = extract_say('{"say":"Line one.\\nLine two.","action":{}}')
    assert "Line one." in say and "Line two." in say and done is True


def test_extract_say_decodes_unicode_escapes():
    # Ollama may JSON-escape non-ASCII; \uXXXX must decode to the real script
    # (Devanagari here), not leak as literal "u0928".
    say, done = extract_say('{"say":"\\u0928\\u092e\\u0938\\u094d\\u0924\\u0947","action":{}}')
    assert say == "नमस्ते" and done is True
    # a partial buffer that stops mid-escape must wait, not emit garbage
    say, done = extract_say('{"say":"Hi \\u09')
    assert say == "Hi " and done is False


def test_begin_user_dedupes_consecutive_turn():
    # streaming-failure fallback re-runs begin_user with the same text; it must
    # not append the user message twice.
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}))
    s.begin_user("hello there")
    s.begin_user("hello there")
    assert [m["content"] for m in s.messages].count("hello there") == 1


class FakeSTT:
    """Returns preset (text, detected_lang, prob) tuples and records the pinned language."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.language = None            # config default = auto-detect

    def transcribe(self, audio, hint=None, language=None, fast=None):
        self.calls.append(language)
        item = self.outputs.pop(0) if self.outputs else ("", None, 0.0)
        return item if isinstance(item, tuple) else (item, None, 0.0)


def test_auto_language_latches_reply_but_stt_stays_auto():
    # A CONFIDENT Whisper detection latches the REPLY language, but STT stays auto-detect
    # (NOT pinned to the lock) so a wrong lock can later self-correct (drift-unlock).
    stt = FakeSTT([("नमस्ते मुझे अपॉइंटमेंट चाहिए", "hi", 0.95), ("ok thanks", "hi", 0.9)])
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}), stt=stt)
    s.lang = None                       # explicit auto-detect
    s.transcribe(b"a")
    assert s.lang_lock == "hi"          # reply language latched Hindi
    assert stt.calls[0] is None         # STT auto-detected
    s.transcribe(b"b")
    assert stt.calls[1] is None         # STT still auto (not pinned) → drift is detectable
    assert s.reply_language("ok thanks") == "Hindi"   # reply stays stable in the locked language


def test_language_drift_unlocks_on_confident_detection():
    # A wrong/old lock self-corrects: one CONFIDENT turn in a new language re-locks.
    stt = FakeSTT([("मुझे चाहिए", "hi", 0.9),      # latch hi
                   ("నాకు కావాలి", "te", 0.9)])     # confident Telugu → re-lock
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}), stt=stt)
    s.lang = None
    s.transcribe(b"1"); assert s.lang_lock == "hi"
    s.transcribe(b"2"); assert s.lang_lock == "te"


def test_language_drift_needs_two_weak_turns():
    # A single LOW-confidence disagreement must NOT unlock; it takes two agreeing turns.
    stt = FakeSTT([("मुझे चाहिए", "hi", 0.9),
                   ("నాకు", "te", 0.5), ("కావాలి", "te", 0.5)])
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}), stt=stt)
    s.lang = None
    s.transcribe(b"1"); assert s.lang_lock == "hi"
    s.transcribe(b"2"); assert s.lang_lock == "hi"   # one weak disagreement: keep the lock
    s.transcribe(b"3"); assert s.lang_lock == "te"   # corroborated → re-locked


def test_low_confidence_needs_two_turns_to_latch():
    # A single low-confidence / ambiguous clip must NOT permanently lock the language;
    # it takes two agreeing turns (corroboration) to latch.
    stt = FakeSTT([("नमस्ते", "hi", 0.4), ("मुझे बुकिंग चाहिए", "hi", 0.45)])
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}), stt=stt)
    s.lang = None
    s.transcribe(b"a")
    assert s.lang_lock is None          # not latched on one weak clip
    s.transcribe(b"b")
    assert s.lang_lock == "hi"          # two agreeing turns → latched

def test_clean_for_speech_strips_markdown():
    from zensuvidha.orchestrator import clean_for_speech
    assert clean_for_speech("**Bold** and `code` and # head") == "Bold and code and  head".replace("  ", " ")
    assert "*" not in clean_for_speech("- item one\n- item two")
    assert clean_for_speech("see [our site](https://x.com)") == "see our site"
    # currency + native script preserved (voice reads ₹ correctly)
    assert "₹500" in clean_for_speech("General Physician ₹500")
    assert clean_for_speech("नमस्ते, समय 9 बजे") == "नमस्ते, समय 9 बजे"


def test_stt_hallucination_guard():
    from zensuvidha.stt import _is_hallucination, _looks_degenerate
    # classic Whisper silence/noise artifacts → dropped
    assert _is_hallucination("Thank you for watching!")
    assert _is_hallucination("Please subscribe")
    assert _is_hallucination(".")
    assert _is_hallucination("a")
    assert _looks_degenerate("5 5 5 5 5 5 5")
    # real caller utterances → NOT dropped
    assert not _is_hallucination("what are your timings")
    assert not _is_hallucination("मुझे अपॉइंटमेंट चाहिए")
    assert not _is_hallucination("yes please book it")


def test_marathi_not_mislatched_as_hindi():
    # Whisper detecting Marathi confidently must win over the Devanagari-script guess (Hindi).
    stt = FakeSTT([("मला अपॉइंटमेंट हवी आहे", "mr", 0.9)])
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}), stt=stt)
    s.lang = None
    s.transcribe(b"a")
    assert s.lang_lock == "mr"
    assert s.reply_language("") == "Marathi"


def test_pinned_language_ignores_stray_script():
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}), stt=FakeSTT([]))
    s.lang = "en"                       # caller pinned English via the dropdown
    assert s.reply_language("ok नमस्ते thanks") == "English"   # a stray char must not flip it


def test_next_chunk_emits_complete_sentences():
    chunk, idx = next_chunk("Hi there. How can I help", 0)
    assert chunk == "Hi there." and idx == 9
    chunk2, _ = next_chunk("Hi there. How can I help", idx)
    assert chunk2 == ""  # no further complete sentence yet


def test_parse_json_robust():
    assert _parse_json('{"say":"ok","action":{"type":"none"}}')["say"] == "ok"
    assert _parse_json('noise {"say":"hi","action":{"type":"none"}} tail')["say"] == "hi"
    assert "say" in _parse_json("totally not json")


# ---- full turn with a fake LLM --------------------------------------------
class FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    def chat(self, messages, force_json=True, model=None, num_predict=None, meta=None,
             num_ctx=None):
        if meta is not None:
            meta["finish_reason"] = "stop"
        return json.dumps(self._payload)


def test_booking_side_effect(tmp_path, monkeypatch):
    # isolate the DB
    import zensuvidha.booking as b
    monkeypatch.setattr(b, "DB", tmp_path / "t.db")

    payload = {"say": "All set.", "action": {"type": "book_appointment",
               "slots": {"name": "Anil", "phone": "9876543210",
                         "doctor": "Dr Sharma", "datetime": "tomorrow 5pm"}}}
    s = Session(load_pack("clinic"), FakeLLM(payload))
    res = s.handle_text("book with Dr Sharma tomorrow 5pm, Anil, 9876543210")
    assert res["action"]["type"] == "book_appointment"
    assert res["action"].get("booking_id") == 1
    assert "#1" in res["say"]
    assert b.list_bookings("clinic")[0]["name"] == "Anil"


def test_booking_accumulates_slots_across_turns(tmp_path, monkeypatch):
    # Reproduces the real bug: the model forgets earlier answers and returns empty replies.
    # The server must remember slots across turns and never say "booked" until all are filled.
    import zensuvidha.booking as b
    monkeypatch.setattr(b, "DB", tmp_path / "acc.db")
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}))
    s.begin_user("I want to book an appointment")   # slots are only collected once asked
    # turn 1 — name + doctor
    r1 = s.finalize(json.dumps({"say": "Sure, what time works?",
                                "action": {"type": "collect", "slots": {"name": "Anil", "doctor": "Dr Sharma"}}}))
    assert r1["action"]["type"] == "collect" and set(r1["action"]["missing"]) == {"phone", "datetime"}
    # turn 2 — model FORGETS earlier slots and gives an EMPTY say; server remembers + asks next
    r2 = s.finalize(json.dumps({"say": "", "action": {"type": "collect", "slots": {"datetime": "Monday 4pm"}}}))
    assert r2["say"].strip() and r2["say"] != "Sorry, could you repeat that?"
    assert r2["action"]["missing"] == ["phone"]
    assert s.slots["name"] == "Anil" and s.slots["datetime"] == "Monday 4pm"
    # turn 3 — phone given → books with ALL accumulated slots
    r3 = s.finalize(json.dumps({"say": "Booking that now", "action": {"type": "book_appointment",
                                "slots": {"phone": "9876543210"}}}))
    assert r3["action"].get("booking_id") == 1 and "#1" in r3["say"]
    assert b.list_bookings("clinic")[0]["name"] == "Anil"


def test_premature_book_downgrades_not_false_confirm(tmp_path, monkeypatch):
    import zensuvidha.booking as b
    monkeypatch.setattr(b, "DB", tmp_path / "pre.db")
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}))
    # model wrongly claims "booked" with only a name → must NOT confirm, must ask for missing
    r = s.finalize(json.dumps({"say": "Your appointment is booked!",
                               "action": {"type": "book_appointment", "slots": {"name": "Anil"}}}))
    assert r["action"]["type"] == "collect"
    assert "book" not in r["say"].lower()      # no false confirmation
    assert not b.list_bookings("clinic")       # nothing actually written


def test_incomplete_booking_becomes_collect():
    payload = {"say": "What's your phone number?",
               "action": {"type": "book_appointment", "slots": {"name": "Anil"}}}
    s = Session(load_pack("clinic"), FakeLLM(payload))
    res = s.handle_text("book me in, I'm Anil")
    assert res["action"]["type"] == "collect"
    assert "phone" in res["action"]["missing"]


def test_deterministic_safety_escalation():
    s = Session(load_pack("clinic"), FakeLLM({"say": "x", "action": {"type": "none"}}))
    res = s.handle_text("I have chest pain")
    assert res["escalated"] is True
    assert "urgent" in res["say"].lower() or "connect" in res["say"].lower()


# ---- simple-turn model routing --------------------------------------------
def test_is_simple_turn():
    from zensuvidha.server import _is_simple_turn
    assert _is_simple_turn("hi")
    assert _is_simple_turn("thank you")
    assert _is_simple_turn("yes please")
    assert not _is_simple_turn("I'd like to book Dr Sharma tomorrow at 5pm")  # digits + long
    assert not _is_simple_turn("my number is 9876543210")                     # digits
    assert not _is_simple_turn("what are your consultation charges today")     # long


# ---- binary audio frame round-trips ---------------------------------------
def test_audio_frame_roundtrip():
    import json as _json
    from zensuvidha.server import _audio_frame
    wav = b"RIFF\x00\x00fake-wav-bytes"
    frame = _audio_frame({"type": "chunk", "seq": 2, "text": "नमस्ते"}, wav)
    hlen = int.from_bytes(frame[:4], "big")
    meta = _json.loads(frame[4:4 + hlen].decode("utf-8"))
    assert meta == {"type": "chunk", "seq": 2, "text": "नमस्ते"}
    assert frame[4 + hlen:] == wav
    # no-audio frame carries an empty tail
    empty = _audio_frame({"type": "chunk", "seq": 1, "text": "hi"}, None)
    hl2 = int.from_bytes(empty[:4], "big")
    assert empty[4 + hl2:] == b""


def test_stt_rejects_character_level_degeneration():
    """Real observed Whisper output: it latched onto the Devanagari nukta and emitted it
    ~120 times. No spaces, so it's ONE token — the word-level check never saw it, and the
    garbage reached the LLM."""
    from zensuvidha.stt import _looks_degenerate
    assert _looks_degenerate("तो अचाएज़" + "़" * 120)
    assert _looks_degenerate("aaaaaaaaaaaa")
    assert _looks_degenerate("5 5 5 5 5 5 5")
    # real utterances must still get through
    for real in ["मुझे अपॉइंटमेंट चाहिए",
                 "what are your timings",
                 "మీ క్లినిక్ ఎన్ని గంటలకు తెరుస్తారు",
                 "haan theek hai, kal shaam paanch baje"]:
        assert not _looks_degenerate(real), real
