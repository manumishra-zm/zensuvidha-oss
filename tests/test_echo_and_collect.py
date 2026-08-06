"""Two failures taken straight from call #11 in data/zensuvidha.db.

1. THE ECHO. The model answered AS the caller instead of TO them:
       caller "मेरा नाम मनु मिश्रा है"  → agent "मेरा नाम मिश्रा है"
       caller "8920429057"             → agent "मेरा मोबाइल नंबर 8920429057 है"
   It reads like a reply, so nothing caught it — and it corrupted the booking, which
   stored the name as "मिश्रा", losing "मनु".

2. THE COLLECT-PHASE REFUSAL. Asked "which day and time?", the caller answered
   "दस भजे सुभा" and got "माफ़ कीजिए, यह जानकारी मेरे पास नहीं है" — the agent abandoned
   its own booking to go check with the front desk.

Run:  pytest -q tests/test_echo_and_collect.py
"""
import json

import pytest

from zensuvidha.guard import ask_line, looks_like_echo
from zensuvidha.orchestrator import Session
from zensuvidha.packs import load_pack


class FakeLLM:
    def __init__(self, payload):
        self._payload = payload
        self.num_predict = 200
        self.num_ctx = 6144
        self.calls = 0

    def chat(self, messages, force_json=True, model=None, num_predict=None, meta=None,
             num_ctx=None):
        self.calls += 1
        if meta is not None:
            meta["finish_reason"] = "stop"
        p = self._payload(self.calls) if callable(self._payload) else self._payload
        return json.dumps(p, ensure_ascii=False)


def _s(payload=None):
    return Session(load_pack("clinic"), FakeLLM(payload or {"kind": "answer", "say": "ok",
                                                            "action": {"type": "none"}}))


def _heard(s, text):
    """Put the caller's words in history the way a real turn would."""
    s.messages.append({"role": "user", "content": text})
    return s


# --------------------------------------------------------------------------- #
# 1. the echo detector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply,caller,values", [
    # verbatim from the transcript
    ("मेरा नाम मिश्रा है", "मेरा ना मनु मिश्रा है", ()),
    ("मेरा मोबाइल नंबर 8920429057 है", "8920429057", ("8920429057",)),
    ("मैंने अपना मोबाइल नंबर आपको दे दिया है", "मैंने अपना मोभाई लंबर आपको दे दिया है", ()),
    # Whisper spells the same word differently between takes — must still match
    ("मेरा मोबाइल नंबर क्या है?", "मेरा मोबाल नंबर", ()),
    # ROMANISED Hindi — a Hinglish caller gets Hinglish back, so the pronouns are Latin
    ("Mujhe doctor se appointment chahiye? Kya aapko specific doctor chahiye?",
     "mujhe doctor se appointment chahiye", ()),
    # echo + a legitimate tail: the appended question dilutes whole-reply overlap, but
    # leading with the caller's line in their own voice is still the failure
    ("Mera naam Manu hai. Kya aapko phone number bhi chahiye?", "mera naam Manu hai", ()),
    # WRONG POSSESSIVE — "what is MY mobile number?". Repeats nothing, so word overlap
    # cannot see it, but a receptionist asking for a detail must address the caller.
    ("मेरा मोबाइल नंबर क्या है?", "मेरा नाम मनु मिश्रा है", ()),
    ("मेरा नाम क्या है?", "मुझे अपॉइंटमेंट चाहिए", ()),
    ("Mera mobile number kya hai?", "mera naam Manu hai", ()),
    ("నా మొబైల్ నంబర్ ఏమిటి?", "నాకు అపాయింట్‌మెంట్ కావాలి", ()),
])
def test_first_person_echo_is_detected(reply, caller, values):
    assert looks_like_echo(reply, caller, values)


@pytest.mark.parametrize("reply,caller", [
    # "I" is not "my" — a question the receptionist asks ABOUT HERSELF is correct
    ("क्या मैं अपॉइंटमेंट बुक कर दूँ?", "हाँ ठीक है"),
    ("मैं सनराइज़ क्लिनिक की रिसेप्शनिस्ट हूँ।", "आप कौन हैं?"),
    ("मेरा नाम तारा है।", "आपका नाम क्या है?"),        # possessive, but not a question
    ("आपका मोबाइल नंबर क्या है?", "मेरा नाम मनु है"),   # the correct form of the same question
])
def test_the_receptionists_own_first_person_is_allowed(reply, caller):
    assert not looks_like_echo(reply, caller, ())


def test_our_own_pre_written_lines_never_trip_the_detector():
    """A recovery line that the guard then rejected would loop forever."""
    for lang in ("Hindi", "Telugu", "English"):
        for field in ("name", "phone", "datetime", "service", "confirm"):
            line = ask_line(field, lang)
            assert not looks_like_echo(line, "मेरा नाम मनु है", ("8920429057",)), (lang, field, line)
        assert not looks_like_echo(ask_line("name", lang, romanized=True), "mera naam Manu hai", ())


@pytest.mark.parametrize("reply,caller,values", [
    # a receptionist repeating the caller's details back is CORRECT — it addresses them
    ("आपका नाम मनु मिश्रा नोट कर लिया है।", "मेरा नाम मनु मिश्रा है", ()),
    ("आपका मोबाइल नंबर 8920429057 सही है?", "8920429057", ("8920429057",)),
    ("हां, आपका अपॉइंटमेंट शनिवार को दस बजे डॉक्टर अनिल शर्मा के साथ बुक कर लिया गया है।",
     "अनिल शर्मा के साथ शनिवार दस बजे अपीजे", ()),
    # first person about HERSELF is the persona speaking, not an echo
    ("मैं सनराइज़ क्लिनिक की रिसेप्शनिस्ट हूँ।", "आप कौन हैं?", ()),
    ("माफ़ कीजिए, यह जानकारी मुझे नहीं है।", "आपकी फीस कितनी है?", ()),
    # an acknowledgement that reuses the caller's words but no first-person pronoun
    ("మను మిశ్రా అనే పేరు తెలుసు", "నా పేరు మను మిశ్రా", ()),
    ("हमारे पास जनरल फिजिशियन और त्वचा विशेषज्ञ हैं।", "यहाँ कौन कौन से डॉक्टर हैं", ()),
    # romanised replies that address the caller are correct Hinglish, not echoes
    ("Aapka naam Manu note kar liya hai. Aapka mobile number kya hai?", "mera naam Manu hai", ()),
    ("Main Suvidha Clinic ki receptionist hoon.", "aap kaun hain", ()),
    ("Aapka appointment kal subah 11 baje book ho gaya hai.", "kal subah 11 baje", ()),
    ("Haan ji. Aapka naam kya hai?", "haan", ()),
])
def test_legitimate_replies_are_not_echoes(reply, caller, values):
    assert not looks_like_echo(reply, caller, values)


# --------------------------------------------------------------------------- #
# 3c. the caller's spoken phone number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("said,expect", [
    ("9876512345", "9876512345"),
    ("मेरा नंबर 8920429057 है", "8920429057"),
    ("నా నంబర్ 9876512345", "9876512345"),
])
def test_a_spoken_phone_number_is_captured_without_the_model(said, expect):
    """Only the model used to file the number, so a turn whose reply the guard replaced
    lost it — and we asked for it again immediately after they had given it."""
    assert _s().slots_from_caller(said).get("phone") == expect


@pytest.mark.parametrize("said", [
    "कल सुबह 10 बजे",              # a time, far too short to be a number
    "9876512345 और 8920429057",    # ambiguous — two candidates, take neither
    "फीस ₹500 है",
    "",
])
def test_non_phone_numbers_are_not_captured(said):
    assert "phone" not in _s().slots_from_caller(said)


def test_short_replies_are_never_echoes():
    """"हाँ" / "जी" must survive — an echo needs a sentence."""
    assert not looks_like_echo("हाँ", "हाँ", ())
    assert not looks_like_echo("जी हाँ", "जी हाँ", ())


def test_verify_replaces_an_echo():
    s = _heard(_s(), "मेरा नाम मनु मिश्रा है")
    out = s.verify("मेरा नाम मिश्रा है", "answer")
    assert out != "मेरा नाम मिश्रा है"
    assert out, "an echo must be replaced, not blanked"


def test_a_booked_turn_may_repeat_the_callers_own_words():
    """The confirmation quotes the caller's details by design — booking_turn exempts it."""
    s = _heard(_s(), "मेरा नाम मनु मिश्रा है")
    say = "मेरा नाम मिश्रा है"
    assert s.verify(say, "answer", booking_turn=True) == say


# --------------------------------------------------------------------------- #
# 2. the collect-phase refusal
# --------------------------------------------------------------------------- #
def _mid_booking(lang_text="दस भजे सुभा", **slots):
    s = _s()
    s.booking_started = True
    s.slots = slots or {"name": "मनु मिश्रा", "phone": "8920429057",
                        "doctor": "General Physician — Dr Anil Sharma"}
    return _heard(s, lang_text)


def test_mid_booking_asks_for_what_is_missing_instead_of_refusing():
    s = _mid_booking()
    refusal = s.safe_say("unknown")
    out = s.verify("", "unknown")
    assert out != refusal, "must not abandon the booking to 'check with the desk'"
    assert out == s.slot_question("datetime")
    assert s.pending_slot == "datetime"


def test_the_refusal_is_kept_when_no_booking_is_under_way():
    """Outside a booking, "I don't have that" is the honest answer and must survive."""
    s = _heard(_s(), "पुणे यहाँ से कितनी दूर है?")
    assert s.verify("", "unknown") == s.safe_say("unknown")


def test_a_rejected_reply_confirms_once_every_slot_is_known():
    """The last slot had just landed when the model's reply was rejected, and the caller
    heard "sorry, I didn't catch that" — at the exact moment everything was in hand."""
    s = _mid_booking(name="मनु", phone="8920429057", doctor="Dr Anil Sharma",
                     datetime="शनिवार दस बजे")
    assert not s.collecting()
    out = s.verify("", "unknown")
    assert out == s.confirm_question()
    assert out not in (s.safe_say("unknown"), s.safe_say("repeat"))


def test_a_finished_booking_stops_asking_for_slots():
    """booking_started stayed true after a booking, so every later rejected reply came
    back as "may I have your name?" long after the appointment was made."""
    s = _s({"kind": "answer", "say": "हो गया", "action": {"type": "book_appointment"}})
    s.booking_started = True
    s.slots = {"name": "मनु", "phone": "8920429057", "doctor": "Dr Anil Sharma",
               "datetime": "शनिवार दस बजे"}
    s.begin_user("हाँ बुक कर दीजिए")
    res = s.finalize(json.dumps({"kind": "answer", "say": "आपका अपॉइंटमेंट बुक हो गया है।",
                                 "action": {"type": "book_appointment"}}, ensure_ascii=False))
    assert res["action"].get("booking_id"), "the booking should have gone through"
    assert not s.booking_started and not s.slots
    # a later unrelated failure is a plain refusal again, not a slot question
    s.begin_user("पुणे यहाँ से कितनी दूर है?")
    assert s.verify("", "unknown") == s.safe_say("unknown")


def test_slot_questions_come_out_in_the_callers_language():
    hi = _heard(_s(), "मुझे अपॉइंटमेंट चाहिए").slot_question("datetime")
    te = _heard(_s(), "నాకు అపాయింట్‌మెంట్ కావాలి").slot_question("datetime")
    en = _heard(_s(), "I need an appointment").slot_question("datetime")
    assert hi == ask_line("datetime", "Hindi")
    assert te == ask_line("datetime", "Telugu")
    # an English caller keeps the pack's own, more specific wording
    assert en == load_pack("clinic")["booking"]["slots"]["datetime"]
    assert len({hi, te, en}) == 3


def test_a_pack_may_override_a_slot_question_per_language():
    s = _s()
    s.pack["booking"]["slots"]["datetime_hi"] = "कौन सा दिन ठीक रहेगा?"
    try:
        assert _heard(s, "मुझे अपॉइंटमेंट चाहिए").slot_question("datetime") == "कौन सा दिन ठीक रहेगा?"
    finally:
        del s.pack["booking"]["slots"]["datetime_hi"]


# --------------------------------------------------------------------------- #
# 3. taking the caller's answer instead of re-asking
# --------------------------------------------------------------------------- #
def test_the_callers_own_wording_fills_the_pending_slot():
    """"दस भजे सुभा" answers "which day and time?" — filing it is what breaks the loop."""
    s = _mid_booking()
    s.pending_slot = "datetime"
    s._answer_to_pending("दस भजे सुभा")
    assert s.slots["datetime"] == "दस भजे सुभा"
    assert not s.collecting()


def test_a_question_is_not_an_answer():
    s = _mid_booking()
    s.pending_slot = "datetime"
    for q in ("क्या आप मुझे सुन सकते हैं?", "आपकी फीस कितनी है", "what time do you close",
              "is Monday okay", "can you hear me"):
        s._answer_to_pending(q)
    assert "datetime" not in s.slots


@pytest.mark.parametrize("said,stored", [
    # "is" mid-sentence is a statement, not a question — and only the name is stored
    ("my name is Manu Mishra", "Manu Mishra"),
    ("it is Manu Mishra", "Manu Mishra"),
    ("Manu Mishra", "Manu Mishra"),
    ("मेरा नाम मनु मिश्रा है", "मनु मिश्रा"),
    ("నా పేరు మను మిశ్రా", "మను మిశ్రా"),
    ("mera naam Manu hai", "Manu"),
])
def test_a_statement_containing_an_auxiliary_is_still_an_answer(said, stored):
    """Matching "is"/"are"/"do" anywhere rejected ordinary statements and lost the name;
    English only inverts for questions, so an auxiliary counts when it LEADS. And the
    booking record must hold the NAME, not the whole sentence around it."""
    s = _mid_booking()
    s.slots.pop("name")
    s.pending_slot = "name"
    s._answer_to_pending(said)
    assert s.slots.get("name") == stored


def test_an_unrecognised_name_sentence_is_kept_verbatim():
    """Stripping too eagerly would lose part of a real name."""
    s = _mid_booking()
    s.slots.pop("name")
    s.pending_slot = "name"
    s._answer_to_pending("Manu Kumar Mishra")
    assert s.slots["name"] == "Manu Kumar Mishra"


def test_a_reply_re_asking_for_a_known_slot_is_replaced():
    """The pack's questions are in the system prompt, so the model reproduces them
    verbatim — including for details the caller has already given."""
    s = _mid_booking()                      # name, phone, doctor known; datetime missing
    pack_q = load_pack("clinic")["booking"]["slots"]["phone"]
    assert s.asks_for_known_slot(pack_q) == "phone"
    assert s.asks_for_known_slot(s.slot_question("datetime")) is None, "the open slot is fine"
    assert s.asks_for_known_slot("Sure, let me check that for you.") is None


def test_a_long_reply_is_not_taken_as_a_slot_value():
    s = _mid_booking()
    s.pending_slot = "datetime"
    s._answer_to_pending("मुझे लगता है कि शायद अगले हफ्ते कभी आ पाऊँगा लेकिन अभी पक्का नहीं कह सकता हूँ जी")
    assert "datetime" not in s.slots


def test_a_phone_number_is_never_captured_this_way():
    """A misheard number is worse than no number — phone keeps its own validation."""
    s = _mid_booking()
    s.slots.pop("phone")
    s.pending_slot = "phone"
    s._answer_to_pending("आठ नौ दो")
    assert not s.slots.get("phone")


def test_a_phone_number_is_never_filed_as_the_name():
    """Asked for a name and told "9876512345", this filed the number AS the name — it is
    short, not a question, and passed the plausibility check."""
    s = _mid_booking()
    s.slots.pop("name")
    s.pending_slot = "name"
    s._answer_to_pending("9876512345")
    assert not s.slots.get("name")
    assert not s._plausible_slot("name", "9876512345")


def test_hinglish_gets_hinglish_questions_not_english():
    """A Hinglish caller is speaking Hindi in Latin letters — the pack's English
    question is the wrong language for them too."""
    s = _heard(_s(), "mujhe doctor se appointment chahiye")
    q = s.slot_question("name")
    assert q != load_pack("clinic")["booking"]["slots"]["name"]
    assert q == ask_line("name", "Hindi", romanized=True)
    assert "aapka" in q.lower()


def test_an_already_known_slot_is_not_overwritten():
    s = _mid_booking()
    s.pending_slot = "name"
    s._answer_to_pending("कल सुबह")
    assert s.slots["name"] == "मनु मिश्रा"


# --------------------------------------------------------------------------- #
# 3b. a turn with no script must not switch the reply language
# --------------------------------------------------------------------------- #
def test_a_digits_only_turn_keeps_the_conversations_language():
    """Reading out a phone number used to flip a Hindi call to English replies —
    "8920429057" has no script, so detection fell through to the default."""
    s = _s()
    s.begin_user("मुझे डॉक्टर से अपॉइंटमेंट चाहिए")
    assert s.reply_language(s.last_user_text()) == "Hindi"
    s.begin_user("8920429057")
    assert s.reply_language(s.last_user_text()) == "Hindi", "language must persist"
    assert s.slot_question("datetime") == ask_line("datetime", "Hindi")


def test_a_scriptless_turn_keeps_telugu_too():
    s = _s()
    s.begin_user("నాకు అపాయింట్‌మెంట్ కావాలి")
    s.begin_user("9876512345")
    assert s.reply_language(s.last_user_text()) == "Telugu"


def test_an_english_call_is_unaffected():
    s = _s()
    s.begin_user("I need an appointment")
    s.begin_user("9876512345")
    assert s.reply_language(s.last_user_text()) in (None, "English")
    assert s.slot_question("name") == load_pack("clinic")["booking"]["slots"]["name"]


# --------------------------------------------------------------------------- #
# 4. the real call, replayed
# --------------------------------------------------------------------------- #
def test_call_11_no_longer_echoes_or_abandons_the_booking():
    """The three turns that went wrong, run back through the guard."""
    turns = [
        # (caller said, what the model produced, why it was wrong)
        ("मेरा नाम मनु मिश्रा है",
         {"kind": "answer", "say": "मेरा नाम मिश्रा है", "action": {"type": "collect"}}),
        ("8920429057",
         {"kind": "answer", "say": "मेरा मोबाइल नंबर 8920429057 है", "action": {"type": "collect"}}),
        ("दस भजे सुभा",
         {"kind": "unknown", "say": "", "action": {"type": "none"}}),
    ]
    s = _s()
    s.booking_started = True
    s.slots = {"doctor": "General Physician — Dr Anil Sharma"}
    spoken = []
    for caller, payload in turns:
        s.begin_user(caller)          # the real entry point: registers caller numbers, latches language
        say = s.verify(payload["say"], payload["kind"])
        spoken.append(say)
        s.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})

    assert spoken[0] != "मेरा नाम मिश्रा है", "still echoing the caller's name"
    assert spoken[1] != "मेरा मोबाइल नंबर 8920429057 है", "still claiming the caller's number"
    assert spoken[2] != s.safe_say("unknown"), "still abandoning the booking"
    assert all(spoken), "no turn may fall silent"
    # every replacement is a real question that moves the booking forward
    questions = {s.slot_question(f) for f in ("name", "phone", "doctor", "datetime")}
    assert all(line in questions for line in spoken), spoken


def test_a_grounding_rejection_on_the_LAST_slot_confirms_not_refuses():
    """The caller's answer is filed BEFORE verify runs, so the turn that completes the
    booking has nothing missing. Gating the recovery on collecting() meant the caller
    heard "I'll check with the desk" at the exact moment every detail was in hand."""
    s = _s()
    s.booking_started = True
    s.slots = {"name": "मनु मिश्रा", "phone": "8920429057",
               "doctor": "Dr Anil Sharma", "datetime": "दस बजे सुबह"}
    s.begin_user("दस बजे सुबह")
    assert not s.collecting(), "nothing left to collect — that is the trap"
    # a reply carrying an ungrounded number gets thrown out by check_reply
    # NOTE the number: 4 used to be ungrounded and is not any more, because the pack
    # now says the station is "about 4km". Every fact added to a pack WIDENS the set
    # of numbers the guard accepts — real knowledge and tight grounding pull against
    # each other — so this pins a figure that appears nowhere in it.
    out = s.verify("अब क्या दिन चाहिए? जैसे 'शनिवार शाम 4444 बजे'।", "answer")
    assert out == s.confirm_question()
    assert out != s.safe_say("unknown"), "must not abandon a completed booking"


# --------------------------------------------------------------------------- #
# language: a caller who switches language mid-call
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lock,text,want", [
    ("en", "आपकी फीस कितनी है?", "Hindi"),          # English call switches to Hindi
    ("en", "మీ ఫీజు ఎంత?", "Telugu"),               # English -> Telugu
    ("te", "आपकी फीस कितनी है?", "Hindi"),          # Telugu -> Hindi
])
def test_a_script_change_overrides_the_language_lock(lock, text, want):
    """The latch stops per-clip flip-flopping, but a transcript in a DIFFERENT SCRIPT is
    far stronger evidence. An English call that switched to Hindi kept getting English."""
    s = _s()
    s.lang_lock = lock
    s.begin_user(text)
    assert s.reply_language(s.last_user_text()) == want


@pytest.mark.parametrize("lock,text,want", [
    ("mr", "मला अपॉइंटमेंट हवी", "Marathi"),         # Marathi and Hindi SHARE Devanagari
    ("hi", "मुझे अपॉइंटमेंट चाहिए", "Hindi"),
    ("hi", "mujhe appointment chahiye", "Hindi"),   # Latin = Hinglish, not a switch
    ("te", "నాకు అపాయింట్‌మెంట్ కావాలి", "Telugu"),
])
def test_the_lock_survives_when_the_script_has_not_changed(lock, text, want):
    """Same script means the lock is the MORE precise answer — Devanagari alone cannot
    tell Marathi from Hindi, and Latin text is Hinglish rather than English."""
    s = _s()
    s.lang_lock = lock
    s.begin_user(text)
    assert s.reply_language(s.last_user_text()) == want


# --------------------------------------------------------------------------- #
# booking intent must not fire on ordinary questions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "what is your consultation fee?",       # "consult" substring-matched "consultation"
    "how much does a consultation cost",
    "do you do dialysis?",
    "what are your timings",
    "आपकी फीस कितनी है?",
    "मुझे डायलिसिस के बारे में जानना है",
])
def test_an_ordinary_question_does_not_start_a_booking(q):
    """Substring matching put a caller who asked a PRICE into slot collection, so every
    later rejected reply came back as "may I have the patient's full name?"."""
    assert not _s()._wants_to_book(q)


@pytest.mark.parametrize("q", [
    "I want an appointment",
    "book an appointment please",
    "can I schedule a visit",
    "I'd like to reserve a slot with Dr Sharma",
    "मुझे अपॉइंटमेंट चाहिए",
    "నాకు అపాయింట్‌మెంట్ కావాలి",
])
def test_a_real_booking_request_is_still_recognised(q):
    assert _s()._wants_to_book(q)


def test_a_fee_question_does_not_leave_the_session_in_collection():
    """End to end: the state must not flip, or the NEXT rejected reply is a slot question."""
    s = _s()
    s.begin_user("what is your consultation fee?")
    s.finalize(json.dumps({"kind": "answer", "say": "General Physician is Rs 500.",
                           "action": {"type": "none"}}))
    assert not s.booking_started
    assert s.recovery_line("unknown") == s.safe_say("unknown")


def test_an_unrelated_sentence_is_never_stored_as_the_appointment_time():
    """REPRODUCED before the fix: the model returned an invented ISO date, the fallback
    substituted whatever the caller last said, and "मेरा नाम मनु मिश्रा है" ("my name is
    Manu Mishra") was written into the appointment time and on into SQLite."""
    s = _s()
    s.booking_started = True
    s.slots = {"phone": "8920429057", "doctor": "General Physician — Dr Anil Sharma"}
    s.pending_slot = "name"                      # we asked for the NAME, not the time
    s.begin_user("मेरा नाम मनु मिश्रा है")
    s.finalize(json.dumps({"kind": "answer", "say": "ठीक है",
                           "action": {"type": "collect",
                                      "slots": {"name": "मनु मिश्रा",
                                                "datetime": "2025-08-05 10:00"}}},
                          ensure_ascii=False))
    assert s.slots.get("name") == "मनु मिश्रा"
    assert not s.slots.get("datetime"), \
        f"an unrelated sentence became the appointment time: {s.slots.get('datetime')!r}"
    assert "datetime" in s.missing_slots(), "the time must still be asked for"


def test_the_callers_own_wording_is_still_kept_when_we_asked_for_the_time():
    """The original intent must survive: the model turns "कल सुबह 11 बजे" into an invented
    ISO date, and the caller\'s own phrasing is the truth."""
    s = _s()
    s.booking_started = True
    s.slots = {"name": "मनु", "phone": "8920429057", "doctor": "Dr Anil Sharma"}
    s.pending_slot = "datetime"                  # we DID ask for the time
    s.begin_user("कल सुबह ग्यारह बजे")
    s.finalize(json.dumps({"kind": "answer", "say": "ठीक है",
                           "action": {"type": "collect",
                                      "slots": {"datetime": "2024-05-29 11:00"}}},
                          ensure_ascii=False))
    assert s.slots.get("datetime") == "कल सुबह ग्यारह बजे"


# --------------------------------------------------------------------------- #
# A booking in progress must not swallow the caller's own questions
# --------------------------------------------------------------------------- #
def test_a_question_asked_mid_booking_is_answered_not_replaced_by_a_slot():
    """Observed on the salon pack: mid-booking, "do you use ammonia free colour?" was
    answered with "May I have your name?". The model had correctly said it did not
    know; the guard then substituted a slot question and the caller's question simply
    vanished. Refusing honestly IS the answer — the booking is kept alive by appending
    the slot question, not by replacing the reply with it."""
    s = _s()
    s.booking_started = True
    s.slots = {}
    _heard(s, "do you use ammonia free colour?")
    out = s.recovery_line("unknown")
    assert out, "the turn was left with nothing to say"
    assert out != s.slot_question("name"), \
        "the caller's question was replaced by a slot question"
    assert s.safe_say("unknown")[:18] in out, "no honest refusal in the reply"
    assert s.slot_question("name") in out, "the booking was abandoned instead of continued"


def test_an_ANSWER_mid_booking_still_gets_the_slot_question():
    """The original fix must survive: a caller answering "which day and time?" with
    "दस बजे सुबह" must NOT hear "I'll check with the desk"."""
    s = _s()
    s.booking_started = True
    s.slots = {}
    _heard(s, "दस बजे सुबह")
    out = s.recovery_line("unknown")
    assert out == s.slot_question("name"), \
        f"a slot answer was met with a refusal: {out[:60]!r}"


def test_a_question_with_all_slots_known_still_confirms():
    s = _s()
    s.booking_started = True
    s.slots = {"name": "Manu", "phone": "8920429057",
               "doctor": "Dr Anil Sharma", "datetime": "10am"}
    _heard(s, "दस बजे सुबह")
    assert s.recovery_line("unknown") == s.confirm_question()


@pytest.mark.parametrize("asked", [
    "do you use ammonia free colour?",
    "क्या आपके पास पार्किंग है?",
    "what is your cancellation policy",
    "మీ దగ్గర పార్కింగ్ ఉందా?",
])
def test_questions_are_recognised_in_every_script(asked):
    s = _s()
    _heard(s, asked)
    assert s._asked_us_something(), f"not recognised as a question: {asked!r}"


@pytest.mark.parametrize("answered", ["Manu Mishra", "8920429057", "दस बजे सुबह",
                                      "tomorrow at 10", "haan theek hai"])
def test_slot_answers_are_not_mistaken_for_questions(answered):
    s = _s()
    _heard(s, answered)
    assert not s._asked_us_something(), f"an answer read as a question: {answered!r}"
