"""Tests for the grounding guard — the layer that stops the model inventing facts.

Every case here is a real failure observed from qwen3:4b on the clinic pack:
a fabricated ₹1,000 neurologist fee, a Hindi reply guillotined mid-word by the
token budget, and Telugu replies that collapsed into repeating one clause.

Run:  pytest -q
"""
import json

from zensuvidha.guard import (GuardConfig, check_reply, is_degenerate, looks_hinglish,
                              numbers_in, pack_numbers, safe_line, script_share,
                              trim_incomplete, ungrounded_numbers)
from zensuvidha.orchestrator import Session, extract_kind, next_chunk
from zensuvidha.packs import load_pack


class FakeLLM:
    """Returns a canned reply, so these tests need no Ollama."""

    def __init__(self, payload, finish_reason="stop"):
        self._payload = payload
        self.num_predict = 200
        self.num_ctx = 6144
        self.finish_reason = finish_reason

    def chat(self, messages, force_json=True, model=None, num_predict=None, meta=None,
             num_ctx=None):
        self.last_num_predict = num_predict
        if meta is not None:
            meta["finish_reason"] = self.finish_reason
        return json.dumps(self._payload, ensure_ascii=False)


CLINIC = load_pack("clinic")


# ---- number extraction -----------------------------------------------------
def test_numbers_normalise_across_scripts_and_formats():
    assert numbers_in("₹1,500") == numbers_in("1500") == {"1500"}
    assert "500" in numbers_in("फ़ीस ५०० रुपये है।")        # Devanagari digits
    assert "500" in numbers_in("ఫీజు ౫౦౦ రూపాయలు")          # Telugu digits


def test_phone_matches_however_it_is_grouped():
    grouped = numbers_in("+91 20 1234 5678")
    assert "912012345678" in grouped and "1234" in grouped


# ---- grounding -------------------------------------------------------------
def test_pack_facts_are_grounded():
    allowed = pack_numbers(CLINIC)
    for legit in ["General Physician is ₹500 and Dermatology ₹700.",
                  "Open Mon–Sat 9am to 8pm, lunch 2 to 3pm.",
                  "Full-body check is ₹4,000, reports in 24 hours.",
                  "సాధారణ వైద్యుడు ₹500."]:
        assert not ungrounded_numbers(legit, allowed), legit


def test_invented_fee_is_caught():
    """The exact failure this guard exists for."""
    allowed = pack_numbers(CLINIC)
    assert ungrounded_numbers("Neurologist ka consultation fee ₹1,000 hai.", allowed) == ["1000"]
    assert ungrounded_numbers("पुणे से मुंबई लगभग 170 किलोमीटर है।", allowed) == ["170"]


def test_ungrounded_reply_is_replaced_with_the_safe_line():
    out, reason = check_reply("Neurologist fee is ₹1,000.", pack=CLINIC,
                              allowed_numbers=pack_numbers(CLINIC), lang_name="English")
    assert reason and reason.startswith("ungrounded_numbers")
    assert "1,000" not in out and "front desk" in out


def test_log_only_mode_reports_but_does_not_rewrite():
    text = "Neurologist fee is ₹1,000."
    out, reason = check_reply(text, pack=CLINIC, allowed_numbers=pack_numbers(CLINIC),
                              lang_name="English", gcfg=GuardConfig({"log_only": True}))
    assert out == text and reason.startswith("ungrounded_numbers")


def test_guard_can_be_switched_off():
    text = "Neurologist fee is ₹1,000."
    out, reason = check_reply(text, pack=CLINIC, allowed_numbers=pack_numbers(CLINIC),
                              lang_name="English", gcfg=GuardConfig({"enabled": False}))
    assert out == text and reason is None


# ---- repetition ------------------------------------------------------------
def test_repeated_clause_is_degenerate():
    looped = ("మేము మంగళవారం నుండి రోజు మిగిలిన సమయం వరకు ఉంటారు, "
              "శనివారం నుండి రోజు మిగిలిన సమయం వరకు ఉంటారు.")
    assert is_degenerate(looped)


def test_normal_replies_are_not_degenerate():
    for ok in ["We offer Basic ₹1,500, Full-body ₹4,000, Diabetic ₹2,500 packages.",
               "Dr Sharma consults 9am to 2pm and Dr Mehta 3pm to 8pm, Monday to Saturday.",
               "क्षमा कीजिए, यह जानकारी मेरे पास नहीं है। क्या मैं और किसी बात में मदद कर सकती हूँ?",
               "Yes. Okay. Sure."]:
        assert not is_degenerate(ok), ok


# ---- truncation ------------------------------------------------------------
def test_truncated_tail_is_dropped_not_spoken():
    cut = "नमस्ते! हमारे पास न्यूरोलॉजिस्ट नहीं है। आपको किस डॉक्टर से संपर्क करना"
    assert trim_incomplete(cut) == "नमस्ते! हमारे पास न्यूरोलॉजिस्ट नहीं है।"


def test_short_reply_without_punctuation_is_left_alone():
    assert trim_incomplete("हाँ, ज़रूर") == "हाँ, ज़रूर"


def test_a_missing_full_stop_is_not_truncation():
    """Regression — the reply was being cut off mid-conversation. Models routinely leave
    the final period off, and trimming on that guess silently ate the last sentence."""
    for complete in ["Hello there! How can I help you today",
                     "Sure, I can book that. What day works for you",
                     "जनरल फिजिशियन की फीस ₹500 है। कोई और सवाल"]:
        out, reason = check_reply(complete, pack=CLINIC, allowed_numbers=pack_numbers(CLINIC),
                                  lang_name=None)                 # truncated=False by default
        assert out == complete, f"guard swallowed the tail of: {complete}"
        assert reason is None


def test_trimming_still_happens_when_the_model_reports_it_ran_out():
    cut = "Dr Sharma consults 9am to 2pm. Dr Mehta sees patients in the after"
    out, reason = check_reply(cut, pack=CLINIC, allowed_numbers=pack_numbers(CLINIC),
                              lang_name=None, truncated=True)
    assert out == "Dr Sharma consults 9am to 2pm." and reason == "truncated"


def test_finish_reason_reaches_the_guard():
    """finalize must only trim when the LLM reported finish_reason == 'length'."""
    payload = {"kind": "answer", "say": "Hello there! How can I help you today",
               "action": {"type": "none"}}
    s = _session(payload)
    s.begin_user("hi")
    assert s.finalize(json.dumps(payload), {"finish_reason": "stop"})["say"] == payload["say"]

    s2 = _session(payload)
    s2.begin_user("hi")
    assert s2.finalize(json.dumps(payload), {"finish_reason": "length"})["say"] == "Hello there!"


# ---- language --------------------------------------------------------------
def test_script_share_spots_a_fallback_into_english():
    assert script_share("జనరల్ ఫిజీషియన్ ఫీజు ₹500, Dr Rao", "Telugu") > 0.5   # loanwords fine
    assert script_share("The consultation fee is 500 rupees.", "Telugu") == 0.0


def test_english_reply_to_a_telugu_caller_is_replaced():
    out, reason = check_reply("The consultation fee is 500 rupees.", pack=CLINIC,
                              allowed_numbers=pack_numbers(CLINIC), lang_name="Telugu")
    assert reason.startswith("wrong_language")
    assert script_share(out, "Telugu") > 0.5


def test_booking_confirmation_survives_the_language_check():
    """Regression: a Hindi confirmation ending in the English booking reference was
    being scored as 'wrong language' and thrown away — losing a real booking."""
    say = "कल शाम 5 बजे. Your booking reference is #6."
    out, reason = check_reply(say, pack=CLINIC, allowed_numbers=pack_numbers(CLINIC) | {"5", "6"},
                              lang_name="Hindi", check_language=False)
    assert out == say and reason is None


# ---- safe lines ------------------------------------------------------------
def test_every_language_has_every_line():
    from zensuvidha.guard import SAFE_LINES
    for lang, table in SAFE_LINES.items():
        assert {"unknown", "scope", "repeat", "ref"} <= set(table), lang
        for key, line in table.items():
            assert line.strip() and "{" not in line.format(biz="X", ref=1)


def test_scope_line_names_the_business():
    assert "Suvidha Clinic" in safe_line("scope", "Hindi", CLINIC)


def test_unknown_language_falls_back_to_english_not_gibberish():
    line = safe_line("unknown", "Swahili", CLINIC)
    assert line == safe_line("unknown", "English", CLINIC)


# ---- hinglish --------------------------------------------------------------
def test_hinglish_is_detected_but_english_is_not():
    assert looks_hinglish("bhai neurologist ka kya rate hai?")
    assert looks_hinglish("mujhe kal subah appointment chahiye")
    assert not looks_hinglish("What health checkup packages do you have?")
    assert not looks_hinglish("Is the doctor available today?")


# ---- streaming helper ------------------------------------------------------
def test_extract_kind_reads_the_verdict_before_the_words():
    assert extract_kind('{"kind": "out_of_scope", "say": "') == "out_of_scope"
    assert extract_kind('{"kind": "unk') is None          # not arrived yet
    assert extract_kind('{"say": "hello"}') is None


# ---- chunking: punctuation inside numbers ---------------------------------
def _stream(full: str) -> list:
    """Chunks as they would be spoken, feeding the buffer one character at a time."""
    start, first, out = 0, True, []
    for n in range(1, len(full) + 1):
        chunk, start = next_chunk(full[:n], start, clause=first)
        if chunk:
            out.append(chunk)
            first = False
    return out


def test_a_price_is_never_split_across_chunks():
    """Regression: the clause break fired on the comma in "₹1,500", so the voice said
    "one" then "five hundred" — and half of a fabricated "₹1,000" was spoken before
    the guard could see the rest of it."""
    assert _stream("Neurologist ka consultation fee ₹1,000 hai.") == \
        ["Neurologist ka consultation fee ₹1,000 hai."]
    # the decimal/colon stay inside one chunk; a comma between WORDS may still break
    assert _stream("The scan takes 1.5 hours, roughly.")[0] == "The scan takes 1.5 hours,"
    assert _stream("Dr Rao consults at 5:30pm on Tuesday.") == ["Dr Rao consults at 5:30pm on Tuesday."]


def test_first_chunk_still_breaks_early_on_a_real_clause():
    """The latency trick must survive the fix: greet before the sentence is finished."""
    assert _stream("Namaste, how may I help you today?")[0] == "Namaste,"


def test_partial_price_is_never_emitted():
    assert next_chunk("Neurologist fee ₹1,", 0, clause=True) == ("", 0)


# ---- session-level end-to-end ---------------------------------------------
def _session(payload):
    return Session(load_pack("clinic"), FakeLLM(payload))


def test_out_of_scope_answers_in_the_callers_language():
    s = _session({"kind": "out_of_scope", "say": "पुणे से मुंबई 170 किलोमीटर है।",
                  "action": {"type": "none"}})
    say = s.handle_text("पुणे से मुंबई कितनी दूर है?")["say"]
    assert "170" not in say
    assert say == safe_line("scope", "Hindi", CLINIC)


def test_invented_number_is_caught_even_when_model_claims_it_is_answering():
    s = _session({"kind": "answer", "say": "Neurologist ki fee ₹1,000 hai.",
                  "action": {"type": "none"}})
    say = s.handle_text("bhai checkup ka kya rate hai?")["say"]
    assert "1,000" not in say
    assert say == safe_line("unknown", "Hindi", CLINIC, romanized=True)   # Hinglish reply


def test_unknown_kind_wins_over_a_stray_collect_action():
    """The model flags 'unknown' but attaches a collect action; answering
    'may I have the patient's full name?' to 'do you do dialysis?' is the bug."""
    s = _session({"kind": "unknown", "say": "",
                  "action": {"type": "collect", "slots": {}, "missing": ["name"]}})
    say = s.handle_text("tumhare yahan raat ko bhi doctor milta hai kya?")["say"]
    assert "name" not in say.lower()
    assert say == safe_line("unknown", "Hindi", CLINIC, romanized=True)


def test_grounded_answer_passes_through_untouched():
    s = _session({"kind": "answer", "say": "जनरल फिजिशियन की फीस ₹500 है।",
                  "action": {"type": "none"}})
    assert s.handle_text("जनरल फिजिशियन की फीस कितनी है?")["say"] == "जनरल फिजिशियन की फीस ₹500 है।"


def test_caller_own_number_may_be_read_back():
    s = _session({"kind": "answer", "say": "Your number 9876512345 — is that right?",
                  "action": {"type": "none"}})
    say = s.handle_text("my number is 9876512345")["say"]
    assert "9876512345" in say


def test_booking_reference_is_spoken_in_the_callers_language(tmp_path, monkeypatch):
    import zensuvidha.booking as b
    monkeypatch.setattr(b, "DB", tmp_path / "g.db")
    s = _session({"kind": "answer", "say": "ठीक है, बुक कर दिया।",
                  "action": {"type": "book_appointment",
                             "slots": {"name": "मनु", "phone": "9876512345",
                                       "doctor": "Dr Sharma", "datetime": "कल शाम 5 बजे"}}})
    res = s.handle_text("कल शाम 5 बजे डॉक्टर शर्मा से अपॉइंटमेंट")
    assert res["action"]["booking_id"] == 1
    assert "#1" in res["say"] and "booking reference" not in res["say"]  # Hindi, not English
    assert "रेफरेंस" in res["say"]


# ---- token budget ----------------------------------------------------------
def test_indic_replies_get_a_bigger_token_budget():
    """Measured on Qwen's tokenizer: the same sentence costs 3.5× in Hindi and 6.2×
    in Telugu. Without scaling, the reply is cut off mid-word — garbled speech."""
    s = _session({"kind": "answer", "say": "ठीक है।", "action": {"type": "none"}})
    s.handle_text("जनरल फिजिशियन की फीस कितनी है?")
    assert s.llm.last_num_predict == 200 * 4          # Hindi

    s_te = _session({"kind": "answer", "say": "సరే.", "action": {"type": "none"}})
    s_te.handle_text("మీ క్లినిక్ ఎన్ని గంటలకు తెరుస్తారు?")
    assert s_te.llm.last_num_predict == 200 * 7       # Telugu costs far more

    s2 = _session({"kind": "answer", "say": "Sure.", "action": {"type": "none"}})
    s2.handle_text("What are your timings?")
    assert s2.llm.last_num_predict is None          # English → model default


# ---- Indic-safe tokenising + native-language facts --------------------------
def test_word_splitter_does_not_destroy_indic_words():
    """`\\w` does not match Unicode combining marks, so re.findall(r"\\w+", "आपकी फीस
    कितनी") returns just ["आपक"] — every matra dropped. That silently broke retrieval
    for every Indian language."""
    from zensuvidha.orchestrator import _words
    assert _words("आपकी फीस कितनी है?") == ["आपकी", "फीस", "कितनी", "है"]
    assert _words("మీ ఫీజులు ఎంత?") == ["మీ", "ఫీజులు", "ఎంత"]
    assert _words("what are your timings?") == ["what", "are", "your", "timings"]


def test_native_facts_are_retrieved_in_the_callers_language():
    s = _session({"kind": "answer", "say": "x", "action": {"type": "none"}})
    for query, expect, lang in [
        ("कौन सा डॉक्टर किस समय अवेलेबल है?", "डॉ. शर्मा", "Hindi"),
        ("आपकी फीस कितनी है?", "₹500", "Hindi"),
        ("ఏ డాక్టర్ ఏ సమయంలో అందుబాటులో ఉంటారు?", "డాక్టర్ శర్మ", "Telugu"),
        ("మీ ఫీజులు ఎంత?", "₹500", "Telugu"),
    ]:
        assert expect in s.native_facts(lang, query), query


def test_native_facts_stay_small_enough_for_the_context():
    """Pasting the whole knowledge base in Telugu (~6x English tokens) overflowed
    num_ctx and made the model refuse questions it could answer."""
    s = _session({"kind": "answer", "say": "x", "action": {"type": "none"}})
    facts = s.native_facts("Telugu", "మీ ఫీజులు ఎంత?")
    assert 0 < len(facts.splitlines()) <= s.NATIVE_FACTS_PER_TURN + 1


def test_english_callers_get_no_native_block():
    s = _session({"kind": "answer", "say": "x", "action": {"type": "none"}})
    assert s.native_facts("English", "what are your fees?") == ""


# ---- services the business does not offer ----------------------------------
def test_unoffered_service_is_refused_without_asking_the_model():
    """The grounding guard cannot catch a FALSE CLAIM built from REAL numbers: asked for
    a neurologist the model answered "consultation fee ₹500" — and ₹500 is a genuine
    pack figure (the GP fee), so every number checked out. The pack lists what it does
    not do, and that is answered deterministically."""
    # a model that WOULD hallucinate; it must never be consulted
    bad = {"kind": "answer", "say": "Neurologist ka consultation fee ₹500 hai, bhai.",
           "action": {"type": "none"}}
    for q in ["bhai neurologist ka kya rate hai?",
              "do you do dialysis here?",
              "क्या आपके यहाँ दांत का इलाज होता है?",
              "MRI స్కాన్ ధర ఎంత?"]:
        s = _session(bad)
        say = s.handle_text(q)["say"]
        assert "500" not in say, f"quoted a real price for an unoffered service: {q}"
        assert say == s.safe_say("scope"), q


def test_offered_services_are_not_caught_by_the_unoffered_list():
    ok = {"kind": "answer", "say": "General Physician is ₹500.", "action": {"type": "none"}}
    for q in ["what is the consultation fee?", "do you do blood tests?",
              "is there a pediatrician?", "आपकी फीस कितनी है?", "మీ ఫీజులు ఎంత?"]:
        s = _session(ok)
        assert s.handle_text(q)["say"] != s.safe_say("scope"), q


# ---- stuck-loop across turns -----------------------------------------------
def test_model_repeating_its_previous_reply_is_broken_out_of():
    """Observed on a real call: the caller reported stomach pain, then asked to book,
    then gave their name — and every turn came back "मैं अच्छा हूँ, कैसे मैं आपकी मदद कर
    सकता हूँ?". is_degenerate() only looks INSIDE one reply; nothing watched across turns."""
    stuck = {"kind": "answer", "say": "मैं अच्छा हूँ, धन्यवाद! कैसे मैं आपकी मदद कर सकता हूँ?",
             "action": {"type": "none"}}
    s = _session(stuck)                       # a model that ALWAYS returns the same line
    first = s.handle_text("नमस्ते, आप कैसे हैं?")["say"]
    assert first == stuck["say"]              # fine the first time
    second = s.handle_text("मेरे लिए अपॉइंटमेंट बुक कर दो")["say"]
    assert second != first, "agent parroted its previous reply"
    third = s.handle_text("मेरा नाम राहुल है")["say"]
    assert third != first


def test_short_acknowledgements_may_recur():
    s = _session({"kind": "answer", "say": "हाँ, ज़रूर।", "action": {"type": "none"}})
    assert s.handle_text("ठीक है?")["say"] == "हाँ, ज़रूर।"
    assert s.handle_text("और एक बात")["say"] == "हाँ, ज़रूर।"   # too short to be a loop
