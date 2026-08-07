"""The Base engine. One runtime, any industry — behaviour comes entirely from
the loaded Industry Pack.

Domain knowledge (persona, business facts, services/menu, playbooks, policies, slot
questions, and the FULL knowledge base) is baked into the cache-stable system prompt, so
it's always available in EVERY language — no fragile per-turn English-keyword retrieval.

Turn primitives (begin_user / finalize) are shared by the blocking CLI and the
streaming server. Streaming helpers (extract_say / next_chunk) let the server
speak sentence by sentence as tokens arrive.
"""
import base64
import json
import logging
import re

from .booking import create_booking
from . import expectation
from .guard import (GuardConfig, LANG_SCRIPT, TOKEN_FACTOR, ask_line, check_reply,
                    is_degenerate, looks_hinglish, looks_like_echo, norm_digits, numbers_in,
                    pack_numbers, safe_line)

log = logging.getLogger("zensuvidha.orchestrator")

_SENT_END = ".!?…।\n"

# The model's own classification of a turn → which pre-written safe line answers it.
# ("answer" is absent on purpose: those turns keep the model's words.)
_KIND_TO_SAFE = {"unknown": "unknown", "out_of_scope": "scope", "unclear": "repeat"}

# language code → display name (for the forced-language reply instruction)
LANG_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "or": "Odia", "as": "Assamese",
}
# display name → code (to latch/pin STT once we know the caller's language)
NAME_TO_CODE = {v: k for k, v in LANG_NAMES.items()}

# Unicode script ranges → language name, for auto-detecting the reply language
# from the transcript when the caller hasn't pinned one.
_SCRIPT_LANG = [
    (0x0900, 0x097F, "Hindi"), (0x0980, 0x09FF, "Bengali"), (0x0A00, 0x0A7F, "Punjabi"),
    (0x0A80, 0x0AFF, "Gujarati"), (0x0B00, 0x0B7F, "Odia"), (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"), (0x0D00, 0x0D7F, "Malayalam"),
    (0x0600, 0x06FF, "Urdu"),
]


# synonyms the LLM might use for a required booking field → canonical field name
_SLOT_SYNONYMS = {
    "name": {"name", "full_name", "fullname", "patient_name", "customer_name", "guest_name"},
    "phone": {"phone", "mobile", "number", "contact", "phone_number", "mobile_number", "cell"},
    "datetime": {"datetime", "date", "time", "when", "slot", "day", "date_time",
                 "appointment_time", "appointment", "pickup_time", "reservation_time"},
    "doctor": {"doctor", "physician", "specialist", "department", "doctor_name"},
    "service": {"service", "treatment", "procedure", "package"},
    "party_size": {"party_size", "people", "guests", "pax", "persons", "party", "number_of_people"},
}


def _slot_alias_map(required: list) -> dict:
    """Map every synonym of a required field → that field name (lowercased keys)."""
    m = {}
    for field in required:
        m[str(field).lower()] = field
        for a in _SLOT_SYNONYMS.get(field, ()):
            m[a] = field
    return m


def clean_for_speech(text: str) -> str:
    """Strip markdown/symbols the TTS would otherwise read aloud (`**`, `` ` ``, `#`,
    bullets, table pipes). Currency symbols (₹) are left for the voice, which reads them
    correctly. The DISPLAYED chat text keeps its formatting — only the TTS input is cleaned."""
    if not text:
        return ""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)      # code fences
    t = re.sub(r"[*_`#>|]", "", t)                       # md emphasis / headings / tables
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)        # bullet markers
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)       # [text](url) → text
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# Word splitter that is SAFE FOR INDIC TEXT. `\w` does not match Unicode combining
# marks (category Mn), so re.findall(r"\w+", "आपकी फीस कितनी") silently returns just
# ["आपक"] — every matra is dropped and the words are destroyed. Split on separators
# instead, so the script's own characters are never thrown away.
_TOKEN_SPLIT = re.compile(r"[\s,.;:!?।॥…\"'`()\[\]{}/\\|—–\-_+*=<>@#$%^&~]+")


def _words(text: str) -> list:
    return [w for w in _TOKEN_SPLIT.split((text or "").lower()) if w]


# Numbers people SAY rather than type. Only the small ones matter — a caller says
# "six months" and "ten o'clock", never "four thousand two hundred". Kept per language
# because the reply is grounded against the caller's own words in their own language.
_SPOKEN_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "hundred": "100",
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5", "पांच": "5",
    "छह": "6", "छे": "6", "सात": "7", "आठ": "8", "नौ": "9", "दस": "10",
    "ग्यारह": "11", "बारह": "12", "बीस": "20", "तीस": "30", "सौ": "100",
    "ఒకటి": "1", "రెండు": "2", "మూడు": "3", "నాలుగు": "4", "ఐదు": "5",
    "ఆరు": "6", "ఏడు": "7", "ఎనిమిది": "8", "తొమ్మిది": "9", "పది": "10",
    "పదకొండు": "11", "పన్నెండు": "12", "ఇరవై": "20", "వంద": "100",
}


def spoken_numbers(text: str) -> set:
    """Digits the caller uttered as words. See the call site in begin_user."""
    out = set()
    for w in _words(text or ""):
        d = _SPOKEN_NUMBERS.get(w.lower())
        if d:
            out.add(d)
    return out


# A name never contains any of these. Anchored where it matters so "Ram Iyer" survives.
_REQUEST_RE = re.compile(
    r"\b(i need|i want|i would|i'd like|can i|could i|do you|does the|is there|are you|"
    r"how much|how long|what is|what are|please|appointment|certificate|report|test)\b"
    r"|मुझे|चाहिए|क्या आप|कितना|कितनी|कैसे"
    r"|నాకు|కావాలి|ఎంత|ఎలా")


# The smallest thing that is still a WAV: a RIFF header plus enough of a fmt chunk to
# describe itself. Anything shorter cannot be decoded.
_MIN_WAV = 44


def _playable(audio):
    """Return `audio` only if it is something a browser can actually play.

    A provider is free to hand back whatever it likes on failure, and one that returns
    a short error string instead of audio used to be forwarded to the caller as if it
    were sound — the client then failed to decode it and the turn was silent with no
    reason recorded anywhere. Treat unplayable output as no output, which is a case the
    rest of the pipeline already handles properly.
    """
    if not audio:
        return None
    if len(audio) < _MIN_WAV or not (audio[:4] == b"RIFF" or audio[:4] == b"FORM"
                                     or audio[:3] == b"ID3" or audio[:4] == b"OggS"):
        import logging
        logging.getLogger("zensuvidha.tts").warning(
            "TTS returned %d bytes that are not audio — treating as a failed synth",
            len(audio))
        return None
    return audio


def dominant_script_lang(text: str, min_share: float = 0.35):
    """Return the language name only if a single non-Latin script DOMINATES the text
    (>= min_share of its letters). Used to LATCH the call's language so a stray
    misrecognised character can't flip the reply language every turn."""
    counts, total = {}, 0
    for c in text or "":
        if not c.isalpha():
            continue
        total += 1
        for lo, hi, name in _SCRIPT_LANG:
            if lo <= ord(c) <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not total or not counts:
        return None
    name, n = max(counts.items(), key=lambda kv: kv[1])
    return name if n / total >= min_share else None


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def _fmt_services(services) -> str:
    out = []
    for s in services:
        line = "- " + s.get("name", "")
        detail = []
        if s.get("detail"):
            detail.append(s["detail"])
        if s.get("price"):
            detail.append(f"price {s['price']}")
        if s.get("duration"):
            detail.append(f"~{s['duration']}")
        if detail:
            line += " — " + "; ".join(detail)
        out.append(line)
    return "\n".join(out)


# People answer "what is your name?" with a whole sentence, and the booking record then
# reads "my name is Manu Mishra". Only the name belongs in the field. Anything not
# matched here is kept verbatim — stripping too eagerly would lose part of a real name.
_NAME_LEAD = re.compile(
    r"^\s*(?:"
    r"my\s+name\s+is|my\s+name's|name\s+is|i\s+am|i'm|this\s+is|it\s+is|it's|"
    r"mera\s+naam|mera\s+nam|naam\s+hai|"
    r"मेरा\s+नाम|मेरी\s+नाम|नाम\s+है|"
    r"నా\s+పేరు|என்\s+பெயர்|ನನ್ನ\s+ಹೆಸರು|എന്റെ\s+പേര്|আমার\s+নাম|"
    r"माझं\s+नाव|माझे\s+नाव|મારું\s+નામ|ਮੇਰਾ\s+ਨਾਂ|ਮੇਰਾ\s+ਨਾਮ|میرا\s+نام|ମୋ\s+ନାମ"
    r")\s*[:,]?\s*", re.IGNORECASE)
# …and the trailing copula those sentences end with.
_NAME_TAIL = re.compile(r"\s*(?:है|hai|ఉంది|ఆగు|아니)\s*[.।]?\s*$", re.IGNORECASE)


def _strip_name_frame(value: str) -> str:
    """"my name is Manu Mishra" → "Manu Mishra"; anything unrecognised is left alone."""
    out = _NAME_TAIL.sub("", _NAME_LEAD.sub("", value or "")).strip(" .,।॥")
    return out or value


_HESITATIONS = frozenset("""
hmm hm hmmm uh uhh um umm er err ah aah oh ok okay yeah yep yes no nope sure right
haan han haa ji jee acha accha achha theek thik matlab
हाँ हां हम्म अच्छा ठीक जी अरे मतलब ओके नहीं ना
అవును సరే ఓకే హా అబ్బా అలాగే
ஆம் சரி ஓகே ஹா
ಹೌದು ಸರಿ ಓಕೆ
അതെ ശരി ഓകെ
হ্যাঁ ঠিক ওকে আচ্ছা
हो बरं ठीक
હા બરાબર ઓકે
ਹਾਂ ਠੀਕ ਓਕੇ
ہاں ٹھیک اوکے
ହଁ ଠିକ ଓକେ
""".split())


def build_system_prompt(pack: dict) -> str:
    p = pack.get("persona", {})
    biz = pack.get("business", {})
    esc = pack.get("escalation", {})
    intents = ", ".join(i["name"].replace("_", " ") for i in pack.get("intents", []))
    required = pack.get("booking", {}).get("required", [])
    slot_q = pack.get("booking", {}).get("slots", {})

    S = []
    S.append(f"You are {p.get('role', 'the receptionist')} at {biz.get('name', 'the business')}, "
             f"answering the phone. Your manner is {p.get('style', 'warm and professional')}.")
    S.append(
        "Talk like a real human receptionist on a call — natural, warm, and BRIEF (1–2 short "
        "sentences). Callers may speak English, Hindi, Hinglish, or another Indian language; "
        "reply in the SAME single language and script the caller is using. Use exactly ONE language "
        "per reply — never mix two languages in a sentence, and never switch the language on your own. "
        "If the caller writes an Indian language in Latin letters (Hinglish, e.g. 'mujhe appointment "
        "chahiye'), reply the same way — in Latin letters, not the native script. "
        "Answer only what was asked — don't recite lists or over-explain. Acknowledge, then move the "
        "call forward, asking one question at a time. Never repeat something you already said.\n"
        "IMPORTANT: if the caller's words are unclear, garbled, cut off, or sound like background "
        "noise, do NOT guess or invent a reply — warmly say you didn't catch that and ask them to "
        "repeat.")
    S.append(
        "The caller's words come from speech recognition and may contain small errors or odd "
        "spellings (e.g. 'Gune' for 'Pune', 'serry' for 'saree', 'doctor sharma' for 'Dr Sharma') — "
        "interpret them by CONTEXT and respond to what they clearly meant; never nitpick spelling. "
        "Handle greetings, thank-yous, and light small talk warmly and briefly, and you may answer "
        "simple questions about who you are and how you can help. "
        "But for unrelated general knowledge — directions, distances, weather, other companies — "
        "politely say that's outside what you can help with, and steer back to what you CAN do here.")
    if intents:
        S.append(f"WHAT YOU HANDLE: {intents}.")

    biz_lines = [f"- Name: {biz.get('name', '')}"]
    for label, key in [("Tagline", "tagline"), ("Established", "established"), ("Hours", "hours"),
                       ("Address", "address"), ("Landmark", "landmark"), ("Phone", "phone"),
                       ("WhatsApp", "whatsapp"), ("Email", "email"), ("Website", "website"),
                       ("Parking", "parking"), ("Payment", "payment")]:
        if biz.get(key):
            biz_lines.append(f"- {label}: {biz[key]}")
    S.append("BUSINESS INFO — the ONLY facts you may state. Never invent names, prices, timings, phone "
             "numbers, or availability. If asked something not covered here or in the knowledge below, "
             "say you'll check and take a message — do NOT guess:\n"
             + "\n".join(biz_lines))

    if pack.get("services"):
        S.append("SERVICES / MENU:\n" + _fmt_services(pack["services"]))
    if pack.get("scenarios"):
        S.append("HOW TO HANDLE COMMON SITUATIONS:\n"
                 + "\n".join(f"- {s['name']}: {s['do']}" for s in pack["scenarios"]))
    if pack.get("policies"):
        S.append("POLICIES (always follow):\n" + "\n".join(f"- {x}" for x in pack["policies"]))

    # Full knowledge base baked into the (cache-stable) system prompt so it's ALWAYS
    # available — including for Hindi/Telugu/etc. callers, where per-turn keyword retrieval
    # (English tags vs native-script query) would otherwise match nothing.
    kb = (pack.get("knowledge") or pack.get("faq") or []) + (pack.get("common") or [])
    if kb:
        S.append("KNOWLEDGE BASE — answer FROM these facts (keep numbers, days, prices, names exact; "
                 "translate only the wording into the caller's language). If something isn't covered "
                 "here or above, say you'll check and take a message — never invent it:\n"
                 + "\n".join(f"- Q: {it.get('q', 'info')}\n  A: {it['a']}" for it in kb))

    if required:
        q_lines = []
        for f in required:
            q = slot_q.get(f)
            q_lines.append(f'- {f}' + (f' → ask: "{q}"' if q else ""))
        S.append("TO COMPLETE A BOOKING, collect the fields below, asking for any that are missing "
                 "ONE at a time:\n" + "\n".join(q_lines)
                 + "\nEACH turn, put EVERY detail gathered so far in action.slots (never drop earlier "
                   "answers). Understand natural phrasing — 'tomorrow', 'Monday at 4pm', 'general "
                   "physician' — and fill the matching slot. Do NOT say the booking is done, and use "
                   "type \"book_appointment\" ONLY, when ALL required fields are filled; until then use "
                   "\"collect\" and ask for the next missing one. Read the phone number back to confirm.")

    esc_cond = "; ".join(esc.get("conditions", [])) or "the caller is angry or asks for a person"
    S.append(f'ESCALATE to a human (action.type="escalate") if: {esc_cond}; or the caller explicitly '
             f'asks for a person.')

    S.append("NEVER INVENT: only state prices, timings, names, phone numbers, addresses, or availability "
             "that appear in BUSINESS INFO, SERVICES, or the knowledge provided. If you are not sure of a "
             "specific detail, do NOT guess a number or name — say you'll confirm with the desk and offer to "
             "take a message. It is always better to say 'let me confirm that' than to state something wrong. "
             "A number you did not read above is ALWAYS wrong — there is no such thing as a reasonable "
             "estimate for a fee, a timing, or a phone number.")

    S.append('OUTPUT — reply with ONE JSON object and nothing else, with "kind" FIRST:\n'
             '{"kind": "answer|unknown|out_of_scope|unclear", "say": "<your spoken reply>", '
             '"action": {"type": "none|collect|book_appointment|escalate", '
             '"slots": {<field: value collected so far>}, "missing": [<fields still needed>]}}\n'
             '\n"kind" is the MOST IMPORTANT field — choose it honestly:\n'
             '- "answer" — everything you are about to say comes from BUSINESS INFO, SERVICES or the '
             'KNOWLEDGE BASE above, or you are collecting booking details, greeting, or making small talk.\n'
             '- "unknown" — the caller asked about THIS business, but the answer is NOT written above: a '
             'service, price, doctor, timing or detail that simply is not listed. Choose this instead of '
             'estimating.\n'
             '- "out_of_scope" — the caller asked something a receptionist here does not handle at all: '
             'general knowledge, distances, weather, news, other companies, or medical/legal advice.\n'
             '- "unclear" — the words were empty, garbled, cut off, or sounded like background noise.\n'
             '\nFor "unknown", "out_of_scope" and "unclear" you may leave "say" empty — the correct sentence '
             'will be spoken in the caller\'s own language for you. Never fill those cases with a guess.\n'
             'Rules: use "book_appointment" ONLY when every required field is filled, otherwise "collect". '
             'Keep "say" short, natural, and specific. Output nothing outside the JSON object.')
    return "\n\n".join(S)


# --------------------------------------------------------------------------- #
def extract_say(buf: str):
    """Incrementally pull the "say" string out of a partial JSON buffer.
    Returns (text_so_far or None, is_terminated)."""
    m = re.search(r'"say"\s*:\s*"', buf)
    if not m:
        return None, False
    i = m.end()
    out, esc = [], False
    while i < len(buf):
        c = buf[i]
        if esc:
            if c == "u":
                # \uXXXX — decode to the real character (matters for Hindi/Telugu/…
                # when Ollama JSON-escapes non-ASCII). Wait if the 4 hex digits
                # haven't fully arrived in this partial buffer yet.
                hexs = buf[i + 1:i + 5]
                if len(hexs) < 4:
                    break
                try:
                    out.append(chr(int(hexs, 16)))
                    i += 5
                    esc = False
                    continue
                except ValueError:
                    out.append("u")
            else:
                out.append({"n": "\n", "t": "\t", "r": ""}.get(c, c))
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            return "".join(out), True
        else:
            out.append(c)
        i += 1
    return "".join(out), False


def extract_kind(buf: str):
    """Pull the "kind" classification out of a (possibly partial) JSON buffer.

    The prompt puts "kind" FIRST precisely so the streaming path can read it before
    any spoken words arrive: a turn the model itself flags as "unknown" or
    "out_of_scope" is then answered with a clean pre-written line, and the model's
    own (often garbled, often invented) attempt is never synthesised at all.
    Returns None while it hasn't arrived yet.
    """
    m = re.search(r'"kind"\s*:\s*"([a-z_]*)"', buf)
    return m.group(1) if m else None


_CLAUSE_END = ",;:—"


def next_chunk(text: str, start: int, clause: bool = False, final: bool = False):
    """Return (chunk, new_start): the completed span of text[start:].

    Normally we break on sentence enders so speech sounds natural. For the VERY
    FIRST chunk of a reply, pass clause=True to also break on commas/semicolons —
    this lets us emit "Namaste," as audio the instant it's generated, shaving a
    few hundred ms off time-to-first-audio without hurting later sentences.
    """
    enders = _SENT_END + (_CLAUSE_END if clause else "")
    end = -1
    for i in range(start, len(text)):
        if text[i] not in enders:
            continue
        # Punctuation INSIDE a number is not a boundary: the comma in "₹1,500", the
        # dot in "1.5 hours", the colon in "5:30pm". Splitting there made the voice
        # say "one" and then "five hundred" as separate phrases — and let the first
        # half of a fabricated price reach the caller before the guard saw the rest.
        # At the end of a partial buffer we can't tell yet, so we wait for more.
        # `final` = the "say" string is complete, so a trailing "…₹500." really is a
        # sentence end rather than a half-arrived number — otherwise the last sentence
        # would never stream and would only be spoken by the flush at the very end.
        if i > 0 and text[i - 1].isdigit():
            if text[i + 1].isdigit() if i + 1 < len(text) else not final:
                continue
        end = i
    if end < start:
        return "", start
    return text[start:end + 1].strip(), end + 1


def say_of(data) -> str:
    """The spoken text from a parsed reply, always as a string.

    A model in JSON mode can still emit `"say": 12345` or `"say": null`; calling
    .strip() on that crashed the turn. Never trust the type.
    """
    if not isinstance(data, dict):
        return ""
    v = data.get("say")
    if v is None or isinstance(v, (dict, list, bool)):
        return ""
    return str(v).strip()


def _parse_json(raw: str) -> dict:
    """Always returns a dict. `json.loads("null")` returns None and `"[]"` returns a
    list — both then blew up on .get(), so a malformed reply killed the call."""
    raw = raw if isinstance(raw, str) else ""
    candidates = [raw]
    m = re.search(r"\{.*\}", raw, re.S)      # JSON embedded in chatter
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            return data
    say, _ = extract_say(raw)
    return {"say": (say or raw.strip()[:400] or "Sorry, could you say that again?"),
            "action": {"type": "none"}}


class Session:
    """One phone call: conversation state + pack + providers."""

    def __init__(self, pack: dict, llm, tts=None, stt=None, guard_cfg=None,
                 speaker_gate=None):
        self.pack = pack
        self.llm = llm
        self.tts = tts
        self.stt = stt
        self.guard = guard_cfg if isinstance(guard_cfg, GuardConfig) else GuardConfig(guard_cfg)
        # Facts this call is allowed to state: every number in the pack, plus every
        # number the CALLER themselves has said (their phone, party size, the time they
        # asked for). Anything else in a reply is an invention — see guard.py.
        self._pack_numbers = pack_numbers(pack)
        self._caller_numbers: set = set()
        self.messages = [{"role": "system", "content": build_system_prompt(pack)}]
        # domain knowledge + shared common/basic knowledge (courtesy, small talk, identity)
        self.voice = pack.get("voice")        # per-pack natural voice; live-overridable
        self.clone = None                     # optional cloned "brand voice" (CloneTTS)
        self.indic = None                     # optional AI4Bharat Indic-Parler voice
        self.lang = "__cfg__"                 # STT/reply language: "__cfg__"=config, None=auto, "hi"/"en"…
        self.lang_lock = None                 # in AUTO mode: the language latched after confirmed detection (code)
        self._lang_cand = None                # pending latch candidate (needs corroboration)
        self._lang_cand_n = 0                 # how many consecutive turns agreed on the candidate
        self._last_det = None                 # (lang_code, prob) from the most recent transcription
        self.stt_fast = None                  # STT speed: None=config, True=greedy+no denoise, False=accurate
        self.stt_denoise = None               # DeepFilterNet: None=config, True/False forced from the UI
        self.model = None                     # LLM override (None = config default); e.g. "qwen2.5:3b" faster
        self.escalated = False
        self.mute_reason = None               # why a reply produced no audio (UI explains it)
        self._recent_says: list[str] = []      # our own last replies, to catch a stuck loop
        self.booking_started = False           # slots are only collected after the caller asks
        self._repeat_streak = 0                # consecutive identical replies
        self._last_atype = None                # action type of the previous turn
        self.slots: dict = {}                 # booking fields accumulated ACROSS turns (small
                                              #   models forget them, so the server remembers)
        self.last_asked_slot = None           # what the caller was last ASKED — looser
                                              #   than pending_slot, rescue-only
        self.pending_slot = None              # the field we last asked for out loud, so the
                                              #   caller's next short reply can answer it
        # Don't play "let me check…" twice running: on a slow box every reply trips the
        # threshold, and a filler before every answer reads as a stuck record, not thought.
        self.filled_this_turn = False
        self.filled_last_turn = False
        # Voice identity for THIS call: enrolled from the first utterance long enough to
        # take a print from, then used to ignore anyone else who speaks near the mic.
        self.speaker_gate = speaker_gate
        self.voiceprint = None
        self.diarizer = None                  # optional: split a turn by speaker first
        self.isolate_on = None                # None = follow config, True/False forced from the UI
        self._was_denoised = False            # last router verdict (gives it hysteresis)
        self._voiceprint_n = 0                # utterances folded into the print so far
        self._gate_rejects = 0                # consecutive refusals — see check_speaker
        self._last_rescue = None              # why the last refused turn was given
                                              #   back, for the inspector; consumed once
        self._reject_vec = None               # embedding of the last refused turn
        self._rival = None                    # a second voice heard while the print was
        self._rival_n = 0                     #   still a guess — see check_speaker
        self._gate_turns = 0                  # turns judged since the print was taken
        self._gate_proven = False             # has it EVER recognised this caller? see below
        self.bookings: list[int] = []

    # ---- greeting / escalation ---------------------------------------------
    def greeting(self) -> str:
        biz = self.pack.get("business", {}).get("name", "us")
        return self.pack.get("greeting") or f"Hello, thank you for calling {biz}. How can I help you?"

    # Words that mean "I want to book". Slots are only collected once the CALLER has
    # actually asked to book: otherwise the model fills name/phone/doctor with junk on
    # an ordinary question, those values stick for the rest of the call, and the badge
    # reads "needs: datetime" on turn one with a corrupted booking underneath.
    # Latin terms need WORD BOUNDARIES. Substring matching made "consultation fee" —
    # one of the commonest questions there is — look like a request to book: it set
    # booking_started, and every later rejected reply then came back as "may I have
    # the patient's full name?" to someone who had only asked a price. "consult" and
    # "slot" are gone entirely: neither expresses intent on its own.
    _BOOK_INTENT_RE = re.compile(
        r"\b(appointments?|book|booking|reserve|reservation|schedule|scheduling)\b")
    # Indic scripts are matched as substrings — they inflect with suffixes rather than
    # standing alone, so a boundary would miss the real forms.
    _BOOK_INTENT_NATIVE = (
        "अपॉइंटमेंट", "अपाइंटमेंट", "अपॉइनमेंट", "बुक", "बुकिंग", "मिलना", "दिखाना", "दिखाऊं", "समय ले",
        "అపాయింట్", "బుక్", "బుకింగ్", "కలవాలి", "చూపించ", "సమయం",
    )

    def _wants_to_book(self, text: str) -> bool:
        t = (text or "").lower()
        return bool(self._BOOK_INTENT_RE.search(t)) or any(k in t for k in self._BOOK_INTENT_NATIVE)

    def _plausible_slot(self, field: str, value) -> bool:
        """Reject a slot value the model clearly invented.

        A hallucinated phone number or a name like "मैं" quietly poisons the booking for
        the whole call, because slots are deliberately remembered across turns.
        """
        v = str(value).strip()
        if not v or len(v) > 80:
            return False
        low = v.lower()
        # Thinking noises and bare acknowledgements. Whisper transcribes them faithfully,
        # they are short, and they are not questions — so without this a caller who says
        # "hmm" while searching for a word gets it filed as their name.
        if low in _HESITATIONS:
            return False
        if low in {"none", "null", "n/a", "na", "unknown", "-", "?", "मैं", "आप", "नेను", "मुझे"}:
            return False
        if field == "phone":
            digits = [c for c in norm_digits(v) if c.isdigit()]
            return 10 <= len(digits) <= 13          # an Indian mobile, not an invented figure
        if field == "datetime":
            if len(v) < 3:
                return False
            # The model turns "कल सुबह 11 बजे" into "2024-05-29 11:00" — a calendar date
            # nobody said, whose digits are then ungrounded and block the confirmation.
            # Reject an invented YEAR; "Monday 4pm" is fine.
            return not any(len(n) == 4 and 1900 <= int(n) <= 2100 and n not in self._caller_numbers
                           for n in numbers_in(v) if n.isdigit())
        if field == "name":
            if not any(c.isalpha() for c in v):
                return False                        # "9876512345" is a phone, not a name
            # Observed live: "I need a medical fitness certificate" was filed as the
            # patient's name, and every later turn of the call was answered as if that
            # were who was calling. A request is not a name — nobody's name contains a
            # first-person pronoun or an interrogative, and real names are short.
            if _REQUEST_RE.search(low) or len(_words(v)) > 4:
                return False
        if field in ("name", "doctor", "service"):
            # the model sometimes echoes the business name or a whole sentence back
            biz = (self.pack.get("business", {}).get("name") or "").lower()
            return not (biz and biz in low) and len(_words(v)) <= 8
        return True

    def slots_from_caller(self, text: str) -> dict:
        """Slot values read straight out of the caller's own words.

        The model kept failing to register a doctor the caller had named three times, so
        the call went round in circles. The pack lists how each doctor is actually said —
        in every script — and we match that ourselves. A value the caller spoke aloud is
        better evidence than anything the model infers.
        """
        found = {}
        t = " " + (text or "").lower() + " "
        for field, table in (self.pack.get("booking", {}).get("slot_aliases", {}) or {}).items():
            for canonical, aliases in (table or {}).items():
                if any(str(a).lower() in t for a in aliases):
                    found[field] = canonical
                    break
        # A phone number the caller read out. Without this it is only ever captured when
        # the MODEL files it — so a turn whose reply the guard replaced lost the number
        # entirely, and we asked for it again right after they had given it.
        # Deliberately strict: exactly ONE 10-13 digit run in the turn. Times, prices and
        # dates are far shorter, so this cannot mistake them for a number.
        phone_field = next((f for f in self.pack.get("booking", {}).get("required", [])
                            if str(f).lower() in ("phone", "mobile", "contact", "number")), None)
        if phone_field:
            digits = [n for n in numbers_in(text or "") if n.isdigit() and 10 <= len(n) <= 13]
            if len(digits) == 1:
                found[phone_field] = digits[0]
        return found

    # Interrogatives across the languages the packs support. A caller asking a question
    # is not answering our slot question, however short their sentence is.
    _QUESTION_MARKERS = frozenset("""
    क्या कौन कौनसा कौनसी कितना कितनी कैसे कहाँ क्यों
    ఏమిటి ఎవరు ఎంత ఎలా ఎక్కడ ఎందుకు ఏది
    என்ன யார் எவ்வளவு எப்படி எங்கே ஏன்
    ಏನು ಯಾರು ಎಷ್ಟು ಹೇಗೆ ಎಲ್ಲಿ ಏಕೆ
    എന്ത് ആര് എത്ര എങ്ങനെ എവിടെ എന്തുകൊണ്ട്
    কি কে কত কিভাবে কোথায় কেন
    काय कोण किती कसे कुठे का
    શું કોણ કેટલું કેવી ક્યાં કેમ
    ਕੀ ਕੌਣ ਕਿੰਨਾ ਕਿਵੇਂ ਕਿੱਥੇ ਕਿਉਂ
    کیا کون کتنا کیسے کہاں کیوں
    କଣ କିଏ କେତେ କିପରି କେଉଁଠି କାହିଁକି
    what who whom whose how when where why which
    """.split())
    # English inverts for questions, so an auxiliary only signals one when it LEADS the
    # sentence. Matching them anywhere rejected "my name is Manu Mishra" as a question
    # and lost the name — "is", "are" and "do" are everywhere in ordinary statements.
    _QUESTION_LEAD = frozenset("is are am was were do does did can could shall should "
                               "will would may might have has had".split())

    def _answer_to_pending(self, text: str):
        """Take the caller's short reply as the answer to the slot we just asked for.

        Without this the call loops: we ask "which day and time?", the caller says
        "दस बजे सुबह", the model fails to file it, and we ask again. The caller's own
        wording is the truth for a free-text field — the model turning it into an ISO
        date is what the grounding guard then rejects.

        Deliberately narrow: only the field we asked for out loud, only a short reply,
        and never a question. `phone` is excluded because it has real validation in
        _plausible_slot and a misheard number is worse than no number.
        """
        field = self.pending_slot
        if not field or field == "phone" or self.slots.get(field):
            return
        # The turn answered a DIFFERENT slot. Asked for a name and told "9876512345",
        # this used to file the phone number AS the name — it is short, not a question,
        # and passes the plausibility check. Anything the caller's own words already
        # resolved elsewhere is not an answer to what we asked.
        from_caller = self.slots_from_caller(text)
        if from_caller and field not in from_caller:
            return
        words = _words(text or "")
        if not words or len(words) > 10:
            return
        low = [w.lower() for w in words]
        if (text or "").strip().endswith("?") or (set(low) & self._QUESTION_MARKERS) \
                or low[0] in self._QUESTION_LEAD:
            return
        value = (text or "").strip()[:60]
        if field == "name":
            value = _strip_name_frame(value)
        if self._plausible_slot(field, value):
            self.slots[field] = value
            self.pending_slot = None
            log.info("guard: filed the caller's own wording for %s: %r", field, value)

    def _asks_for_unoffered(self, text: str) -> bool:
        """The caller asked for a service this business does not provide.

        The grounding guard cannot catch this on its own: asked for a neurologist, the
        model answered "consultation fee ₹500" — and ₹500 IS a real pack number (the GP
        fee), so every number checked out. The claim was false, not the figures. The pack
        lists what it does NOT do, and we answer that deterministically.
        """
        t = " " + (text or "").lower() + " "
        return any(k.lower() in t for k in self.pack.get("not_offered", []))

    def _keyword_escalation(self, text: str) -> bool:
        kws = [k.lower() for k in self.pack.get("escalation", {}).get("keywords", [])]
        t = text.lower()
        return any(k in t for k in kws)

    # ---- language resolution -----------------------------------------------
    def _effective_lang(self):
        """The pinned STT/reply language, or None for AUTO mode. '__cfg__' follows the
        config default (which may itself be None = auto)."""
        if self.lang == "__cfg__":
            return getattr(self.stt, "language", None)
        return self.lang

    def reply_language(self, last_user: str = ""):
        """The ONE language the agent must reply in this turn.
        Pinned language wins; in auto mode the latched language wins; before anything is
        latched, prefer a CONFIDENT Whisper detection over raw script guessing (so e.g.
        Marathi isn't mistaken for Hindi from the Devanagari script alone)."""
        eff = self._effective_lang()
        if eff:
            return LANG_NAMES.get(eff)
        if self.lang_lock:
            locked = LANG_NAMES.get(self.lang_lock)
            # The lock exists to stop per-clip flip-flopping from unreliable detection.
            # But a transcript written in a DIFFERENT SCRIPT is far stronger evidence than
            # a latch from earlier audio: an English call that switches to Hindi was still
            # being answered in English. Only a genuine script change overrides — Marathi
            # and Hindi share Devanagari, so the (more precise) lock keeps those, and
            # Latin text keeps it too because that is Hinglish, not a switch to English.
            cur = dominant_script_lang(last_user)
            if cur and locked and LANG_SCRIPT.get(cur) != LANG_SCRIPT.get(locked):
                log.info("language: caller switched script %s -> %s; following the caller",
                         locked, cur)
                return cur
            return locked
        if self._lang_cand:                       # keep the reply language STABLE while the
            return LANG_NAMES.get(self._lang_cand)  # latch corroborates (no per-clip flipping)
        if self._last_det and self._last_det[1] >= 0.6:
            nm = LANG_NAMES.get(self._last_det[0])
            if nm:
                return nm
        # typed Hinglish has no script and no Whisper detection to go on
        return (dominant_script_lang(last_user)
                or ("Hindi" if looks_hinglish(last_user) else None)
                # A digits-only turn ("8920429057") or a bare "ok" carries no script at
                # all, and falling through to English here switched the reply language
                # MID-CALL — a Hindi caller reading out their phone number got an
                # English question back. The conversation is the evidence this turn lacks.
                or self._recent_user_language())

    def _recent_user_language(self):
        """The language of the caller's most recent turn that actually had a script."""
        for m in reversed(self.messages):
            if m.get("role") != "user":
                continue
            lang = dominant_script_lang(m.get("content", ""))
            if lang:
                return lang
        return None

    # How many native-language facts to put in front of the model per turn. Small on
    # purpose: Telugu costs ~6× the tokens of English, so pasting the whole knowledge
    # base overflowed num_ctx and pushed out the conversation — the model then refused
    # questions it could answer and mangled prices it was copying ("₹700" → "₹70:00").
    NATIVE_FACTS_PER_TURN = 4

    def _relevant(self, entries: list, query: str, k: int) -> list:
        """The k entries whose own words best overlap the caller's question.

        Retrieval used to be impossible here — English tags against a Telugu query match
        nothing. Now that the facts are stored in the caller's language, plain word
        overlap works, in that language, with no extra model.
        """
        q = {w for w in _words(query) if len(w) >= 3}
        if not q:
            return entries[:k]
        scored = []
        for e in entries:
            words = {w for w in _words(f"{e[0]} {e[1]}") if len(w) >= 3}
            hits = len(q & words)
            # prefix credit: Indic words inflect heavily, so "సమయం"/"సమయంలో" should match
            partial = sum(1 for a in q for b in words if a != b and (a[:4] == b[:4]))
            scored.append((hits * 2 + partial, e))
        scored.sort(key=lambda x: -x[0])
        return [e for score, e in scored[:k] if score > 0]

    def native_facts(self, lang_name: str, query: str = "") -> str:
        """The pack's knowledge already written in the caller's language.

        A knowledge entry may carry `a_hi` / `a_te` / … beside its English `a`. When the
        caller speaks that language we hand the model the ready-made sentence, so it
        QUOTES instead of translating. Translating 24 English facts while composing
        Telugu is exactly where a small model gives up — it answers the easy question
        (the address) and drops the one that was asked, or echoes the question back.
        """
        code = NAME_TO_CODE.get(lang_name or "")
        if not code or code == "en":
            return ""
        key = f"a_{code}"
        pack = self.pack
        kb = (pack.get("knowledge") or []) + (pack.get("common") or [])
        kkey = f"k_{code}"        # native keywords: what a caller actually says
        entries = [(f"{it.get(kkey, '')} {it.get('q', '')}", it[key]) for it in kb if it.get(key)]
        entries += [(s.get("name", ""), s[key]) for s in (pack.get("services") or []) if s.get(key)]
        picked = self._relevant(entries, query, self.NATIVE_FACTS_PER_TURN)
        biz = pack.get("business", {})
        head = [f"- {biz[f'{k}_{code}']}" for k in ("name", "hours", "address")
                if biz.get(f"{k}_{code}")]
        lines = [f"- {a}" for _q, a in picked]
        if not lines:
            return ""                       # nothing relevant → don't bloat the prompt
        return "\n".join(head[:1] + lines)

    def last_user_text(self) -> str:
        return next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")

    def reply_style(self, last_user: str = ""):
        """(language name, is_romanised) for this turn — the single source of truth for
        the forced-language instruction, the reply token budget, and the guard."""
        name = self.reply_language(last_user)
        # romanised = Latin letters with no native script → Hinglish and friends
        roman = bool(re.search(r"[A-Za-z]", last_user)) and dominant_script_lang(last_user) is None
        return name, roman

    # Measured on the clinic pack: the system prompt alone is ~4,000 tokens, the
    # native-language facts add ~1,000 in Telugu, and the conversation adds more — past
    # the old 6,144. On overflow Ollama silently drops the OLDEST messages, so the agent
    # loses its own instructions mid-call and starts looping.
    MIN_CONTEXT = 12288

    def context_budget(self, last_user: str = ""):
        """num_ctx for this call.

        Deliberately CONSTANT — it must not vary by language. Ollama reallocates its KV
        cache whenever num_ctx changes, which reloads the model: sizing it per turn
        pushed every reply from ~3s to ~16s.
        """
        base = getattr(self.llm, "num_ctx", None)
        if not base:
            return None
        return max(base, self.MIN_CONTEXT)

    def reply_budget(self, last_user: str = ""):
        """num_predict for this turn, or None to use the model default.

        The same sentence costs 3.5× the tokens in Hindi and 6.2× in Telugu, so a budget
        tuned for English guillotines native-script replies mid-word — which the caller
        hears as garbled speech. Scale it by the language actually being spoken
        (guard.TOKEN_FACTOR holds the measured ratios).
        """
        base = getattr(self.llm, "num_predict", None) or getattr(self.llm, "max_tokens", None)
        if not base:
            return None
        name, roman = self.reply_style(last_user)
        # only NATIVE scripts need the bigger budget — romanised Hinglish tokenises
        # much like English.
        if name and not roman and LANG_SCRIPT.get(name):
            return int(base * TOKEN_FACTOR.get(name, 4))
        return None

    # ---- knowledge injection ------------------------------------------------
    def call_messages(self) -> list:
        """Messages for the next LLM call: base system prompt + (optional) a forced-
        language instruction + per-turn retrieved knowledge. Extras aren't persisted."""
        extras = []
        last_user = self.last_user_text()
        name, romanized = self.reply_style(last_user)
        if name and name != "English" and romanized:
            extras.append({"role": "system",
                           "content": f"The caller is writing {name} in Latin/Roman letters (Hinglish-style). "
                                      f"Reply in the SAME romanised Latin style — do NOT switch to {name}'s "
                                      f"native script. One language only; keep numbers, days, prices, names exact."})
        elif name:
            extras.append({"role": "system",
                           "content": f"The caller is speaking {name}. Reply ENTIRELY in {name} using its "
                                      f"native script. Use ONE language only — do NOT mix in English or any "
                                      f"other language, and do NOT switch languages between turns. "
                                      f"Translate your wording into {name}, but keep every number, time, day, "
                                      f"price, and name exactly as given (₹500 stays ₹500; the days/hours stay "
                                      f"the same, just written in {name}). Translating does NOT license "
                                      f"inventing: if a fact is not written in the facts above, set "
                                      f'"kind":"unknown" rather than producing a {name} sentence that '
                                      f"contains a number you made up."})
        # Facts already written in the caller's language, appended AFTER the stable
        # prefix so Ollama still reuses its KV cache. The model quotes these instead of
        # translating the English knowledge base mid-sentence.
        if name and name != "English" and not romanized:
            facts = self.native_facts(name, last_user)
            if facts:
                extras.append({"role": "system",
                               "content": f"FACTS IN {name} — the caller's question is most likely "
                                          f"answered below. Use these sentences AS WRITTEN (you may "
                                          f"shorten them); do NOT translate them back to English and do "
                                          f"NOT rewrite the numbers, names or times:\n" + facts})
        if not extras:
            return self.messages
        # Append extras at the END (not after the system prompt) so the stable
        # [system + history] prefix stays byte-identical across turns → Ollama
        # reuses its KV cache and only re-processes the small trailing block.
        return self.messages + extras

    # ---- turn primitives ----------------------------------------------------
    def begin_user(self, user_text: str):
        # Guard against a duplicate consecutive user turn: if streaming fails after
        # begin_user() already ran, the non-streaming fallback calls this again with
        # the same text — appending it twice would corrupt the conversation history.
        if not (self.messages and self.messages[-1]["role"] == "user"
                and self.messages[-1]["content"] == user_text):
            self._append("user", user_text)
        # a number the caller gave us is a fact we're allowed to repeat back
        self._caller_numbers |= numbers_in(user_text)
        # People say numbers as WORDS on the phone — "my baby is six months old",
        # "ten in the morning". The guard only collected digits, so a reply that
        # answered them numerically ("at 6 months") was rejected as ungrounded and the
        # caller got a slot question instead of their answer. Spoken forms count as the
        # caller having said the number, because they did.
        self._caller_numbers |= spoken_numbers(user_text)
        if self._keyword_escalation(user_text):
            self.escalated = True
            msg = self.pack.get("escalation", {}).get(
                "message", "Let me connect you to a team member right away.")
            self._append("assistant",
                         json.dumps({"say": msg, "action": {"type": "escalate"}}))
            return {"say": msg, "action": {"type": "escalate"}, "escalated": True}
        if self._asks_for_unoffered(user_text):
            say = self.safe_say("scope")          # already in the caller's language
            self._append("assistant",
                         json.dumps({"say": say, "action": {"type": "none"}},
                                    ensure_ascii=False))
            log.info("deterministic: caller asked for a service the pack does not offer")
            return {"say": say, "action": {"type": "none"}, "escalated": self.escalated}
        return None

    def note_interrupted(self):
        """Called when a turn is cancelled by barge-in. Keep history paired so the
        model isn't left with a dangling user message."""
        if self.messages and self.messages[-1]["role"] == "user":
            self._append("assistant",
                         json.dumps({"say": "", "action": {"type": "none"}}))

    # keep the prompt bounded on long calls: system prompt + the last MAX_TURNS pairs.
    MAX_TURNS = 16

    def _trim_history(self):
        max_msgs = 1 + self.MAX_TURNS * 2      # system + N (user,assistant) pairs
        if len(self.messages) > max_msgs:
            self.messages = [self.messages[0]] + self.messages[-(max_msgs - 1):]

    def _append(self, role: str, content):
        """Append to the conversation and keep it bounded.

        Trimming used to happen ONLY in finalize(), so every path that appends without
        finalizing — escalations, services we don't offer, barge-ins, the go-on nudge —
        grew self.messages forever. On a long call the prompt eventually overflows
        num_ctx, Ollama silently drops the OLDEST messages (the system prompt with all
        the agent's instructions), and the agent starts looping mid-call.
        """
        self.messages.append({"role": role, "content": content})
        self._trim_history()

    def allowed_numbers(self) -> set:
        """Every number this call may legitimately state: the pack's facts, whatever the
        caller told us, and the booking references we ourselves issued."""
        return set(self._pack_numbers) | self._caller_numbers | {str(b) for b in self.bookings}

    def allowed_caller_numbers(self) -> set:
        """Only what the CALLER told us — the values an echo would claim as its own."""
        return set(self._caller_numbers)

    def safe_say(self, kind: str = "unknown") -> str:
        """The pre-written reply for this call's language, used whenever the model's own
        words must be replaced. Accepts either a model `kind` ("out_of_scope") or a
        template name ("scope"); anything unrecognised falls back to "unknown"."""
        lang, roman = self.reply_style(self.last_user_text())
        k = (kind or "").strip().lower()
        return safe_line(_KIND_TO_SAFE.get(k, k), lang, self.pack, roman)

    def booking_ref(self, bid) -> str:
        """"Your booking reference is #12." — in the caller's own language."""
        lang, roman = self.reply_style(self.last_user_text())
        return safe_line("ref", lang, self.pack, roman, ref=bid)

    # ---- booking-aware recovery --------------------------------------------
    def missing_slots(self) -> list:
        """Required booking fields still unknown, counting what the caller just said.

        Called mid-stream, BEFORE finalize has folded this turn's answer in, so the
        caller's current words are read here too — otherwise we'd ask for the very
        thing they are in the middle of telling us.
        """
        required = self.pack.get("booking", {}).get("required", [])
        known = dict(self.slots)
        known.update(self.slots_from_caller(self.last_user_text()))
        return [k for k in required if not known.get(k)]

    def collecting(self) -> bool:
        """A booking is under way and something is still missing."""
        return bool(self.booking_started and self.missing_slots())

    def slot_question(self, field) -> str:
        """Ask for one booking field, in the caller's language.

        The pack's own question is English-only, which is why a Hindi caller used to be
        asked in English (or, worse, the model improvised). A pack can override per
        language with `booking.slots.<field>_hi`; otherwise the pre-written line in
        guard.ASK_LINES is used, and English packs keep their exact wording.
        """
        if not field:
            return ""
        slot_q = self.pack.get("booking", {}).get("slots", {}) or {}
        lang, roman = self.reply_style(self.last_user_text())
        code = NAME_TO_CODE.get(lang or "")
        if code and code != "en":
            # A Hinglish caller is speaking Hindi, just in Latin letters — the pack's
            # English question is the wrong language for them too, so ask_line wins
            # for romanised calls as well as native-script ones.
            native = slot_q.get(f"{field}_{code}") if not roman else None
            return native or ask_line(field, lang, roman)
        return slot_q.get(field) or ask_line(field, lang, roman)

    def confirm_question(self) -> str:
        """"I have everything — shall I book it?", in the caller's language.

        This used to be an English sentence, so on a Hindi call the language check threw
        it away and the caller heard "sorry, I didn't catch that" at the exact moment
        every detail had in fact been collected.
        """
        lang, roman = self.reply_style(self.last_user_text())
        return ask_line("confirm", lang, roman)

    def asks_for_known_slot(self, say: str):
        """The field this reply is asking for, if we already have it.

        The pack's slot questions are in the system prompt, so the model reproduces them
        verbatim — including for details it has already been given ("what mobile number
        should we use?" right after the caller read one out). Matching against the pack's
        own wording is exact enough to be safe; a freely-worded question is left alone.
        """
        n = " ".join(_words(say or ""))
        if len(n) < 10:
            return None
        slot_q = self.pack.get("booking", {}).get("slots", {}) or {}
        for field, value in self.slots.items():
            if not value:
                continue
            for variant in (slot_q.get(field), self.slot_question(field)):
                if variant and " ".join(_words(variant)) == n:
                    return field
        return None

    def asked_which_slot(self, say: str):
        """Which slot this line asks for — whether or not we already have it.

        The sibling of `asks_for_known_slot`, which only looks at slots already filled
        because its job is spotting a STALE re-ask. This one answers a different
        question: what did the caller just get asked? Used only by the expectation
        rescue, which fails safe, so a miss costs nothing and a wrong guess cannot
        refuse anybody.
        """
        said = [w for w in _words(say or "") if len(w) >= 3]
        if len(" ".join(said)) < 10:
            return None
        booking = self.pack.get("booking", {}) or {}
        slot_q = booking.get("slots", {}) or {}
        fields = booking.get("required", []) or []

        # Exact first. The pack's slot questions are in the system prompt, so the model
        # reproduces them verbatim most of the time and this is unambiguous.
        n = " ".join(_words(say or ""))
        for field in fields:
            for variant in (slot_q.get(field), self.slot_question(field)):
                if variant and " ".join(_words(variant)) == n:
                    return field

        # Then tolerantly, because the model does sometimes word its own question. The
        # vocabulary comes from the PACK's phrasing, so this works for any pack and any
        # language without a hand-written keyword list. Requires most of the question's
        # distinctive words AND a clear winner: two fields tying means we cannot tell,
        # and the rescue is better off with no answer than a wrong one.
        said_set = set(said)
        # Score each FIELD once. Scoring per variant made a field its own runner-up —
        # `slot_q[field]` and `slot_question(field)` are usually the same string — so
        # the tie-break could never be satisfied and this branch never fired.
        per_field = {}
        for field in fields:
            variants = {v for v in (slot_q.get(field), self.slot_question(field)) if v}
            scores = []
            for variant in variants:
                key = {w for w in _words(variant) if len(w) >= 3}
                if len(key) >= 2:
                    scores.append(len(key & said_set) / len(key))
            if scores:
                per_field[field] = max(scores)
        if not per_field:
            return None
        ranked = sorted(per_field.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        # A clear winner only. Two fields scoring alike means we cannot tell which was
        # asked, and the rescue is better off with no answer than a wrong one.
        if best_score >= 0.5 and best_score > runner_up:
            return best
        return None

    def recovery_line(self, kind: str = "") -> str:
        """What to say when a reply has been rejected.

        Mid-booking the honest, useful answer is NOT "I'll check with the desk" — the
        caller is answering our own questions. On a real call, "दस बजे सुबह" (a direct
        answer to "which day and time?") came back as "I don't have that information",
        which abandoned the booking entirely. Ask for what's still missing instead.

        ...unless they asked us something. A booking in progress does not make every
        later turn a slot answer, and the substitution turned a genuine question into a
        non-sequitur: asked "do you use ammonia free colour?" mid-booking, the agent
        replied "May I have your name?" — the caller's question simply vanished. When
        they ASKED, say honestly that we do not know AND keep the booking alive by
        adding the next slot question to it.
        """
        if self.booking_started and self._asked_us_something():
            refusal = self.safe_say(kind or "unknown")
            missing = self.missing_slots()
            if missing:
                q = self.slot_question(missing[0])
                if q:
                    self.pending_slot = missing[0]
                    log.info("guard: caller asked something we cannot answer mid-booking "
                             "— refusing honestly, then asking for %s", missing[0])
                    return f"{refusal} {q}"
            return refusal
        if self.booking_started:
            missing = self.missing_slots()
            if missing:
                q = self.slot_question(missing[0])
                if q:
                    self.pending_slot = missing[0]
                    log.info("guard: mid-booking recovery — asking for %s instead of refusing",
                             missing[0])
                    return q
            elif self.slots:
                # Every detail is in hand. The caller has just finished answering our
                # questions — they are owed a confirmation, not "sorry, I didn't catch
                # that", which is what they got the moment the last slot landed.
                log.info("guard: all slots collected — confirming instead of refusing")
                return self.confirm_question()
        return self.safe_say(kind or "unknown")

    def _asked_us_something(self) -> bool:
        """Did the caller's last turn pose a question, rather than answer ours?

        A booking in progress is not a licence to treat everything they say as a slot
        answer. `_words` is used rather than a regex because the backslash-w class drops
        Indic combining marks and destroys the word entirely.
        """
        from .guard import _is_question
        last = self.last_user_text()
        return bool(last) and _is_question(last, set(_words(last.lower())))

    def verify(self, say: str, kind: str = "", *, booking_turn: bool = False,
               truncated: bool = False) -> str:
        """Last gate before the caller hears anything.

        Two layers. First, the model's own classification: a turn it flagged as
        "unknown"/"out_of_scope"/"unclear" is answered with OUR pre-written sentence in
        the caller's language, so a weak model never has to compose a refusal in Telugu.
        Second — for everything else, including turns where the model claimed "answer" —
        the grounding check in guard.py, which catches the invented ₹1,000 the model was
        confident about.
        """
        # Only a turn that actually booked or escalated is exempt: those speak the
        # caller's own details or the pack's own escalation wording. Everything else
        # honours the model's verdict — a small model will happily flag "unknown" and
        # then attach a stray "collect" action, and answering "may I have your name?"
        # to "do you do dialysis?" is precisely the failure we're removing.
        if (kind or "").strip().lower() in _KIND_TO_SAFE and not booking_turn:
            if say:
                log.info("guard: model self-reported '%s' — replacing its own wording", kind)
            return self.recovery_line(kind)
        # Answering AS the caller instead of TO them ("मेरा नाम मिश्रा है" back at someone
        # who just gave their name). It reads as a reply, so nothing else catches it — and
        # it corrupted a real booking, which stored the surname alone.
        if say and self.guard.enabled and not booking_turn \
                and looks_like_echo(say, self.last_user_text(), self._caller_numbers):
            log.warning("guard: reply echoes the caller in their own voice — replacing: %s",
                        say[:100])
            return self.recovery_line("unclear")
        # Parroting an earlier reply. Checked here as well as in retry_if_repeating so the
        # STREAMING path (which never calls that) still ends up consistent: what finalize
        # records must equal what was actually spoken.
        if say and self._is_repeat(say):
            log.warning("guard: reply repeats an earlier turn — replacing")
            return self.recovery_line("repeat")
        lang, roman = self.reply_style(self.last_user_text())
        checked, reason = check_reply(say, pack=self.pack, allowed_numbers=self.allowed_numbers(),
                                      lang_name=lang, romanized=roman, gcfg=self.guard,
                                      check_language=not booking_turn, truncated=truncated)
        if reason:
            log.warning("guard: %s | dropped: %s", reason, say[:120])
            # A reply thrown out during a booking gets the same treatment as one the
            # model itself flagged: check_reply's own replacement is the generic refusal,
            # which walks away from the collection.
            # Gated on booking_started, NOT collecting(): the caller's answer is filed
            # BEFORE verify runs, so the turn that completes the last slot has nothing
            # missing — and the caller heard "I'll check with the desk" at the exact
            # moment every detail was in hand. recovery_line picks slot-question vs
            # confirm vs refusal itself.
            # ("truncated" only trims the model's words — those are still its reply.)
            if reason != "truncated" and self.booking_started:
                return self.recovery_line("unknown")
        return checked

    def finalize(self, raw: str, meta: dict | None = None, spoken: str | None = None) -> dict:
        """`meta` carries the model's finish_reason, so the guard can tell a reply that
        genuinely ran out of tokens from one that merely lacks a full stop.

        `spoken` is what the caller ACTUALLY heard, when the streaming guard blocked the
        model's reply and substituted a safe line. Without it the guard's own verdict was
        discarded at the last step: the caller heard "may I have your name?" while the UI
        transcript, the conversation history and the `turns` table all recorded the
        rejected sentence — the one thing the guard exists to make sure nobody ever sees.
        The booking side-effects below still run on the model's `action`; only the WORDS
        are replaced, because only the words were.
        """
        data = _parse_json(raw)
        action = data.get("action", {}) or {}
        say = say_of(data)
        kind = data.get("kind") or ""
        atype = action.get("type")
        required = self.pack.get("booking", {}).get("required", [])
        slot_q = self.pack.get("booking", {}).get("slots", {})

        # --- accumulate booking slots ACROSS turns (the model often forgets earlier
        #     answers; the server keeps them so collection never restarts). Slot keys are
        #     normalised to the pack's required field names, so a model that files "time"
        #     for the required "datetime" still satisfies the check (no infinite re-ask). ---
        #     Two gates, both learned from a real call: the caller must actually have
        #     asked to book (or already be booking), and each value must be plausible —
        #     otherwise "needs: datetime" showed up on turn one over invented details.
        last_user = self.last_user_text()
        # An explicit book_appointment IS the intent, so a model that books still collects.
        if self.booking_started or atype == "book_appointment" or self._wants_to_book(last_user):
            self.booking_started = True
            # what the caller SAID wins over what the model inferred
            self.slots.update(self.slots_from_caller(last_user))
            self._answer_to_pending(last_user)
            if atype in ("book_appointment", "collect") and isinstance(action.get("slots"), dict):
                alias = _slot_alias_map(required)
                for k, v in action["slots"].items():
                    if v in (None, "", []):
                        continue
                    field = alias.get(str(k).lower(), k)
                    if self._plausible_slot(field, v):
                        self.slots[field] = v
                    elif (field == "datetime" and last_user
                          and not self.slots.get(field)
                          and self.pending_slot == "datetime"):
                        # Free-text field: when the model turns "कल सुबह 11 बजे" into an
                        # invented ISO date, the caller's own phrasing is the truth.
                        # But ONLY when we actually asked for the time and don't have one:
                        # unconditionally it stored whatever the caller last said — a
                        # reproduced case wrote "मेरा नाम मनु मिश्रा है" ("my name is Manu
                        # Mishra") into the appointment time, and straight on into SQLite.
                        self.slots[field] = last_user[:60]
                        log.info("guard: kept the caller's wording for datetime, not %r", v)
                    else:
                        log.info("guard: dropped implausible slot %s=%r", field, v)
        if atype in ("book_appointment", "collect"):
            action["slots"] = dict(self.slots)

        if atype == "book_appointment":
            missing = [k for k in required if not self.slots.get(k)]
            if not missing:                       # everything gathered → actually book
                bid = create_booking(self.pack["id"], dict(self.slots))
                self.bookings.append(bid)
                action["booking_id"] = bid
                # The reference is appended AFTER verification (below), for two reasons:
                # it is our sentence rather than the model's, and an English "your booking
                # reference is #12" tacked onto a Hindi reply used to read as a language
                # switch and get the whole confirmation thrown away.
                self.slots = {}                   # reset for any next booking on the call
                # …and close the collection. Leaving it open made every later rejected
                # reply come back as "may I have your name?" long after the booking was
                # done; a genuine second booking re-opens it via _wants_to_book.
                self.booking_started = False
                self.pending_slot = None
            else:                                 # model claimed booked but info is incomplete →
                action["type"] = "collect"        # do NOT falsely confirm; ask for what's missing
                action["missing"] = missing
                say = self.slot_question(missing[0])
                self.pending_slot = missing[0]    # our own question → we know what we asked
        elif atype == "collect":
            missing = [k for k in required if not self.slots.get(k)]
            action["missing"] = missing
            if not say:                           # model gave a collect action but no spoken line
                if missing:
                    say = self.slot_question(missing[0])
                    self.pending_slot = missing[0]
                else:
                    say = self.confirm_question()
            else:
                stale = self.asks_for_known_slot(say)
                if stale and missing:
                    log.info("guard: model re-asked for %s, which we already have — "
                             "asking for %s instead", stale, missing[0])
                    say = self.slot_question(missing[0])
                    self.pending_slot = missing[0]
                elif len(missing) == 1:
                    # The model wrote its own question. We only know WHICH field it means
                    # when exactly one is left — guessing with more outstanding would file
                    # the caller's next answer against the wrong slot.
                    self.pending_slot = missing[0]
                else:
                    self.pending_slot = None

        if atype == "escalate":
            self.escalated = True

        # verify before anything is persisted or spoken
        booked = bool(action.get("booking_id"))
        say = self.verify(say, kind, booking_turn=booked or atype == "escalate",
                          truncated=(meta or {}).get("finish_reason") == "length")
        if not say and not booked:
            lang, roman = self.reply_style(self.last_user_text())
            say = safe_line("repeat", lang, self.pack, roman)
        if booked and f"#{action['booking_id']}" not in say:
            say = (say.rstrip(". ") + ". " if say else "") + self.booking_ref(action["booking_id"])

        # What did the caller just get asked? `pending_slot` cannot answer this: it is
        # deliberately left None whenever more than one field is outstanding, because
        # filing an answer against the wrong slot is worse than not filing it. That is
        # correct for COLLECTION and useless for the expectation rescue, which only
        # needs to know what question the caller is replying to — and a booking spends
        # most of its turns with several fields missing, so the rescue would almost
        # never have fired.
        #
        # Derived from the FINAL spoken line, after verify(), because that is the
        # sentence the caller actually heard and is answering.
        self.last_asked_slot = self.pending_slot or self.asked_which_slot(say)

        self._remember_say(say)                      # so the next turn can spot a loop
        self._last_atype = action.get("type")        # was the previous turn a slot request?
        data["say"], data["action"] = say, action    # persist a coherent turn to history
        if spoken is not None and spoken.strip():
            # Record what was SAID, not what was rejected — in the history the model sees
            # next turn, in the transcript the UI shows, and in the row the DB stores.
            say = spoken
            data = dict(data)
            data["say"] = spoken
        self._append("assistant", json.dumps(data, ensure_ascii=False))
        self._trim_history()
        return {"say": say, "action": action, "escalated": self.escalated}

    # ---- blocking turn (CLI) -----------------------------------------------
    def handle_text(self, user_text: str) -> dict:
        sc = self.begin_user(user_text)
        if sc:
            return sc
        meta: dict = {}
        raw = self.llm.chat(self.call_messages(), force_json=True, model=self.model,
                            num_predict=self.reply_budget(user_text), meta=meta,
                            num_ctx=self.context_budget(user_text))
        raw = self.retry_if_looped(raw, meta)        # looping inside one reply
        raw = self.retry_if_repeating(raw, meta)     # repeating the previous reply
        return self.finalize(raw, meta)

    # ---- second chance ------------------------------------------------------
    def retry_short(self, meta: dict | None = None) -> str:
        """One more attempt with the reins pulled in.

        A model that rambles in Telugu will usually get it right when told to answer in
        a single short sentence — long-form generation is where a small model runs dry
        and starts repeating. Used only for QUALITY failures: an invented fact is not
        worth a second try, but a real answer buried under a loop is.
        """
        msgs = self.call_messages() + [{
            "role": "system",
            "content": "Your previous attempt was rejected because it repeated itself. Answer "
                       "again in ONE short, clear sentence, entirely in the caller's language, "
                       "using ONLY the facts written above. Do not repeat any phrase. If the "
                       'fact is not written above, set "kind":"unknown" and leave "say" empty.'}]
        return self.llm.chat(msgs, force_json=True, model=self.model,
                             num_predict=self.reply_budget(self.last_user_text()), meta=meta,
                             num_ctx=self.context_budget(self.last_user_text()))

    # How many of our own recent replies to remember when checking for a stuck loop.
    REPEAT_MEMORY = 3

    def _is_repeat(self, say: str) -> bool:
        """True when the model is about to say what it just said.

        Observed on a real call: the caller reported stomach pain, then asked to book an
        appointment, then gave their name — and every single turn came back
        "मैं अच्छा हूँ, कैसे मैं आपकी मदद कर सकता हूँ?". Once a phrase is in the history a
        small model will happily copy it forever. is_degenerate() only looks INSIDE one
        reply, so nothing was watching across turns.
        """
        cur = set(_words(say))
        if len(cur) < 4:
            return False                      # "हाँ, ज़रूर" may legitimately recur
        # Mid-booking, asking again for a detail the caller still hasn't given is normal
        # receptionist behaviour — allow it once before calling it a loop.
        if (self.booking_started and self._last_atype == "collect" and self._repeat_streak < 1
                and any(not self.slots.get(k)
                        for k in self.pack.get("booking", {}).get("required", []))):
            return False
        norm = " ".join(_words(say))
        for prev in self._recent_says:
            if norm == prev:
                return True
            pw = set(prev.split())
            if pw and len(cur & pw) / len(cur | pw) >= 0.9:      # near-identical
                return True
        return False

    def looks_like_repeat_start(self, partial: str) -> bool:
        """True while a STREAMING reply is forming and already matches the opening of a
        reply we gave earlier.

        The streaming path speaks each sentence as it is generated, so waiting for the
        whole reply to spot a loop is too late — the caller has already heard it. A
        repeat is emitted verbatim, so a prefix match is enough to stop it in time.
        """
        n = " ".join(_words(partial))
        if len(n) < 15:
            return False
        return any(prev.startswith(n) for prev in self._recent_says)

    def _remember_say(self, say: str):
        norm = " ".join(_words(say))
        self._repeat_streak = self._repeat_streak + 1 if (
            self._recent_says and norm == self._recent_says[-1]) else 0
        self._recent_says.append(norm)
        del self._recent_says[:-self.REPEAT_MEMORY]

    def retry_if_repeating(self, raw: str, meta: dict | None = None) -> str:
        """Re-run a turn that would just repeat the last reply; fall back to asking the
        caller to say it again, which is at least honest and moves the call forward."""
        if not self.guard.enabled:
            return raw
        say = say_of(_parse_json(raw))
        if not say or not self._is_repeat(say):
            return raw
        log.info("guard: reply repeats an earlier turn — retrying once")
        try:
            msgs = self.call_messages() + [{
                "role": "system",
                "content": "You just gave this exact reply on an earlier turn: \"" + say[:200] +
                           "\". Do NOT say it again. Answer THIS message specifically, using the "
                           "facts above. If you cannot understand the caller, set \"kind\":\"unclear\" "
                           "and leave \"say\" empty."}]
            second = self.llm.chat(msgs, force_json=True, model=self.model,
                                   num_predict=self.reply_budget(self.last_user_text()), meta=meta,
                                   num_ctx=self.context_budget(self.last_user_text()))
        except Exception as e:  # noqa: BLE001
            log.warning("guard: repeat-retry failed (%s)", e)
            second = ""
        say2 = say_of(_parse_json(second)) if second else ""
        if say2 and not self._is_repeat(say2):
            return second
        # Still stuck. Do NOT claim we didn't hear them — the caller may have spoken
        # perfectly clearly (they had, three times, naming their doctor). If a booking is
        # under way, move it forward by asking for the next thing we genuinely need.
        missing = [k for k in self.pack.get("booking", {}).get("required", [])
                   if not self.slots.get(k)]
        if self.booking_started and missing:
            log.warning("guard: still repeating — advancing the booking instead")
            return json.dumps({"kind": "answer", "say": "",
                               "action": {"type": "collect", "slots": {}, "missing": missing}},
                              ensure_ascii=False)
        log.warning("guard: still repeating after retry — asking the caller to repeat")
        return json.dumps({"kind": "unclear", "say": "", "action": {"type": "none"}})

    def retry_if_looped(self, raw: str, meta: dict | None = None) -> str:
        """Re-run a reply that collapsed into repetition; keep the better of the two."""
        if not (self.guard.enabled and self.guard.check_repetition):
            return raw
        say = say_of(_parse_json(raw))
        if not say or not is_degenerate(say):
            return raw
        log.info("guard: reply looped — retrying once with a tighter instruction")
        try:
            second = self.retry_short(meta)
        except Exception as e:  # noqa: BLE001  — the safe line still covers us
            log.warning("guard: retry failed (%s); falling back to the safe line", e)
            return raw
        say2 = say_of(_parse_json(second))
        return second if say2 and not is_degenerate(say2) else raw

    def handle_audio(self, audio_path: str) -> dict:
        heard = self.transcribe(audio_path)
        if not heard:
            lang, roman = self.reply_style(self.last_user_text())
            return {"say": safe_line("repeat", lang, self.pack, roman),
                    "action": {"type": "none"}, "heard": ""}
        res = self.handle_text(heard)
        res["heard"] = heard
        return res

    def transcribe(self, audio) -> str:
        """`audio` may be raw WAV bytes (from the WS) or a file path (CLI)."""
        if not self.stt:
            return ""
        hint = ", ".join(self.pack.get("vocabulary", []))
        eff = self._effective_lang()
        # STT language: a config/UI pin (eff) is honoured; in AUTO we let Whisper detect
        # every turn (NOT pinned to the lock) so a wrong lock can be DETECTED and corrected —
        # only the REPLY language is locked (stable), which is what stops the flip-flopping.
        # `denoise` is only passed when the caller has actually toggled it: stt.py is a
        # pluggable provider interface, and adding a required kwarg would break every
        # existing adapter. Providers that don't know it fall back cleanly.
        # `denoise=False` unconditionally: pipeline.prepare has already decided
        # whether this turn needed cleaning and, if so, done it. Leaving the provider's
        # own switch alive ran DeepFilterNet a SECOND time on audio it had just
        # produced — a straight ~500ms of added latency, and a second pass of
        # suppression on speech that had already been through it once.
        kw = {"hint": hint, "language": eff, "fast": self.stt_fast, "denoise": False}
        try:
            text, det, prob = self.stt.transcribe(audio, **kw)
        except TypeError:
            kw.pop("denoise", None)
            log.debug("STT provider does not support the denoise switch — ignoring it")
            text, det, prob = self.stt.transcribe(audio, **kw)
        if text and det:
            self._last_det = (det, prob)
        if eff is None and text:
            self._consider_lock(text, det, prob)
        return text

    def _consider_lock(self, text: str, det: str | None, prob: float):
        """Decide/adjust the call's latched reply language. Trusts Whisper's own detection
        when reasonably confident (it separates Marathi/Hindi, Assamese/Bengali, English —
        which the Unicode block cannot), else the transcript's dominant script. Works for the
        FIRST latch AND for drift: if a clear language disagrees with the current lock for two
        turns (or once very confidently), we re-lock — so a wrong lock self-corrects."""
        cand = det if (det and prob >= 0.55) else NAME_TO_CODE.get(dominant_script_lang(text) or "")
        if not cand or cand not in LANG_NAMES:
            return                                   # no usable signal → keep whatever we have
        if cand == self.lang_lock:
            self._lang_cand, self._lang_cand_n = None, 0   # confirms the lock → clear drift count
            return
        if cand == self._lang_cand:
            self._lang_cand_n += 1
        else:
            self._lang_cand, self._lang_cand_n = cand, 1
        # (re)latch on a very confident single detection, or two agreeing turns (corroboration)
        if (det == cand and prob >= 0.75) or self._lang_cand_n >= 2:
            self.lang_lock = cand
            self._lang_cand, self._lang_cand_n = None, 0

    # ---- TTS convenience ----------------------------------------------------
    def tts_bytes(self, text: str):
        text = clean_for_speech(text)
        if not text.strip():
            return None
        try:
            if self.voice == "__brand__" and self.clone:   # owner's cloned brand voice
                return _playable(self.clone.synth(text))
            if self.indic and str(self.voice).startswith("parler:"):   # AI4Bharat Indian voice
                return _playable(self.indic.synth(text))
            if not self.tts:
                return None
            audio = _playable(self.tts.synth(text, self.voice))
            # The voice couldn't pronounce this script (e.g. macOS `say` with no Hindi
            # voice installed). Silence here is indistinguishable from the agent cutting
            # out, so remember it and let the UI explain.
            self.mute_reason = ("no_voice_for_script"
                                if audio is None and getattr(self.tts, "last_skipped_script", False)
                                else None)
            return audio
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger("zensuvidha.tts").warning("synthesis failed: %s", e)
            return None

    # A voiceprint built from ONE utterance is a thin sample. These control how far it
    # is allowed to widen using the caller's own later turns.
    VOICEPRINT_MAX = 5          # stop refining after this many contributions
    VOICEPRINT_MARGIN = 0.10    # only matches this far ABOVE threshold may contribute,
                                #   so an impostor who scrapes past the threshold once
                                #   cannot pull the print toward themselves. The thin-
                                #   print problem is solved by the provisional window
                                #   below rather than by lowering this.
    # If the enrolled print is bad, EVERY later turn is refused and the caller is locked
    # out of their own call with no way back. Observed live: a print taken from a short
    # fragment scored the same person 0.01-0.08 against themselves, turn after turn.
    # Nobody's real voice drifts that far, so a run of near-zero scores means the PRINT
    # is wrong, not the speaker — throw it away and enrol again.
    REENROL_AFTER = 3           # consecutive refusals before we distrust the print
    # ...but only two when the print has never been CORROBORATED. A print built from one
    # utterance and never confirmed by the caller's own later turns has earned no
    # loyalty. The case that matters: a call opening while music plays enrols the SINGER
    # — measured, the real caller then scores 0.14 and 0.03 against it and loses a turn
    # for every refusal it takes to notice. Halving that halves the damage, and the
    # protection is unchanged: it still has to be the SAME voice refused each time,
    # which is what stops a series of strangers taking the call over.
    REENROL_PROVISIONAL = 2
    # A print built from ONE short utterance is a guess, not an identity. Until it has
    # been corroborated by a second turn the gate must not refuse anybody — it widens
    # the print instead. Observed live: enrolled on a 2.1s "Can you hear me?", the
    # caller's own next sentences scored 0.45 and 0.51 and were discarded.
    # ...and the provisional phase must END even if corroboration never arrives. A
    # caller in a noisy shop may produce only MIXED clips, none of which may widen the
    # print — leaving the gate permanently off and answering anybody. Measured live:
    # a colleague speaking ALONE was answered, four turns into a call, because the print
    # had never been confirmed. After this many turns the gate enforces with whatever it
    # has; an imperfect print is still better than no gate.
    # TWO, and no more. At the first mismatch we genuinely cannot tell "my print is
    # wrong" from "this is a stranger" — accepting risks one stray turn, refusing risks
    # the caller's. That choice is unavoidable, so the honest thing is to bound it: two
    # turns is exactly long enough for a returning voice to win as a rival (RIVAL_WINS),
    # which is the mechanism that repairs a bad print. After that the gate enforces,
    # even if nothing ever corroborated the print — a caller in a noisy shop produces
    # only mixed clips, none of which can widen it, and the gate must not stay off for
    # the whole call because of that. Measured live: at 3 turns a colleague speaking
    # ALONE was still being answered.
    PROVISIONAL_MAX_TURNS = 2
    VOICEPRINT_TRUST_N = 3      # three of the caller's own utterances before the gate
                                #   is allowed to refuse anyone. Two was not enough: the
                                #   print reached n=2 on the very next turn and started
                                #   enforcing again while it was still mostly one clip.
    # ...but "provisional" is not "open house". A voice merely NEAR the print is
    # plausibly the caller being misjudged by a thin sample; anything further away is
    # somebody else, or somebody else mixed IN. Every number here is measured on this
    # codebase:
    #     a different speaker            0.03 - 0.15   refuse
    #     the caller PLUS a colleague     0.364        refuse — a mixed clip must never
    #                                                  widen the print toward the
    #                                                  person who interrupted
    #     the caller misjudged by their   0.45, 0.51   forgive — this is the reported
    #     own 2.1s enrolment clip                      lockout, and it is the caller
    # The floor sits in the gap between the last two.
    PROVISIONAL_FLOOR = 0.40
    REENROL_SAME = 0.55         # ...and only when it is the SAME voice each time. Three
                                #   DIFFERENT strangers must never hand the call to the
                                #   third of them; one voice locked out repeatedly is the
                                #   caller, and that is the case worth recovering.

    def clean_audio(self, audio):
        """Run one turn through the whole audio front-end, in the measured order.

        Isolation first, on RAW (identity is worse on denoised audio, and this is the
        only step that can remove another human VOICE — a colleague, a television, a
        singer on a background track). Denoising second, and only when the recording
        is actually noisy: it costs ~500ms and makes a clean turn measurably worse.

        Returns (audio, info) and hands back exactly what it was given whenever
        anything is uncertain. See pipeline.py for the measurements behind the order.
        """
        from . import pipeline
        if self.stt is None:
            return audio, None
        out, info = pipeline.prepare(
            audio,
            decode=self.stt._decode,
            denoiser=getattr(self.stt, "denoiser", None),
            diarizer=self.diarizer,
            gate=self.speaker_gate,
            voiceprint=self.voiceprint,
            denoise_mode=self.stt_denoise,
            isolate_mode=self.isolate_on,
            was_denoised=self._was_denoised)
        self._was_denoised = info["denoised"]
        if info["isolated"] or info["denoised"]:
            log.info("audio: %s%s%s(%.0fms)",
                     f"{info['speakers']} voices, kept {info['kept_s']:.1f}s of "
                     f"{info['total_s']:.1f}s " if info["isolated"] else "",
                     (f"denoised ({info['snr_db']}dB floor-to-voice) "
                      if info["snr_db"] is not None else "denoised (forced) ")
                     if info["denoised"] else "",
                     "" if info["isolated"] or info["denoised"] else "unchanged ",
                     info["iso_ms"] + info["den_ms"])
        return out, info

    # Kept as the old name so nothing that already calls it breaks; the pipeline
    # module is now the single place that decides what happens to a turn's audio.
    def isolate_caller(self, audio):
        return self.clean_audio(audio)

    def check_speaker(self, audio, speakers=None, heard=None):
        """Is this the person whose call this is?

        `speakers` is how many voices diarization COUNTED in this clip, or None when
        nobody counted. It decides whether a forgiven near-miss may widen the print:
        the score alone cannot, because the caller misjudged by a thin sample and the
        caller with a colleague talking over them BOTH land at ~0.45-0.51 (measured).
        Only the voice count tells them apart, and the reported lockout showed "1
        voice" on every discarded turn.

        The FIRST utterance long enough to take a voiceprint from defines the caller —
        on a phone line that is by definition the person who rang. Everything after is
        compared against it, so a bystander or a television never becomes a turn.

        `heard` is the transcript, when there is one. It is consulted ONLY on the
        refusal path, as a second opinion the voiceprint cannot give: a turn that
        answers the question we just asked is the caller even when the audio scores
        badly. It can rescue a turn and never discard one — see `expectation.py`.

        Returns (accept, similarity). Accepts whenever it cannot judge: a false reject
        silences the actual caller, which is a far worse failure than letting one stray
        utterance through.
        """
        # Cleared per judgement so it can never describe a PREVIOUS turn. The server
        # consumes it on every accepted turn, so this is belt-and-braces — but a stale
        # "rescued" tag in the inspector would be actively misleading, and the cost of
        # preventing that is one assignment.
        self._last_rescue = None
        if self.speaker_gate is None:
            return True, None
        from .speaker import MIN_ENROL_S
        if self.voiceprint is None:
            vec = self.speaker_gate.embed(audio, min_seconds=MIN_ENROL_S)
            if vec is not None:
                self.voiceprint = vec
                self._voiceprint_n = 1
                log.info("speaker: enrolled the caller's voice")
            return True, None                     # never judge the enrolling turn
        # judge() hands back the embedding it already computed. Calling matches() and
        # then embed() again ran the ECAPA encoder TWICE on every refused turn — the
        # single most expensive operation in the gate, paid twice for one answer.
        self._gate_turns += 1
        ok, sim, vec = (self.speaker_gate.judge(self.voiceprint, audio)
                        if hasattr(self.speaker_gate, "judge")
                        else (*self.speaker_gate.matches(self.voiceprint, audio), None))
        if ok:
            # It has now demonstrated, on THIS call and THIS microphone, that it can
            # recognise this caller. Only from here is it entitled to refuse anybody.
            self._gate_proven = True
        if not ok and not self._gate_proven:
            # IT HAS NEVER RECOGNISED THEM. A gate that has not once matched the caller
            # has no evidence it can, and refusing on that is not caution — it is a coin
            # toss that silences whoever is actually on the phone.
            #
            # This is not hypothetical. The 0.55 threshold was calibrated on macOS `say`
            # voices, where the same speaker scores 0.867 and the closest impostor 0.429.
            # Measured on a real caller's own microphone, in a room the router judged
            # CLEAN (13-34dB floor-to-voice), their own consecutive turns scored
            # 0.27 / 0.37 / 0.27 / 0.29 / 0.41 — never once above the threshold. The gate
            # could not recognise them at all, so every refusal it made was noise.
            #
            # Once it HAS matched them once, the threshold is meaningful for this mic and
            # normal enforcement resumes.
            log.info("speaker: %.2f is below %.2f, but this gate has never matched the "
                     "caller on this call — it has not earned the right to refuse them",
                     sim if sim is not None else -1, self.speaker_gate.threshold)
            # Answer it, and let the RIVAL rule decide whose voice this is. Widening
            # blindly here would let one stranger's turn poison the print, which is the
            # thing the rival rule exists to prevent: it adopts a voice only once the
            # SAME one has come back, which a passer-by does not do and the caller does.
            return self._note_rival(vec, sim, speakers)
        if not ok:
            # PROVISIONAL PRINT — do not refuse the caller on the strength of one
            # utterance. A thin sample is not merely imprecise, it is UNRECOVERABLE:
            # too close to trigger the re-enrolment rescue (which wanted < 0.25) and
            # too far to pass refinement (which wanted > 0.65), so the caller was
            # refused for the rest of the call with no way back. Observed live —
            # enrolled on a 2.1s "Can you hear me?", their own next sentences scored
            # 0.45 and 0.51 and were discarded, twice.
            #
            # Only a NEAR miss is forgiven, and only while the print is still thin:
            # that shape means "plausibly them, judged by a bad sample", and folding
            # the turn in is exactly the repair. A voice near zero is somebody else and
            # is still refused, even on turn two.
            if (self._voiceprint_n < self.VOICEPRINT_TRUST_N
                    and self._gate_turns <= self.PROVISIONAL_MAX_TURNS
                    and sim is not None):
                # NEVER REFUSE ON A GUESS. While the print is one un-corroborated
                # sample, a disagreement means we do not know which of the two voices
                # is the caller — and refusing is the guess that silences them. That
                # contradicts this component's own rule, and it is how a call opening
                # while music plays cost the caller a turn: the SONG was enrolled, the
                # caller then scored 0.14 against it and was thrown away.
                #
                # So let the call decide instead. A near miss is the caller misjudged by
                # a thin sample — widen. Anything further away is a genuinely different
                # voice, which we remember as a RIVAL and still answer. Whichever voice
                # comes back is the one having the conversation; a song does not follow
                # up on its own question.
                if sim < self.PROVISIONAL_FLOOR:
                    return self._note_rival(vec, sim, speakers)
                log.info("speaker: %.2f is below %.2f, but the voiceprint is still "
                         "provisional (%d utterance(s)) and this is close enough to be "
                         "the same person — accepting and widening it instead",
                         sim, self.speaker_gate.threshold, self._voiceprint_n)
                self._gate_rejects, self._reject_vec = 0, None
                self._rival, self._rival_n = None, 0
                # Learn from it unless the clip is KNOWN to hold more than one voice.
                # A mixed clip scores like a misjudged caller, and folding one in widens
                # the caller's identity toward whoever interrupted them — so a counted
                # 2+ is refused as evidence.
                #
                # But `None` means nobody counted, which is not the same as "mixed", and
                # treating it as such reopened the original lockout by a side door: with
                # isolation off, the caller's own near misses could never corroborate the
                # print, so the provisional window expired with a print still too thin to
                # recognise them and they were refused at 0.49 on their own call.
                if (speakers or 1) <= 1:
                    before = self._voiceprint_n
                    self._widen_voiceprint(audio)
                    if self._voiceprint_n > before:
                        # This turn REPAIRED the print, so it must not burn the
                        # provisional clock. The cap exists for turns we can learn
                        # nothing from; a near miss is the opposite of that.
                        #
                        # Observed live, with music playing as the call opened: the
                        # caller scored 0.09, then 0.43, then 0.52 against a print
                        # enrolled from a contaminated first turn. The widening was
                        # working — 0.43 -> 0.52, climbing toward the 0.55 threshold —
                        # and the turn cap closed the window one turn early and
                        # discarded them. Repair has to be allowed to finish; the
                        # `_voiceprint_n` condition still ends the phase once the print
                        # has actually been corroborated.
                        self._gate_turns = max(0, self._gate_turns - 1)
                else:
                    log.info("speaker: %d voices in that clip — accepting the turn but "
                             "not learning the caller's voice from it", speakers)
                return True, sim
            # A run of scores this low means the enrolled print is wrong, not that three
            # different strangers spoke in a row. Discard it and re-enrol on this turn,
            # so a bad first sample cannot lock the caller out for the whole call.
            if vec is None:                       # older gate without judge()
                vec = self.speaker_gate.embed(audio)
            same_as_last = (self._reject_vec is not None and vec is not None
                            and self.speaker_gate.similarity(self._reject_vec, vec)
                            >= self.REENROL_SAME)
            self._reject_vec = vec if vec is not None else self._reject_vec
            # The guard against an intruder taking over is SAME_AS_LAST — one voice
            # refused over and over — not how low the score is. Requiring < 0.25 as
            # well created a dead zone: a caller at 0.45-0.51 was rejected every turn
            # and never rescued, because they were too CLOSE to their own bad print.
            # The earlier lockout scored 0.01-0.08 and the one before that 0.45-0.51;
            # both were the real caller, which is the point — the score cannot tell
            # you, only the consistency can.
            if sim is not None and (same_as_last or self._gate_rejects == 0):
                self._gate_rejects += 1
                need = (self.REENROL_AFTER if self._voiceprint_n >= self.VOICEPRINT_TRUST_N
                        else self.REENROL_PROVISIONAL)
                if self._gate_rejects >= need and (speakers or 1) <= 1:
                    # ...and never from a clip KNOWN to hold more than one voice. Widening
                    # refuses those and so does rival adoption, for the same reason: a
                    # blend of two people must not become the caller's identity. The
                    # rescue was the one path that still would, so a caller with somebody
                    # always talking beside them could have that MIXTURE enrolled as
                    # themselves.
                    log.warning("speaker: %d refusals in a row at ~%.2f — the voiceprint "
                                "looks wrong, re-enrolling on this turn",
                                self._gate_rejects, sim)
                    self.voiceprint, self._voiceprint_n = None, 0
                    self._gate_rejects, self._reject_vec = 0, None
                    self._gate_turns = 0
                    return self.check_speaker(audio, speakers, heard)  # enrol on this utterance
            else:
                self._gate_rejects = 0
            # LAST CHANCE, and the only one this signal gets: does the TURN look like
            # the answer to the question we just asked out loud?
            #
            # The voiceprint has one failure it cannot see from the inside — loud audio
            # at the mic drives the caller's score against their OWN voice to 0.07, at
            # which point every refusal it makes is noise. This asks something the
            # acoustics cannot: ten digits arriving straight after "may I have your
            # mobile number?" is the caller, whatever ECAPA thinks of the recording.
            #
            # Rescue ONLY. It sits on the refusal path and nowhere else, so a wrong
            # score here can never cost the caller a turn — only give one back.
            if heard:
                rescue, why = expectation.should_rescue(
                    heard, self.pending_slot or self.last_asked_slot, self.pack,
                    (self.pack.get("booking", {}) or {}).get("slot_aliases"))
                if rescue:
                    log.info("speaker: %.2f is below %.2f, but this turn IS the answer "
                             "we asked for (%s) — accepting it",
                             sim if sim is not None else -1,
                             self.speaker_gate.threshold, why)
                    # The refusal streak is deliberately NOT cleared: this turn is
                    # evidence the PRINT is wrong, and `_gate_rejects` reaching
                    # REENROL_AFTER is what repairs it. Clearing it here would trade a
                    # permanent fix for a per-turn rescue, and the first turn that
                    # happened not to match an expectation would be refused again.
                    self._last_rescue = why
                    return True, sim
            log.info("speaker: ignoring a different voice (similarity %.2f < %.2f)",
                     sim if sim is not None else -1, self.speaker_gate.threshold)
            return ok, sim
        self._gate_rejects, self._reject_vec = 0, None
        # Fold a CONFIDENT match back into the print. One utterance is a thin sample of a
        # voice: measured, a caller enrolled on one clean clip falls from 0.87 to 0.42 in
        # heavy noise and gets rejected — silencing the actual caller, the worst failure
        # this component has. Averaging a few accepted utterances widens the print to
        # cover how they really sound on THIS line.
        # Only well-clear matches contribute, and only a handful, so a borderline
        # impostor who slips through once cannot drag the print toward themselves.
        if (sim is not None and speakers != 0
                and sim >= self.speaker_gate.threshold + self.VOICEPRINT_MARGIN):
            self._widen_voiceprint(audio)
        return ok, sim

    # A rival must be the SAME voice returning, not simply another stranger — the same
    # rule that stops a series of different people taking over the call.
    RIVAL_SAME = 0.55
    RIVAL_WINS = 2          # ...heard this many times while the print is still a guess

    def _note_rival(self, vec, sim, speakers):
        """Answer a turn we cannot attribute, and remember the voice that spoke it.

        Returns (True, sim) always: while the enrolled print is a guess, refusing is
        not a safer answer than accepting, it is just a different guess — and the one
        that costs the caller their turn.

        A rival is only REMEMBERED from a clip known to hold one voice. The same reason
        a mixed clip may not widen the print applies twice over here: promoting one
        would make a blend of two people the caller for the rest of the call.
        """
        if vec is None or speakers != 1:
            return True, sim
        if self._rival is not None and \
                self.speaker_gate.similarity(self._rival, vec) >= self.RIVAL_SAME:
            self._rival_n += 1
        else:
            self._rival, self._rival_n = vec, 1
        if self._rival_n >= self.RIVAL_WINS:
            # This voice has come back; the enrolled one never did. The conversation is
            # being had by the rival, so it is the caller and the original was whatever
            # happened to be making a noise when the call opened.
            log.warning("speaker: a second voice has now spoken %d times while the "
                        "enrolled print (%.2f) never returned — treating IT as the "
                        "caller", self._rival_n, sim)
            self.voiceprint, self._voiceprint_n = self._rival, 1
            self._rival, self._rival_n, self._gate_turns = None, 0, 0
            self._gate_rejects, self._reject_vec = 0, None
        else:
            log.info("speaker: %.2f does not match the provisional print — answering "
                     "anyway and remembering this voice", sim)
        return True, sim

    def _widen_voiceprint(self, audio):
        """Fold this utterance into the caller's print.

        One utterance is a thin sample of a voice: measured, a caller enrolled on one
        clean clip falls from 0.87 to 0.42 in heavy noise and is then refused — the
        worst failure this component has. Averaging a few utterances widens the print
        to cover how they actually sound on THIS line. Capped, so a borderline impostor
        who slips through once cannot drag it toward themselves.
        """
        if self.voiceprint is None or self._voiceprint_n >= self.VOICEPRINT_MAX:
            return
        from .speaker import MIN_ENROL_S
        vec = self.speaker_gate.embed(audio, min_seconds=MIN_ENROL_S)
        if vec is None:
            return
        import numpy as np
        n = self._voiceprint_n
        merged = (self.voiceprint * n + vec) / (n + 1)
        norm = float(np.linalg.norm(merged))
        if norm:
            self.voiceprint = merged / norm
            self._voiceprint_n = n + 1
            log.info("speaker: voiceprint widened to %d utterance(s)", n + 1)

    def tts_stream(self, text: str):
        """Incremental synthesis for this call's active voice, or None if unavailable.

        Returns an iterator of (pcm_int16_bytes, sample_rate) when the selected
        provider can render a sentence progressively, so the server can ship audio
        before the clip is finished. None means "use tts_bytes" — the whole-clip path
        is untouched, so no provider is made slower by this existing.
        """
        text = clean_for_speech(text)
        if not text.strip():
            return None
        if self.voice == "__brand__" and self.clone:
            prov, voice = self.clone, None
        elif self.indic and str(self.voice).startswith("parler:"):
            prov, voice = self.indic, None
        else:
            prov, voice = self.tts, self.voice
        stream = getattr(prov, "synth_stream", None) if prov else None
        if stream is None:
            return None
        try:
            return stream(text, voice)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger("zensuvidha.tts").warning("stream setup failed: %s", e)
            return None

    def turn(self, text: str | None = None, audio_path: str | None = None) -> dict:
        res = self.handle_audio(audio_path) if audio_path else {**self.handle_text(text), "heard": text}
        audio = self.tts_bytes(res["say"])
        res["audio_b64"] = base64.b64encode(audio).decode() if audio else None
        return res
