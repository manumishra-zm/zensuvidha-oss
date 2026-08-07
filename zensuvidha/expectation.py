"""Does this turn look like an answer to the question we just asked?

A SECOND opinion on identity, alongside the voiceprint — not instead of it. pyannote,
ERes2Net, ECAPA and DeepFilterNet all keep doing exactly what they did; this adds one
more signal to consult when they refuse.

Why it exists
-------------
The acoustic path answers "whose voice is this?" and, when it works, that is the better
answer. But it has one documented failure it cannot see from the inside: loud audio at
the microphone drives the caller's score against their OWN voice to 0.07. At that point
the similarity number is not merely imprecise, it is uninformative, and every refusal
made on it is noise.

This signal never touches the voiceprint, so it survives exactly the case that breaks
it. A turn carrying ten digits, arriving straight after we asked for a mobile number, is
the caller — whatever ECAPA thinks of the recording.

The rule that keeps it safe
---------------------------
It can only ever RESCUE a turn, never discard one. Callers say things with no bearing on
the business at all — "hello?", "can you hear me?", "haan", "my son has a fever" — and a
relevance threshold that rejected those would repeat the mistake the 0.55 speaker
threshold made, in a domain where the caller has no way to try harder.

    score >= RESCUE_AT  ->  a turn the gate REFUSED is accepted instead
    score <  RESCUE_AT  ->  nothing happens; the gate's own verdict stands

There is no path from this module to a rejection, and `test_expectation.py` pins that by
asserting it over the whole corpus of things callers actually say.
"""
from __future__ import annotations

import re

# ── what "an answer" looks like ─────────────────────────────────────────────────
# Deliberately narrow. A weak guess must not rescue a stranger, so each rule fires only
# on a shape a bystander is unlikely to produce at the exact moment we asked for it.

_DIGIT_RUN = re.compile(r"\d[\d\s\-]{8,}\d")          # a phone, however it is grouped
_TIME_HINT = re.compile(
    r"\b\d{1,2}\s*(?:o'?clock|am|pm|baje)"
    r"|\d{1,2}\s*(?:बजे|గంటల)"
    r"|\b(?:morning|afternoon|evening|tonight|tomorrow|today|noon|midnight"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|kal|aaj|subah|shaam)\b"
    r"|(?:सुबह|शाम|दोपहर|कल|आज|रात)"
    r"|(?:ఉదయం|సాయంత్రం|రేపు|ఈరోజు)",
    re.I)
_WORD_SPLIT = re.compile(r"[^\wऀ-෿]+")      # keeps Devanagari/Telugu/Tamil…

# Confidence each shape earns. Scores ACCUMULATE, so two independent medium signals can
# rescue but no single weak one can. Only the two shapes a passer-by essentially never
# satisfies by coincidence — the exact number we asked for, and one of our own proper
# nouns — are strong enough alone.
STRONG, MEDIUM, WEAK = 1.0, 0.5, 0.25
RESCUE_AT = 1.0
MAX_SCAN = 2000        # chars; a real turn is far shorter

_VOCAB_KEY = "_expectation_vocab"     # cached ON the pack, so it dies with the pack


def _norm_digits(text: str) -> str:
    """Digits only, so '8-9-2-0…' and '8920…' are the same number."""
    return re.sub(r"\D", "", text or "")


def expectation_score(text: str, pending_slot: str | None, pack: dict | None = None,
                      aliases: dict | None = None) -> tuple[float, str]:
    """How much this turn looks like the answer we were waiting for.

    Returns (score, why). `why` goes to the log and the inspector — a rescue nobody can
    explain afterwards is one nobody can debug.

    Pure: no session state, no model, no I/O. A few regexes over one short string.
    """
    text = (text or "").strip()
    if not text:
        return 0.0, ""
    # A real turn is at most ~40s of speech. Anything vastly longer is a degenerate
    # transcript the STT trimmer did not fully catch, and scanning all of it makes this
    # unbounded on a path that runs for every refused turn. Measured: 6us on a typical
    # turn, 1.7ms on 20k chars — small either way, but bounded beats small.
    if len(text) > MAX_SCAN:
        text = text[:MAX_SCAN]

    score, why = 0.0, []

    # ── the pending slot: the strongest signal, because WE chose the question ──
    if pending_slot == "phone":
        digits = _norm_digits(text)
        # Exactly one plausible run. Two candidates means we cannot tell which was
        # meant — the same rule `_capture_phone` already applies to the slot itself.
        if 10 <= len(digits) <= 13 and len(_DIGIT_RUN.findall(text)) <= 1:
            score += STRONG
            why.append(f"a {len(digits)}-digit number right after we asked for one")

    elif pending_slot == "datetime":
        if _TIME_HINT.search(text):
            score += MEDIUM
            why.append("a time expression right after we asked when")

    elif pending_slot == "name":
        # A name is short and is not a question. Both halves matter: "what is your
        # name?" said back at us is an ECHO, not an answer, and must not rescue.
        # WEAK on purpose — a bystander's "hello there" has this shape too, so it can
        # only ever corroborate something else.
        words = text.split()
        if 1 <= len(words) <= 5 and "?" not in text:
            score += WEAK
            why.append("a short non-question right after we asked for a name")

    if pending_slot and aliases:
        # Doctor, service, stylist, table — whatever this pack calls its slots. The
        # caller using one of OUR OWN proper nouns is close to conclusive: a television
        # in the background does not say "Dr Anil Sharma" on cue.
        named = _names_one_of(text, aliases.get(pending_slot) or {})
        if named:
            score += STRONG
            why.append(f"names {named} right after we asked which one")

    # ── the pack's own vocabulary: corroboration only, never decisive ──
    # This is the branch most likely to be wrong (a caller describing a symptom in
    # their own words may score zero), so it is capped below RESCUE_AT by itself.
    if pack:
        hits = _pack_overlap(text, pack)
        if hits >= 2:
            score += MEDIUM
            why.append(f"uses {hits} of this business's own terms")
        elif hits == 1:
            score += WEAK
            why.append("uses one of this business's terms")

    return score, "; ".join(why)


def _names_one_of(text: str, table: dict) -> str | None:
    """The canonical slot value this turn names, if any.

    Matches on whole words so a pack entry like "Rao" cannot be satisfied by the "rao"
    inside an unrelated word.
    """
    said = {w for w in _WORD_SPLIT.split(text.lower()) if w}
    for canonical, spellings in (table or {}).items():
        for name in [canonical] + list(spellings or []):
            if not name:
                continue
            parts = [p for p in _WORD_SPLIT.split(str(name).lower()) if p]
            if parts and all(p in said for p in parts):
                return canonical
    return None


def _pack_overlap(text: str, pack: dict) -> int:
    """How many of the business's own distinctive words this turn uses."""
    vocab = _pack_vocab(pack)
    if not vocab:
        return 0
    said = {w for w in _WORD_SPLIT.split(text.lower()) if len(w) >= 4}
    return len(said & vocab)


def _pack_vocab(pack: dict) -> frozenset:
    """The pack's distinctive words, built once per pack.

    Cached ON the pack dict rather than in a module-level map keyed by id(): CPython
    reuses addresses after a free, so an id()-keyed cache can hand one pack another
    pack's vocabulary. Storing it here also means it is collected with the pack instead
    of leaking for the life of the process.

    Short tokens are dropped — matching on "and"/"the" would make every utterance look
    relevant and quietly turn this into a rubber stamp.
    """
    hit = pack.get(_VOCAB_KEY)
    if hit is not None:
        return hit
    words: set[str] = set()
    for svc in (pack.get("services") or []):
        words |= set(_WORD_SPLIT.split(str(svc.get("name", "")).lower()))
    for item in (pack.get("knowledge") or []) + (pack.get("common") or []):
        words |= set(_WORD_SPLIT.split(str(item.get("q", "")).lower()))
    vocab = frozenset(w for w in words if len(w) >= 4)
    try:
        pack[_VOCAB_KEY] = vocab
    except Exception:                 # a read-only mapping — recompute each time
        pass
    return vocab


def should_rescue(text: str, pending_slot: str | None, pack: dict | None = None,
                  aliases: dict | None = None) -> tuple[bool, str]:
    """Should a turn the voiceprint refused be accepted anyway?

    The only entry point the gate uses. Returning False changes nothing — the gate's own
    decision stands — so a bad score here can never cost the caller a turn.
    """
    score, why = expectation_score(text, pending_slot, pack, aliases)
    return score >= RESCUE_AT, why
