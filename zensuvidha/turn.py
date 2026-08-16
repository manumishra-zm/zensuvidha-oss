"""When is the caller finished? One place, for every transport.

WHY THIS MODULE EXISTS

The decision "have they stopped talking" was being made in three places that could not
see each other:

    web/index.html        the browser endpointer — silence, learned pause, slot extras
    zensuvidha/server.py  the lexical verdict (looks_incomplete / looks_complete) and
                          the prosody verdict, pushed to the browser as flags
    transport.Endpointer  a second, simpler copy for a carrier, which has no browser

Three copies of one policy is how they drift. The telephony path already had a
DIFFERENT window ladder from the browser — 800/1200ms with no learned pause, no slot
extras, no prosody — so a caller on a phone line would have been endpointed by rules
this project spent two rounds of live testing tuning away from.

So the numbers live here, once. The browser is sent them on connect and evaluates them;
`transport.Endpointer` imports them directly. Adding a signal now means adding it in one
file, and a phone caller gets it for free.

WHAT DECIDES A TURN, in order of how much it is trusted:

    1. the WORDS      looks_incomplete / looks_complete — a finished phone number
                      admits no continuation; "मेरा मोबाइल नंबर" plainly does
    2. a FILLER       "um", "मतलब", "यानी" — someone searching for a word, which
                      silence cannot distinguish from someone who has finished
    3. the VOICE      the pitch contour (zensuvidha/prosody.py)
    4. the CALLER     how long THIS person pauses mid-sentence, learned per call
    5. the CLOCK      how long they have been talking at all

EAGERNESS is the one knob an operator should have. A clinic taking bookings from
elderly callers wants Patient; a restaurant taking table numbers wants Eager. It scales
the ladder rather than replacing it, so everything measured above still holds.
"""
from __future__ import annotations

# --- the ladder, as measured on real callers --------------------------------------
# Every number here was arrived at by a live call going wrong. They are not defaults
# in the "seemed reasonable" sense and should not be changed without a recording.
NORMAL = {
    # A one-word answer is finished the moment it stops. "haan" needs no thinking time
    # and making it wait is most of what makes a call feel slow.
    "base_ms": 800,
    # …but an utterance long enough to HAVE a middle gets longer, because the middle is
    # where people stop to find a word. The old flat ~470ms cut callers off mid-sentence:
    # people pause before saying a phone number, and "मेरा मोबाइल नंबर" was being sent
    # WITHOUT the number.
    "long_ms": 1200,
    "long_utt_ms": 1500,
    # Ceiling, so a noisy line can never hold a turn open forever.
    "max_ms": 2000,
    # The transcript is unambiguously finished (a complete phone number, a bare "yes") —
    # nothing can follow, so close now instead of sitting out a window sized for someone
    # still thinking.
    "settled_ms": 400,
    # The words say they are mid-sentence. This outranks everything below it.
    "hold_extra_ms": 900,
    # A filler is weaker evidence than a dangling postposition but stronger than silence.
    "filler_extra_ms": 500,
    # People pause differently depending on what was ASKED. Reading a phone number aloud
    # has long gaps between digit groups; a name does not.
    "expect_extra_ms": {"phone": 600, "datetime": 300, "name": 150},
    # How much of this caller's demonstrated mid-sentence pause to clear.
    "pause_factor": 1.25,
    # What the pitch contour is allowed to do. Asymmetric on purpose: closing early
    # chops a sentence in half, waiting too long only costs a pause.
    "tone_scale": {"finished": 0.80, "holding": 1.25, "unsure": 1.0},
}

# Eagerness scales the ladder; it does not replace it. `settled_ms` is deliberately NOT
# scaled up for Patient — "the words are finished and nothing can follow" is a fact
# about the transcript, not a preference, and making a completed phone number wait
# longer is just a slower call.
EAGERNESS = {
    "eager":   {"window": 0.70, "hold": 0.80, "max": 0.75},
    "normal":  {"window": 1.00, "hold": 1.00, "max": 1.00},
    "patient": {"window": 1.45, "hold": 1.30, "max": 1.60},
}
DEFAULT_EAGERNESS = "normal"


def policy(eagerness: str | None = None) -> dict:
    """The full ladder for one eagerness setting, ready to send to a client.

    Returned as plain data rather than behaviour so the browser can evaluate it without
    a second implementation of the rules, and so a change here reaches the phone path
    and the browser path in the same commit.
    """
    name = (eagerness or DEFAULT_EAGERNESS).lower()
    if name not in EAGERNESS:
        name = DEFAULT_EAGERNESS
    k = EAGERNESS[name]
    p = dict(NORMAL)
    p["expect_extra_ms"] = dict(NORMAL["expect_extra_ms"])
    p["tone_scale"] = dict(NORMAL["tone_scale"])
    p["base_ms"] = round(NORMAL["base_ms"] * k["window"])
    p["long_ms"] = round(NORMAL["long_ms"] * k["window"])
    p["max_ms"] = round(NORMAL["max_ms"] * k["max"])
    p["hold_extra_ms"] = round(NORMAL["hold_extra_ms"] * k["hold"])
    p["filler_extra_ms"] = round(NORMAL["filler_extra_ms"] * k["hold"])
    p["eagerness"] = name
    return p


def window_ms(pol: dict, *, utt_ms: float = 0.0, learned_pause_ms: float = 0.0,
              hold: bool = False, settled: bool = False, filler: bool = False,
              expect: str | None = None, tone: str | None = None) -> int:
    """How much silence ends THIS turn, given everything currently known about it.

    The order is the whole design and each step is a bug that was fixed:

      * `settled` closes early — but only when nothing is holding it open, because
        "unfinished" is the conservative answer whenever the two disagree.
      * `hold` and `filler` ADD time rather than replacing the window, so a long
        sentence keeps its long window and a short one is not made to wait for it.
      * the clamp is applied AFTER the tone scale. Scaling an already-clamped window
        pushed it past the ceiling, and the client stores the result as the learned
        pause — so one "holding" verdict used to ratchet the caller's window up for the
        rest of the call, above its own maximum.
    """
    base = pol["base_ms"] if utt_ms < pol["long_utt_ms"] else max(
        pol["long_ms"], learned_pause_ms * pol["pause_factor"])
    extra = pol["expect_extra_ms"].get(expect or "", 0)
    if hold:
        extra += pol["hold_extra_ms"]
    elif filler:
        # Weaker than a dangling postposition, stronger than silence — and never both,
        # or a "मेरा नंबर, उम्म…" turn would be charged twice for one hesitation.
        extra += pol["filler_extra_ms"]
    if settled and not (hold or filler):
        return int(pol["settled_ms"])
    win = min(pol["max_ms"] + extra, base + extra)
    if hold:
        return int(win)                       # the words outrank the voice
    scale = pol["tone_scale"].get(tone or "", 1.0)
    return int(min(pol["max_ms"] + extra, round(win * scale)))
