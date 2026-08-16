"""Is that the end of a sentence, or a pause in the middle of one?

Silence cannot answer this. Neither can words alone. What CAN is the thing every
listener uses and this codebase has never looked at: the pitch of the voice.

    "my name is Manu Mishra."        pitch FALLS through the last syllable → done
    "my name is…"                    pitch stays LEVEL, held                → not done

Every decision path in this project so far has been made on amplitude. `micAnalyser`
computes an FFT and feeds it ONLY to the orb visualiser; the one hand-written filter
on a decision path is the 300-3400Hz biquad, and that is still an energy measurement.
So the pitch contour is genuinely unused information, sitting in audio we already have,
costing one numpy pass to read.

WHY THIS AND NOT A MODEL. The alternative is a learned turn detector — LiveKit ships
one and it is a good idea. But theirs is bound to their agent session, ships under a
non-OSI licence, and covers 14 languages of which Telugu is not one; and training our
own needs the real call recordings this project does not yet have. Declination is not
a learned fact about English, it is what happens when a speaker runs out of breath
support at the end of an utterance, and it holds in Hindi and Telugu declaratives too.
It is the part of the signal available without data.

HOW IT IS USED. It never decides alone. `looks_incomplete()` — the words — always wins,
because a caller who says "my mobile number is" with a beautiful falling contour has
still not given us the number. Prosody only moves the window within the bounds the
existing rules already set:

    finished  →  close sooner   (waiting on someone who has stopped is what makes
                                 a call feel slow)
    holding   →  wait longer    (a held pitch is someone still holding the floor)
    unsure    →  change nothing (inside the dead band — see below)
    unreadable→  change nothing (whispered, too short, or too noisy to read)

Note there is no "rising". A yes/no question does raise pitch, but it could not be
separated reliably on the calibration set, and a capability that cannot be demonstrated
should not be in the API. Questions are handled by the words instead.

FAILS OPEN in every branch: any doubt returns None and the endpointer behaves exactly
as it did before.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger("zensuvidha.prosody")

SR = 16000

# Human speech, generously bounded. Below 60Hz is mains hum and rumble; above 400Hz is
# above almost every adult speaker, and letting the search go higher makes octave errors
# far more likely than it makes a real detection.
F0_MIN, F0_MAX = 60.0, 400.0

FRAME_MS = 40           # long enough to hold two periods of the lowest pitch we accept
HOP_MS = 20

# How much of the tail to read. The final fall happens over roughly the last syllable or
# two; reading further back mixes in earlier phrases and flattens the slope toward zero.
TAIL_MS = 700

# A frame is only usable if the autocorrelation peak is this clear. Unvoiced sounds
# (s, sh, f) and noise have no periodicity, and a weak peak is an octave error waiting
# to happen.
VOICED_R = 0.35

# Minimum usable frames before any verdict. Two points make a line through anything.
MIN_FRAMES = 4

# ---- the score, and why it is a score rather than a threshold on the slope ----------
#
# Measured (see the calibration table at the foot of this file): the pitch slope
# ALONE does not separate the two cases.
# "My name is Manu Mishra." finished at -6.6 st/s and the same sentence cut after "is"
# read -6.7 — indistinguishable. Energy decay alone does not separate them either
# (a finished clip measured 0.97, higher than three of the four unfinished ones).
# TOGETHER they do, because the two failures are uncorrelated: the clip whose pitch
# gives it away has flat energy, and the clip whose energy gives it away has flat pitch.
#
#     score = -slope_st_s + DECAY_WEIGHT * (1 - decay)
#
# …but the separating margin on synthetic speech was 8.2 against 7.1, which is far too
# thin to hang a knife-edge threshold on — nine clips of `say` output is not a
# distribution. So there is a DEAD BAND. The detector may only speak when it is well
# clear of the middle, and says nothing at all in between. On the measured data that
# means it acts on four clips out of nine and is wrong on none of them; the five it
# declines are exactly the ones where a threshold would have been guessing.
DECAY_WEIGHT = 10.0
FINISHED_SCORE = 12.0     # …well above the highest unfinished clip measured (7.1)
HOLDING_SCORE = 5.0       # …well below the lowest finished clip measured (8.2)


def _f0_track(x, sr: int = SR):
    """Pitch per frame over the tail, by autocorrelation. None where unvoiced.

    Autocorrelation rather than YIN or a neural tracker on purpose: this runs on every
    speculative frame, and the decision it feeds only needs the SHAPE of the contour,
    not a musically accurate pitch. The precision that would buy is not the precision
    that is missing.
    """
    import numpy as np
    x = np.asarray(x, dtype="float32").reshape(-1)
    if x.size < int(0.2 * sr):
        return [], []
    tail = x[-int(TAIL_MS / 1000 * sr):]
    n = int(FRAME_MS / 1000 * sr)
    hop = int(HOP_MS / 1000 * sr)
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    f0s, times, rms = [], [], []
    for start in range(0, max(1, len(tail) - n + 1), hop):
        frame = tail[start:start + n]
        if frame.size < n:
            break
        frame = frame - frame.mean()
        energy = float(np.sqrt(np.mean(frame ** 2)))
        rms.append(energy)
        times.append(start / sr)
        if energy < 1e-4:
            f0s.append(None)
            continue
        # Normalised autocorrelation. The r[0] division is what makes VOICED_R mean the
        # same thing at any loudness — without it a quiet voiced frame and a loud
        # unvoiced one are indistinguishable.
        r = np.correlate(frame, frame, mode="full")[n - 1:]
        if r[0] <= 0:
            f0s.append(None)
            continue
        r = r / r[0]
        window = r[lo:hi + 1]
        if window.size == 0:
            f0s.append(None)
            continue
        k = int(np.argmax(window)) + lo
        f0s.append(sr / k if r[k] >= VOICED_R else None)
    return f0s, (times, rms)


def finality(pcm, sr: int = SR) -> dict | None:
    """Read the end of an utterance. Returns None when it cannot be read.

    {"verdict": "finished"|"holding"|"unsure", "score": float, "slope_st_s": float,
     "decay": float, "voiced_frames": int}
    """
    try:
        import numpy as np
        f0s, (times, rms) = _f0_track(pcm, sr)
        if not f0s:
            return None
        pairs = [(t, f) for t, f in zip(times, f0s) if f]
        if len(pairs) < MIN_FRAMES:
            return None                       # whispered, too short, or too noisy

        # Work in semitones: a 20Hz drop means something completely different at 90Hz
        # (a male voice) than at 220Hz (a female one), and a Hz slope would call the
        # same gesture "decisive" for one caller and "flat" for the other.
        t = np.array([p[0] for p in pairs], dtype="float64")
        st = np.array([12.0 * math.log2(p[1] / F0_MIN) for p in pairs], dtype="float64")
        span = t[-1] - t[0]
        if span <= 0.05:
            return None
        # Median-split slope, not least squares. One octave error — the failure mode
        # autocorrelation actually has — drags a regression line hard, and a single bad
        # frame must not be able to invent a falling contour.
        half = len(st) // 2
        slope = (float(np.median(st[half:])) - float(np.median(st[:half]))) / (span / 2)

        # Energy decay across the same window, as the corroborating signal.
        e = np.asarray(rms, dtype="float64")
        e = e[e > 0]
        decay = 1.0
        if e.size >= 4:
            first, last = float(np.median(e[:e.size // 2])), float(np.median(e[e.size // 2:]))
            decay = last / first if first > 0 else 1.0

        score = -slope + DECAY_WEIGHT * (1.0 - min(decay, 1.0))
        if score >= FINISHED_SCORE:
            verdict = "finished"
        elif score <= HOLDING_SCORE:
            verdict = "holding"
        else:
            verdict = "unsure"            # inside the dead band — say nothing
        return {"verdict": verdict, "score": round(score, 2),
                "slope_st_s": round(slope, 2),
                "decay": round(decay, 3), "voiced_frames": len(pairs)}
    except Exception as e:  # noqa: BLE001
        log.debug("prosody unreadable (%s) — the endpointer keeps its own window", e)
        return None


def window_scale(verdict: str | None) -> float:
    """How much to scale the silence window. 1.0 = leave it exactly as it was.

    Deliberately gentle, and asymmetric. Closing early on a wrong reading chops a
    caller's sentence in half — the failure this project has been tuned twice to avoid
    ("it breaks my long sentences") — while waiting too long merely costs a pause.
    """
    # The shortening is SMALLER than the lengthening, deliberately. The first version
    # had 0.75/1.20, which cut by more than it waited and contradicted the paragraph
    # above; a test caught it.
    if verdict == "finished":
        return 0.80
    if verdict == "holding":
        return 1.25
    return 1.0               # "unsure", or nothing readable: behave exactly as before


# MEASURED, `python scripts/bench_prosody.py`. `say` is a synthesiser and its prosody is
# a model of a speaker, not a speaker — the same caveat that has already made one
# threshold in this project wrong by more than two-fold. These are the reason the
# thresholds sit where they do, not proof they are right for a real microphone.
#
#   clip                 expected    score  slope st/s   decay   verdict
#   finished-name        finished     10.2        -6.6    0.64   unsure    (dead band)
#   finished-booking     finished      8.1        -7.9    0.97   unsure    (dead band)
#   finished-short       finished     36.4       -27.8    0.14   finished
#   wh-question          finished     32.2       -25.2    0.30   finished
#   cut-name             holding       7.0        -6.7    0.96   unsure    (dead band)
#   cut-booking          holding       5.1        -2.5    0.74   unsure    (dead band)
#   cut-number           holding       6.8        -4.3    0.75   unsure    (dead band)
#   cut-time             holding       3.2        -2.5    0.93   holding
#   yesno-question       holding       3.1        -0.9    0.78   holding
#
#   4 acted on and correct · 5 declined · 0 WRONG · 1.6ms per call
#
# Read the two rows that justify the whole design: finished-name at -6.6 st/s and
# cut-name at -6.7 are the SAME slope and opposite answers, and finished-booking's
# decay of 0.97 is higher than three of the four unfinished clips. Either signal alone
# would be a coin toss on those. Combined, the finished clips score 8.1-36.4 and the
# unfinished 3.1-7.0 — and the dead band between 5 and 12 is what stops the 8.1/7.0
# pair, one point apart, from being decided by an arithmetic accident.
#
# A yes/no question reads as "holding" (score 3.1). That is a real limitation and it is
# harmless: questions are already handled by the words — `looks_incomplete` returns
# False on a trailing "?" — so the worst it costs is a slightly longer pause before a
# question is answered. Rise detection is deliberately NOT claimed here; a capability
# that cannot be demonstrated should not be in the API.
