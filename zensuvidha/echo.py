"""Server-side echo suppression — hearing ourselves is not the caller talking.

Why this has to exist
---------------------
Every scrap of echo handling in this system comes from the browser: `getUserMedia`'s
`echoCancellation: true`. That is genuinely good AEC, and it is also the single reason
the barge-in logic works at all. A telephony transport has none of it — a carrier hands
you the far-end audio with your own voice mixed in, and nothing has cancelled anything.

Without this, the first thing the agent does on a real phone line is interrupt itself:
it speaks, hears its own voice come back, treats that as barge-in, stops, then treats
the tail as a new turn and answers it. That loop is the blocker between this being a
browser demo and being reachable on a phone number.

What this is, honestly
----------------------
**Echo suppression, not adaptive cancellation.** A real AEC estimates the room's
impulse response with an adaptive filter (NLMS and friends) and subtracts a predicted
echo sample by sample. That is a bigger, more delicate piece of engineering, and getting
it wrong degrades the caller's own speech — which is the worst failure this codebase has.

This does something narrower and much harder to get wrong: it keeps a reference of what
we PLAYED, and when a microphone frame is well explained by that reference at some lag,
it refuses the frame instead of subtracting from it.

    real AEC          y[n] - ŷ[n]   subtract a prediction, keep the residual
    this              keep or drop  a whole frame, on the evidence

Refusing a frame can only ever lose audio that was mostly our own echo. Subtracting
badly can corrupt audio that was mostly the caller. Given the choice, and given that
the browser path already has real AEC, the conservative one is right — and it composes
with a proper AEC later rather than blocking it.

The measurement it uses
-----------------------
Normalised cross-correlation between the mic frame and the recent reference, over a
window of plausible lags. Normalised, so it responds to SHAPE rather than level: a quiet
echo of our own sentence still correlates ~1.0, while a loud unrelated voice does not.
That is exactly the discrimination a level gate cannot make, and it is why the existing
client-side level guard needed the browser's AEC underneath it.
"""
from __future__ import annotations

import logging

log = logging.getLogger("zensuvidha.echo")

SR = 16000

# How much history to keep. Echo arrives within a few hundred ms on any sane path, but
# the reference also has to be long enough to SLIDE the comparison window across — with
# one second, a one-second frame left no room to search and every delayed echo passed.
# 2s costs 128KB per call.
REFERENCE_S = 2.0

# Above this, the frame is explained by our own output. Chosen well clear of the values
# unrelated speech reaches: two different people saying different words correlate around
# 0.1-0.3, an attenuated copy of the same signal sits near 1.0. The gap is wide, which
# is the whole reason to measure shape rather than level.
ECHO_CORRELATION = 0.62

# Below this the frame is too quiet to judge, and near-silence correlates with anything.
MIN_RMS = 0.004

# The comparison window slid across the reference. Long enough to carry the shape of a
# syllable or two, short enough to leave room to search inside REFERENCE_S.
WINDOW_S = 0.35


class EchoSuppressor:
    """Per-call. Holds what we played and judges what comes back.

    Not thread-safe by design: one call, one WebSocket, one event loop. Sharing an
    instance between calls would let one caller's echo silence another's speech.
    """

    def __init__(self, sr: int = SR, enabled: bool = True):
        self.sr = sr
        self.enabled = enabled
        self._ref = None            # rolling window of what we most recently played
        self._recurring = []        # short clips the CLIENT may play at any moment
        self._played_s = 0.0
        self.suppressed = 0         # frames refused, for the inspector
        self.judged = 0

    # ---- what we played --------------------------------------------------
    def note_output(self, samples) -> None:
        """Record audio we are about to play, so it can be recognised coming back."""
        if not self.enabled:
            return
        try:
            import numpy as np
            x = np.asarray(samples, dtype="float32").reshape(-1)
            if not x.size:
                return
            keep = int(REFERENCE_S * self.sr)
            self._ref = x[-keep:] if self._ref is None else \
                np.concatenate([self._ref, x])[-keep:]
            self._played_s += x.size / self.sr
        except Exception as e:  # noqa: BLE001
            log.debug("echo: could not record output (%s)", e)

    def note_recurring(self, samples) -> None:
        """Remember a short clip that may be played at ANY time, indefinitely.

        The rolling reference holds the last couple of seconds, which is right for
        speech we stream as we generate it. It is wrong for audio handed to the client
        once and played later on its own schedule — the backchannel is pre-loaded at
        greeting time and murmured minutes into the call, by which point the ring buffer
        has long forgotten it. Measured: recognised right after preload, invisible ten
        seconds later, which is exactly when it actually plays.
        """
        if not self.enabled:
            return
        try:
            import numpy as np
            x = np.asarray(samples, dtype="float32").reshape(-1)
            # Bounded on purpose. These are murmurs and greetings, not sentences; a
            # growing list would turn every frame into a linear scan of the whole call.
            if x.size and len(self._recurring) < 4:
                self._recurring.append(x[:int(2.0 * self.sr)])
        except Exception as e:  # noqa: BLE001
            log.debug("echo: could not record a recurring clip (%s)", e)

    def note_recurring_wav(self, raw: bytes, decode) -> None:
        if not self.enabled or not raw:
            return
        try:
            data = decode(raw)
            if data is not None:
                self.note_recurring(data)
        except Exception as e:  # noqa: BLE001
            log.debug("echo: could not decode a recurring clip (%s)", e)

    def note_output_wav(self, raw: bytes, decode) -> None:
        """Same, from an encoded frame. `decode` is injected so this module stays free
        of soundfile — the audio front-end already owns that dependency."""
        if not self.enabled or not raw:
            return
        try:
            data = decode(raw)
            if data is not None:
                self.note_output(data)
        except Exception as e:  # noqa: BLE001
            log.debug("echo: could not decode our own output (%s)", e)

    # ---- what came back --------------------------------------------------
    def is_echo(self, samples) -> tuple[bool, float]:
        """Is this microphone frame mostly our own voice coming back?

        Returns (verdict, correlation). FAILS OPEN in every uncertain case — no
        reference yet, too quiet to judge, numpy missing, anything raised. A false
        "echo" silences the caller, which is strictly worse than answering our own
        tail once, and this file must never become the reason somebody cannot be heard.
        """
        self.judged += 1
        if not self.enabled or (self._ref is None and not self._recurring):
            return False, 0.0
        try:
            import numpy as np
            x = np.asarray(samples, dtype="float32").reshape(-1)
            if x.size < self.sr // 50:                 # < 20ms — nothing to correlate
                return False, 0.0
            # NaN/inf reaches here from a malformed frame. It already failed open via
            # the exception path, but only after numpy emitted a warning per frame —
            # which on a bad line is a warning per turn, drowning the log that would
            # explain the bad line.
            if not np.isfinite(x).all():
                return False, 0.0
            if float(np.sqrt(np.mean(x * x))) < MIN_RMS:
                return False, 0.0                      # near-silence matches anything

            # Every reference we might be hearing: the rolling window, plus anything
            # the client holds and can play on its own schedule.
            refs = [r for r in ([self._ref] + self._recurring)
                    if r is not None and r.size >= self.sr // 50]
            if not refs:
                return False, 0.0

            # Slide a WINDOW of the frame across the whole reference, rather than
            # aligning the frame's tail at a handful of offsets. The first version did
            # the latter and scored a 120ms-delayed echo at 0.371 — it passed straight
            # through, which is the one case that matters, because echo is ALWAYS
            # delayed. When the frame and the reference were the same length there was
            # simply no room left to slide and the lag search never ran at all.
            win = min(x.size, int(WINDOW_S * self.sr))
            usable = [r for r in refs if r.size >= win]
            if not usable:
                return False, 0.0
            # Probe the LOUDEST window of the frame, not the first one. A 300ms-delayed
            # echo starts with 300ms of near-silence, so probing the head correlated at
            # 0.451 and passed straight through — the delay simply moved the signal out
            # of the window being compared. Picking the window by energy makes the
            # measurement independent of where in the frame the echo begins.
            e2 = np.cumsum(np.concatenate([[0.0], (x.astype("float64")) ** 2]))
            energy = e2[win:] - e2[:-win]
            probe = x[int(np.argmax(energy)):][:win] if energy.size else x[:win]
            if probe.size < win:
                probe = x[:win]
            p = probe - probe.mean()
            pn = float(np.linalg.norm(p))
            if pn == 0.0:
                return False, 0.0

            # Normalised cross-correlation at every offset, computed with sliding sums
            # instead of a Python loop: O(n log n) rather than O(n·w), which keeps this
            # off the critical path even on a long utterance.
            best = 0.0
            for ref in usable:
                r64 = ref.astype("float64")
                cs = np.cumsum(np.concatenate([[0.0], r64]))
                cs2 = np.cumsum(np.concatenate([[0.0], r64 ** 2]))
                sums = cs[win:] - cs[:-win]
                sumsq = cs2[win:] - cs2[:-win]
                means = sums / win
                var = np.maximum(sumsq - win * means * means, 1e-12)
                norms = np.sqrt(var)
                # `p` is zero-mean, so sum(seg·p) already equals sum((seg-mean)·p) —
                # the per-window mean cancels out of the numerator.
                num = np.correlate(r64, p.astype("float64"), mode="valid")
                ncc = np.abs(num) / (norms * pn + 1e-12)
                if ncc.size:
                    best = max(best, float(ncc.max()))
                if best >= 0.995:
                    break

            if best >= ECHO_CORRELATION:
                self.suppressed += 1
                log.info("echo: refusing a frame that is %.2f correlated with our own "
                         "output", best)
                return True, best
            return False, best
        except Exception as e:  # noqa: BLE001
            log.debug("echo: could not judge a frame (%s) — accepting it", e)
            return False, 0.0

    def reset(self) -> None:
        """Forget the reference. Called when the caller barges in and we stop playing,
        so a stale tail cannot explain away the words they interrupted us with.

        Recurring clips are KEPT: the client still holds them and can still play one,
        so forgetting them here would reopen the hole this suppressor exists to close.
        """
        self._ref = None
