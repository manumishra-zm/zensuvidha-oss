#!/usr/bin/env python
"""Does the pitch contour actually separate "finished" from "paused mid-sentence"?

The claim prosodic endpointing rests on is that a speaker's pitch falls through the end
of a declarative utterance and stays level when they stop mid-thought. That is textbook,
and textbook is not the same as measurable on the audio this system receives — so this
prints the numbers and lets them be argued with.

    python scripts/bench_prosody.py                 # synthetic, from macOS `say`
    python scripts/bench_prosody.py --dir clips/    # your own: *.done.wav / *.cut.wav

`say` is a synthesiser: its prosody is a MODEL of a speaker, not a speaker. It gets the
declination right because that is what it was built to imitate, which makes it a fair
test of the detector's arithmetic and a poor test of its thresholds. The speaker gate's
threshold was calibrated this way and proved wrong on a real microphone by more than
two-fold. Re-run this with --dir on real recordings before trusting the numbers.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zensuvidha.prosody import finality  # noqa: E402

# (label, expected, voice, full sentence, prefix where the caller would still be talking)
#
# THE FIXTURE DESIGN IS THE WHOLE EXPERIMENT, and the obvious version is wrong. Asking
# `say` to speak "My name is" synthesises it AS A COMPLETE SENTENCE, so it gets a
# textbook final fall — the exact contour we are trying to distinguish it from. Measured
# that way the detector scored 3/8 and called every fragment "falling", which is the
# fixture being wrong, not the arithmetic.
#
# So a "cut" clip is made by speaking the WHOLE sentence and truncating it where the
# prefix ends. That audio is genuinely mid-utterance: the pitch is still up, because the
# speaker has not reached the end yet. It is the same thing a caller's microphone
# delivers when they stop to think.
#
# Note also that only YES/NO questions rise in English. A wh-question ("what are your
# charges?") falls, like a statement — expecting otherwise was a mistake in the first
# version of this file, not a defect in the detector.
CASES = [
    ("finished-name",    "finished", "Karen",  "My name is Manu Mishra.", None),
    ("finished-booking", "finished", "Karen",  "I would like to book an appointment.", None),
    ("finished-short",   "finished", "Daniel", "Yes that is correct.", None),
    ("wh-question",      "finished", "Daniel", "What are your consultation charges?", None),
    ("cut-name",         "holding",   "Karen",  "My name is Manu Mishra.", "My name is"),
    ("cut-booking",      "holding",   "Karen",  "I would like to book an appointment.",
                                              "I would like to book an"),
    ("cut-number",       "holding",   "Daniel", "My mobile number is nine eight two zero.",
                                              "My mobile number is"),
    ("cut-time",         "holding",   "Karen",  "I need an appointment tomorrow morning.",
                                              "I need an appointment"),
    ("yesno-question",   "holding",  "Karen",  "Are you open on Sunday?", None),
]


def say_wav(voice: str, text: str, out: str):
    aiff = out.replace(".wav", ".aiff")
    r = subprocess.run(["say", "-v", voice, "-o", aiff, text], capture_output=True)
    if r.returncode != 0:
        return False
    subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    aiff, out], capture_output=True, check=True)
    os.remove(aiff)
    return True


def truncate_to(full: str, probe: str, out: str):
    """Cut `full` at the spoken length of `probe`, minus its trailing silence.

    The probe's own tail is silence plus the release of its final word; keeping that
    would put a synthesised ending back into the clip, which is what this whole
    construction exists to avoid.
    """
    import numpy as np
    import soundfile as sf
    a, sr = sf.read(full)
    b, _ = sf.read(probe)
    a = np.asarray(a, dtype="float32").reshape(-1)
    b = np.asarray(b, dtype="float32").reshape(-1)
    thresh = float(np.max(np.abs(b))) * 0.02
    voiced = np.nonzero(np.abs(b) > thresh)[0]
    if voiced.size == 0:
        return None
    n = int(voiced[-1])
    if n < int(0.3 * sr) or n >= a.size:
        return None
    sf.write(out, a[:n], sr, subtype="PCM_16")
    return out


def measure(path: str):
    import soundfile as sf
    data, sr = sf.read(path)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    t0 = time.time()
    out = finality(data, sr)
    return out, (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="real clips: NAME.done.wav / NAME.cut.wav")
    args = ap.parse_args()

    rows = []
    if args.dir:
        for p in sorted(glob.glob(os.path.join(args.dir, "*.wav"))):
            base = os.path.basename(p)
            expect = ("finished" if ".done." in base else
                      "holding" if ".cut." in base else None)
            if expect is None:
                print("  (skipping %s — name it .done.wav or .cut.wav)" % base)
                continue
            rows.append((base, expect, p))
    else:
        tmp = tempfile.mkdtemp(prefix="zs_prosody_")
        for label, expect, voice, text, prefix in CASES:
            p = os.path.join(tmp, label + ".wav")
            if not say_wav(voice, text, p):
                continue
            if prefix:
                # Speak the prefix only to learn how long it takes, then cut the FULL
                # sentence there. The audio kept is mid-utterance, not a second
                # synthesis with its own ending.
                probe = os.path.join(tmp, label + ".probe.wav")
                if not say_wav(voice, prefix, probe):
                    continue
                p = truncate_to(p, probe, os.path.join(tmp, label + ".cut.wav"))
                if p is None:
                    continue
            rows.append((label, expect, p))
        print("Synthetic `say` clips. The arithmetic is being tested here, NOT the\n"
              "thresholds — see the module docstring.\n")

    if not rows:
        print("nothing to measure")
        return

    print("%-20s %-9s %7s %10s %7s %6s %7s  %s"
          % ("clip", "expected", "score", "slope st/s", "decay", "ms", "verdict", ""))
    hits = quiet = 0
    for label, expect, path in rows:
        out, ms = measure(path)
        if out is None:
            print("%-20s %-9s %7s %10s %7s %6.1f %7s" % (label, expect, "-", "-", "-", ms,
                                                         "unreadable"))
            quiet += 1
            continue
        v = out["verdict"]
        ok = v == expect
        hits += ok
        quiet += v == "unsure"
        print("%-20s %-9s %7.1f %10.1f %7.2f %6.1f %7s  %s"
              % (label, expect, out["score"], out["slope_st_s"], out["decay"], ms, v,
                 "" if ok else ("(dead band — no opinion)" if v == "unsure" else "<-- WRONG")))
    wrong = len(rows) - hits - quiet
    print("\n%d acted on and correct, %d declined (dead band), %d WRONG."
          % (hits, quiet, wrong))
    print("Wrong is the only number that matters: the dead band exists so that a "
          "borderline\nclip changes nothing, and a clip it does act on is one it is "
          "well clear about.")
    print("A detector that cannot separate these on SYNTHETIC speech, where the "
          "contour is\nexaggerated, has no chance on a real one — this is a floor, "
          "not a result.")


if __name__ == "__main__":
    main()
