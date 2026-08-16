#!/usr/bin/env python
"""Which loudness number should open a turn — full-band RMS, or the speech band?

`web/vad-worklet.js` computes BOTH on every 32ms hop and posts them:

    rms       the whole signal, 0 - Nyquist
    rmsBand   through a 300-3400Hz two-pole cascade

...and `handleFrame` has always gated on `rms`. So the filter that was written and
benchmarked is measured and thrown away, every frame, on every call. This script decides
whether that is a bug or a reprieve, because switching is NOT free: the energy gate is

    gate = max(noiseFloor * 4.0, 0.006)

and band-limiting lowers the signal's amplitude, so the ADAPTIVE half self-corrects
(noiseFloor tracks whatever it is fed) while the ABSOLUTE floor of 0.006 does not. Moving
to a quieter signal without moving that number makes the gate relatively deafer — which
is exactly the kind of silent tuning regression this codebase keeps finding late.

WHAT IS MEASURED. Separation: how much louder speech is than noise, under each gate.

    separation = rms(speech) / rms(noise)

A bigger number means the threshold has more room to sit between them, which is the
whole job of the gate. The comparison that matters is the RATIO of the two separations —
absolute levels differ between the gates by construction and say nothing on their own.

WHY NODE. The filter under test is the one that ships, run as it ships, rather than a
numpy port of it. The last time this cascade was measured, the ideal FFT brick-wall said
4.47x mean and the real two-pole filter delivered 2.79x — the reimplementation was
optimistic by 60%. So `vad-worklet.js` is loaded and executed here, and if it is edited
this bench follows it.

    python scripts/bench_vad.py              # synthetic noise + macOS `say` speech
    python scripts/bench_vad.py --dir clips/ # ...and your own *.wav alongside

The synthetic noise is honest — hiss, rumble and hum are well described by their spectra.
The SPEECH is the weak half: `say` is a synthesiser, and a real microphone in a real room
has already proved a calibrated threshold wrong here by more than two-fold. Re-run with
--dir on real recordings before moving a constant on the strength of this.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLET = os.path.join(ROOT, "web", "vad-worklet.js")
SR = 48000          # the worklet runs at the AudioContext rate; 48k is the common one
DUR = 3.0


# ---------------------------------------------------------------- the noise floor

def _rng(seed):
    return np.random.default_rng(seed)


def hiss(n, seed=1):
    """Broadband white noise — a bad line, a cheap preamp."""
    return _rng(seed).normal(0, 1, n)


def rumble(n, seed=2):
    """Fan / air-conditioning: energy concentrated well below the speech band. This is
    the shape the cascade should win on by the largest margin, and the one a phone in a
    hand produces most of (handling noise, wind, a pocket)."""
    x = _rng(seed).normal(0, 1, n)
    # one-pole lowpass, hard, around ~120Hz
    a = np.exp(-2 * np.pi * 120 / SR)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    return y


def hum(n, seed=3):
    """Mains hum: 50Hz and its harmonics."""
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * f * t) / k
               for k, f in enumerate((50, 100, 150, 200), start=1))


def traffic(n, seed=4):
    """Low-mid weighted broadband, the road outside the window."""
    x = _rng(seed).normal(0, 1, n)
    a = np.exp(-2 * np.pi * 500 / SR)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    return y


def hiss_high(n, seed=5):
    """Tape/preamp hiss above the speech band — where a lowpass should help."""
    x = _rng(seed).normal(0, 1, n)
    return np.diff(np.concatenate([[0.0], x]))     # crude highpass (differentiator)


NOISES = {"fan / AC rumble": rumble, "broadband hiss": hiss, "high hiss": hiss_high,
          "traffic": traffic, "mains hum": hum}


# ---------------------------------------------------------------- the speech

def say_speech(text="My mobile number is nine eight two one double four."):
    """macOS `say` → float32 @ SR. Returns None where `say` is unavailable."""
    if not shutil.which("say"):
        return None
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        aiff = os.path.join(d, "s.aiff")
        wav = os.path.join(d, "s.wav")
        try:
            subprocess.run(["say", "-o", aiff, text], check=True, timeout=60,
                           capture_output=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@%d" % SR,
                            aiff, wav], check=True, timeout=60, capture_output=True)
            data, sr = sf.read(wav)
        except Exception:      # noqa: BLE001
            return None
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return _resample(np.asarray(data, dtype="float64"), sr, SR)


def _resample(x, src, dst):
    if src == dst:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x)


def load_dir(path):
    import soundfile as sf
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.wav"))):
        data, sr = sf.read(f)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        out.append((os.path.basename(f), _resample(np.asarray(data, "float64"), sr, SR)))
    return out


# ------------------------------------------------- the filter, as it actually ships

DRIVER = r"""
const fs = require('fs');
%(worklet_src)s
const buf = fs.readFileSync(process.argv[1]);
const x = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const sampleRate = %(sr)d;
const hp = new Biquad('hp', 300, sampleRate), lp = new Biquad('lp', 3400, sampleRate);
let full = 0, band = 0;
for (let i = 0; i < x.length; i++) {
  const b = lp.step(hp.step(x[i]));
  full += x[i] * x[i];
  band += b * b;
}
console.log(JSON.stringify({full: Math.sqrt(full / x.length),
                            band: Math.sqrt(band / x.length)}));
"""


def _worklet_biquad_src():
    """The Biquad class out of the shipping worklet, without the processor around it.

    Cut at `class VadFramer`, which is the only part that needs AudioWorkletGlobalScope.
    If the class is renamed this raises here rather than silently benching nothing.
    """
    src = open(WORKLET, encoding="utf-8").read()
    if "class Biquad" not in src:
        raise SystemExit("vad-worklet.js no longer defines `class Biquad` — "
                         "this bench measures the shipping filter and cannot find it")
    return src[src.index("class Biquad"):src.index("class VadFramer")]


def levels(x):
    """(full-band RMS, speech-band RMS) for one signal, via the real worklet filter."""
    node = shutil.which("node")
    if not node:
        raise SystemExit("node is required: this bench runs the shipping JS filter")
    prog = DRIVER % {"worklet_src": _worklet_biquad_src(), "sr": SR}
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        f.write(np.asarray(x, dtype="float32").tobytes())
        path = f.name
    try:
        out = subprocess.run([node, "-e", prog, path], capture_output=True,
                             text=True, timeout=120)
        if out.returncode != 0:
            raise SystemExit("node failed: " + out.stderr)
        r = json.loads(out.stdout)
    finally:
        os.unlink(path)
    return r["full"], r["band"]


def norm(x, rms=0.05):
    """Scale to a fixed RMS so 'separation' compares gates, not levels."""
    x = np.asarray(x, dtype="float64")
    cur = float(np.sqrt((x ** 2).mean()))
    return x if cur <= 0 else x * (rms / cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="a directory of real *.wav speech recordings")
    args = ap.parse_args()

    n = int(DUR * SR)
    speech = []
    s = say_speech()
    if s is not None:
        speech.append(("say (synthetic)", s))
    if args.dir:
        speech.extend(load_dir(args.dir))
    if not speech:
        raise SystemExit("no speech to measure: `say` unavailable and no --dir given")

    print("Speech vs noise, both at equal RMS. Separation = speech / noise.")
    print("Higher is better; the RATIO of the two columns is the decision.\n")
    print("%-22s %-22s %10s %10s %8s" % ("speech", "noise", "full-band", "300-3400", "gain"))
    print("-" * 76)

    gains = []
    for sname, sig in speech:
        sf_full, sf_band = levels(norm(sig))
        for nname, fn in NOISES.items():
            nf_full, nf_band = levels(norm(fn(n)))
            sep_full = sf_full / nf_full if nf_full else float("inf")
            sep_band = sf_band / nf_band if nf_band else float("inf")
            gain = sep_band / sep_full if sep_full else float("nan")
            gains.append(gain)
            print("%-22s %-22s %10.2f %10.2f %7.2fx"
                  % (sname[:22], nname, sep_full, sep_band, gain))

    print("-" * 76)
    g = np.array(gains)
    print("mean gain from band-limiting: %.2fx   (min %.2fx, max %.2fx)"
          % (g.mean(), g.min(), g.max()))

    # The part that decides whether the switch is safe, not just whether it is better.
    print("\nWhat band-limiting does to the SPEECH level itself — this is what the")
    print("absolute floor (max(noiseFloor*4, 0.006) in index.html) has to move by:")
    for sname, sig in speech:
        f, b = levels(norm(sig))
        print("  %-22s full %.4f  band %.4f   ->  floor x %.2f"
              % (sname[:22], f, b, (b / f) if f else float("nan")))
    print("\nA gain above ~1.5x across the board argues for switching the gate to")
    print("rmsBand AND scaling 0.006 by the factor printed just above.")


if __name__ == "__main__":
    main()
