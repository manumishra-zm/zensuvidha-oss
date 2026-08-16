#!/usr/bin/env python
"""Compare speech-recognition backends on the SAME audio: latency and word error rate.

Why this exists
---------------
STT is the dominant cost on the audio path — measured 1070ms at 3s of speech against
66ms for isolation and 34ms for the speaker gate — and every CPU-side lever has been
tried and rejected (more CTranslate2 threads measured WORSE on this M1; greedy decoding
was not reliably faster). Adding a backend is only worth it if it is measurably better,
and "measurably" has a specific meaning in this codebase: an earlier claim that MLX
would be 1.3-1.5x faster turned out to be 2% when it was actually run.

Usage
-----
    # synthetic clips from macOS `say` — fine for LATENCY, indicative only for accuracy
    python scripts/bench_stt.py

    # the measurement that actually matters: your own recordings
    python scripts/bench_stt.py --dir recordings/
        …where recordings/ holds foo.wav beside foo.txt containing what was said.

Every threshold in this repo that was tuned on `say` output has needed correcting
against real audio at least once — the speaker gate's was wrong by more than 2x. Treat
the synthetic numbers here the same way.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zensuvidha.config import load_config  # noqa: E402

SAY_CLIPS = [
    ("en", "Karen", "I would like to book an appointment with the skin doctor tomorrow morning"),
    ("en", "Karen", "What are your consultation charges and are you open on Sunday"),
    ("en", "Daniel", "My mobile number is nine eight two zero four two nine zero five seven"),
    ("hi", "Lekha", "मुझे कल सुबह डॉक्टर से मिलना है"),
]


def _words(text: str):
    """Split on separators, never on \\w — that drops Indic combining marks."""
    return [w for w in re.split(r"[\s.,!?;:।॥\-\"'()]+", (text or "").lower()) if w]


def wer(ref: str, hyp: str) -> float:
    """Levenshtein over words, normalised by the reference length."""
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def make_say_clips(outdir: str):
    """Generate fixtures with macOS `say`. Returns [(wav, reference, language)]."""
    os.makedirs(outdir, exist_ok=True)
    out = []
    for i, (lang, voice, text) in enumerate(SAY_CLIPS):
        wav = os.path.join(outdir, "clip%d.wav" % i)
        if not os.path.isfile(wav):
            aiff = wav.replace(".wav", ".aiff")
            r = subprocess.run(["say", "-v", voice, "-o", aiff, text],
                               capture_output=True)
            if r.returncode != 0:
                print("  (skipped %s — voice %r unavailable)" % (lang, voice))
                continue
            subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                            "-c", "1", aiff, wav], capture_output=True, check=True)
            os.remove(aiff)
        out.append((wav, text, lang))
    return out


def load_dir(d: str):
    """Real recordings: foo.wav beside foo.txt holding what was actually said."""
    out = []
    for wav in sorted(glob.glob(os.path.join(d, "*.wav"))):
        txt = wav[:-4] + ".txt"
        if not os.path.isfile(txt):
            print("  (skipping %s — no matching .txt reference)" % os.path.basename(wav))
            continue
        out.append((wav, open(txt, encoding="utf-8").read().strip(), None))
    return out


def bench(name: str, stt, clips, runs: int):
    if stt is None:
        print("\n%-16s unavailable" % name)
        return
    # Warm first, always. An earlier "2.1x faster" reading in this project was a
    # cold-start artifact, and the first Qwen3-Embedding number was off by 12x for the
    # same reason.
    stt.transcribe(clips[0][0])
    rows, tot_wer, tot_ms = [], 0.0, 0.0
    runs = max(1, int(runs))      # 0 would leave `text` unbound and `best` None below
    for wav, ref, lang in clips:
        best, text = None, ""
        for _ in range(runs):
            t0 = time.time()
            text, _det, _p = stt.transcribe(wav, language=lang)
            ms = (time.time() - t0) * 1000
            best = ms if best is None else min(best, ms)
        e = wer(ref, text)
        rows.append((os.path.basename(wav), best, e, text))
        tot_wer += e
        tot_ms += best
    print("\n%s" % name)
    print("  %-12s %9s  %6s   %s" % ("clip", "ms", "WER", "transcript"))
    for f, ms, e, text in rows:
        print("  %-12s %9.0f  %6.2f   %s" % (f, ms, e, (text or "")[:52]))
    print("  %-12s %9.0f  %6.2f   <- mean" % ("", tot_ms / len(rows), tot_wer / len(rows)))
    return tot_ms / len(rows), tot_wer / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory of real .wav recordings with .txt references")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per clip (best wins)")
    args = ap.parse_args()

    cfg = load_config()
    stt_cfg = dict(cfg["stt"])

    if args.dir:
        clips = load_dir(args.dir)
        print("Real recordings: %d clips from %s" % (len(clips), args.dir))
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        clips = make_say_clips(os.path.join(here, "..", "models", ".bench_clips"))
        print("Synthetic `say` clips: %d. LATENCY is meaningful; accuracy is not — "
              "\n  every threshold tuned on `say` output in this repo has needed "
              "correcting against real audio." % len(clips))
    if not clips:
        print("nothing to measure")
        return

    from zensuvidha.stt import FasterWhisperSTT, WhisperCppSTT
    results = {}
    try:
        results["faster-whisper (CPU int8)"] = bench(
            "faster-whisper (CPU int8)", FasterWhisperSTT(stt_cfg), clips, args.runs)
    except Exception as e:  # noqa: BLE001
        print("faster-whisper unavailable: %s" % e)
    try:
        results["whisper.cpp (Metal)"] = bench(
            "whisper.cpp (Metal)", WhisperCppSTT(stt_cfg), clips, args.runs)
    except Exception as e:  # noqa: BLE001
        print("\nwhisper.cpp unavailable: %s" % e)

    good = {k: v for k, v in results.items() if v}
    if len(good) == 2:
        (n1, (m1, w1)), (n2, (m2, w2)) = good.items()
        print("\n%s vs %s: %.2fx on latency, WER %.2f vs %.2f"
              % (n2, n1, m1 / m2 if m2 else 0, w2, w1))
        print("A speedup that costs accuracy is not a speedup — this path already "
              "\nrejected DeepFilterNet for exactly that trade.")


if __name__ == "__main__":
    main()
