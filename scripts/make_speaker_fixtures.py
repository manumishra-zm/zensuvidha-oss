#!/usr/bin/env python
"""Generate DISTINCT speakers, to test the parts of this system that judge identity.

The problem this solves
-----------------------
Every threshold outside the speaker gate is calibrated on macOS `say` voices, and one of
them was already proved wrong on a real microphone by more than two-fold. The isolation
tests, the speaker gate and the prosody bench all need *several different people saying
the same words*, and `say` gives you a handful of synthetic voices that are far more
distinguishable than real speakers — which is exactly the wrong bias for a gate whose
job is telling two real people apart.

A voice cloner produces as many distinct speakers as you have reference clips, saying
whatever you like, in a controlled way. That is a much better fixture source than a
fixed set of system voices, and unlike real recordings it needs nobody's consent.

    python scripts/make_speaker_fixtures.py --refs refs/ --out fixtures/

`refs/` holds one wav per speaker (10-15s each); `fixtures/` gets every line spoken by
every speaker, plus the mixed clips the isolation tests actually need.

Uses whatever cloner the project is configured with — `tts.provider: clone` and
`requirements-clone.txt`. Nothing here knows which one.

MEASURED, and read this before trusting the output
--------------------------------------------------
Checked with this project's OWN speaker gate and recogniser, VoxCPM-0.5B on an M1:

    English clone   similarity to reference 0.716   transcribes cleanly
    Hindi clone     similarity 0.659, but the WORDS came back as English gibberish
                    — and from a Hindi reference it ran away to 28s of audio for a
                    4s sentence, which Whisper read as "(speaking in foreign language)"

So: **English fixtures from a cloner are usable; Indic ones are not, yet.** For the
Indic side, `say` remains the honest fallback and real recordings remain the real
answer. Verify anything this produces the same way — the check is at the bottom of
this file and takes seconds.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lines chosen for what they exercise, not for variety: a phone number for the digit
# grounding, a name for the slot filter, a question for the prosody rise case, and one
# long enough to have a middle for the endpointer.
LINES = [
    "My name is Manu Mishra.",
    "My mobile number is nine eight two zero four two nine zero five seven.",
    "Are you open on Sunday?",
    "I would like to book an appointment with the skin doctor tomorrow morning.",
    "Yes that is correct.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="directory of reference wavs, one per speaker")
    ap.add_argument("--out", required=True, help="where to write the fixtures")
    ap.add_argument("--mix", action="store_true",
                    help="also write two-speaker clips, which is what isolation needs")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    from zensuvidha.config import load_config
    from zensuvidha.tts import get_tts

    refs = sorted(glob.glob(os.path.join(args.refs, "*.wav")))
    if not refs:
        print("no reference wavs in %s" % args.refs)
        return
    os.makedirs(args.out, exist_ok=True)

    cfg = dict(load_config()["tts"])
    cfg["provider"] = "clone"
    made = {}
    for ref in refs:
        who = os.path.splitext(os.path.basename(ref))[0]
        cfg["reference"] = ref
        voice = get_tts(cfg)
        if voice is None:
            print("no cloner configured — see requirements-clone.txt")
            return
        made[who] = []
        for i, line in enumerate(LINES):
            audio = voice.synth(line)
            if not audio:
                print("  %s line %d: the voice declined it" % (who, i))
                continue
            path = os.path.join(args.out, "%s_%d.wav" % (who, i))
            open(path, "wb").write(audio)
            made[who].append(path)
            print("  %s_%d.wav  %s" % (who, i, line[:44]))

    if args.mix and len(made) >= 2:
        # A stranger speaking in a GAP is the shape that actually breaks things —
        # measured on this codebase, it drags whole-clip similarity 0.867 -> 0.34 and
        # the CALLER's own turn gets rejected. True simultaneous overlap is survivable.
        (a_name, a), (b_name, b) = list(made.items())[:2]
        if a and b:
            xa, sr = sf.read(a[3] if len(a) > 3 else a[0])
            xb, _ = sf.read(b[0])
            gap = np.zeros(int(0.4 * sr), dtype="float32")
            mixed = np.concatenate([np.asarray(xa, dtype="float32"), gap,
                                    np.asarray(xb, dtype="float32")])
            out = os.path.join(args.out, "mixed_%s_then_%s.wav" % (a_name, b_name))
            sf.write(out, mixed, sr)
            print("  %s  (caller, 0.4s gap, stranger)" % os.path.basename(out))

    print("\nVerify before trusting these — a clone that scores below the gate's own "
          "threshold\nis not a fixture, it is a second bug:")
    print("  python - <<'EOF'\n"
          "  from zensuvidha.config import load_config\n"
          "  from zensuvidha.speaker import get_speaker_gate\n"
          "  from zensuvidha.stt import get_stt\n"
          "  cfg = load_config()['stt']; g = get_speaker_gate(cfg); s = get_stt(cfg)\n"
          "  ref = g.embed(open('refs/<speaker>.wav','rb').read())\n"
          "  v   = g.embed(open('%s/<speaker>_0.wav','rb').read())\n"
          "  print('similarity', g.similarity(ref, v))   # want >= the gate threshold\n"
          "  print('words', s.transcribe('%s/<speaker>_0.wav')[0])\n"
          "  EOF" % (args.out, args.out))


if __name__ == "__main__":
    main()
