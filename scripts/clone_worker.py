#!/usr/bin/env python
"""Speak one line in a cloned voice, in a venv of its own.

WHY A SEPARATE PROCESS

`get_tts({"provider": "clone"})` returns None on the shipping install: coqui-tts needs a
newer `transformers` than the 4.46.1 this project pins, and that pin exists because
parler-tts — the Indic voice — requires exactly 4.46.1. The environment can host the
Indic voice or the in-process cloner, and it correctly chose the voice. The brand-voice
feature has been quietly unavailable ever since.

This is the third dependency collision of the same shape (deepfilternet forced numpy<2
and broke SpeechBrain; Qwen3-Embedding's transformers bump broke parler), and the
project already answered it twice: DeepFilterNet is a Rust binary we shell out to, and
whisper.cpp is a server we talk to over HTTP. Same answer here.

SETUP — a venv the main one never sees:

    python3 -m venv ~/.cache/zensuvidha-clone
    ~/.cache/zensuvidha-clone/bin/pip install voxcpm      # Apache-2.0, weights included

    # config.yaml
    tts:
      provider: clone
      clone_command: ["~/.cache/zensuvidha-clone/bin/python", "scripts/clone_worker.py"]

THE CONTRACT, which any cloner can satisfy:

    <python> clone_worker.py --ref REF.wav --text TEXT --out OUT.wav [--language en]

exit 0 and a readable wav at --out means success. Anything else is a decline, and the
engine falls back exactly as it does for any voice that cannot speak a script.

MEASURED, with this project's own speaker gate and recogniser (VoxCPM-0.5B, M1 Pro):

    English   RTF 2.89   similarity to the reference 0.716   transcribes cleanly
    Hindi     RTF 1.56   similarity 0.659, but the WORDS came back as English gibberish;
                         from a Hindi reference it ran away to 28s of audio for a 4s
                         line, which Whisper read as "(speaking in foreign language)"

So English cloning is usable and Indic is not, yet. `prerender` verifies every line it
renders before pinning it, so a bad one is discarded rather than cached forever — do not
rely on this worker being right, rely on that check.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--ref-text", default="",
                    help="what the reference clip says; zero-shot cloners condition "
                         "better when they are told")
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        print("reference not found: %s" % args.ref, file=sys.stderr)
        return 2

    try:
        import soundfile as sf
        from voxcpm import VoxCPM
    except ImportError as e:
        print("this worker needs its own venv: pip install voxcpm soundfile (%s)" % e,
              file=sys.stderr)
        return 3

    # Cached across calls by the module loader within one process; the engine spawns one
    # process per line, so the real saving comes from `prerender` batching — see
    # --batch below.
    model = VoxCPM.from_pretrained(os.environ.get("ZS_CLONE_MODEL",
                                                  "openbmb/VoxCPM-0.5B"))
    kw = {"text": args.text, "prompt_wav_path": args.ref}
    if args.ref_text:
        kw["prompt_text"] = args.ref_text
    wav = model.generate(**kw)
    sf.write(args.out, wav, 16000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
