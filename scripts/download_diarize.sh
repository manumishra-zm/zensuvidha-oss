#!/usr/bin/env bash
# Vendor the speaker-diarization models used for SEGMENT-LEVEL speaker gating.
#
# What this buys you, measured on this codebase: the whole-utterance speaker gate
# REJECTS the caller's own turn when a colleague speaks in a gap (similarity drops
# 0.867 -> 0.34). With these models the utterance is split by speaker and only the
# caller's parts are transcribed, so one stray sentence no longer discards everything
# they said — or lands a stranger's words in the transcript.
#
#     bash scripts/download_diarize.sh      # ~45MB total, then set stt.diarize: true
#
# Fully offline afterwards. ONNX only — no torch, no GPU.
#   segmentation  pyannote-segmentation-3.0  MIT         ~7MB
#   embedding     3D-Speaker ERes2Net        Apache-2.0  ~38MB
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/diarize"
REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_TGZ="$REL/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB="$REL/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

mkdir -p "$DEST"

if [ -f "$DEST/segmentation.onnx" ] && [ -f "$DEST/embedding.onnx" ]; then
  echo "  ✓ already installed in models/diarize/"
  exit 0
fi

if ! "$ROOT/.venv/bin/python" -c "import sherpa_onnx" 2>/dev/null; then
  echo "  ! sherpa-onnx is not installed. Run:  pip install sherpa-onnx" >&2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "  ↓ segmentation (pyannote-segmentation-3.0, MIT)"
curl -fsSL --retry 3 -o "$tmp/seg.tar.bz2" "$SEG_TGZ"
tar xjf "$tmp/seg.tar.bz2" -C "$tmp"
found="$(find "$tmp" -name 'model.onnx' | head -1)"
[ -n "$found" ] || { echo "  ✗ segmentation model not found in the archive" >&2; exit 1; }
cp "$found" "$DEST/segmentation.onnx"

echo "  ↓ speaker embedding (3D-Speaker ERes2Net, Apache-2.0)"
curl -fsSL --retry 3 -o "$DEST/embedding.onnx.part" "$EMB"
if [ "$(wc -c <"$DEST/embedding.onnx.part")" -lt 5000000 ]; then
  rm -f "$DEST/embedding.onnx.part"
  echo "  ✗ embedding model looked truncated — not installing." >&2; exit 1
fi
mv "$DEST/embedding.onnx.part" "$DEST/embedding.onnx"

echo
echo "Done. Now set in config.yaml:"
echo "    stt:"
echo "      diarize: true"
echo "      speaker_gate: true      # diarization needs an enrolled voiceprint"
