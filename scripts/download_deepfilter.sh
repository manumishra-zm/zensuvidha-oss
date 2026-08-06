#!/usr/bin/env bash
# Fetch the DeepFilterNet standalone binary so the UI's "DeepFilter" toggle works.
#
# OPTIONAL, and off by default. Measured on this codebase, DeepFilterNet makes
# transcription WORSE (WER 0.10 vs 0.00 raw) and the speaker gate worse (0.596 vs
# 0.675) — it won 0 of 10 conditions in an SNR sweep from +20dB down to -5dB. It is
# wired in so you can hear the difference yourself rather than take that on trust.
#
#     bash scripts/download_deepfilter.sh
#
# ~28MB, MIT/Apache-2.0, model weights unrestricted. Nothing is installed into the
# Python environment — this is deliberately the Rust binary, because
# `pip install deepfilternet` pins numpy<2 and imports torchaudio.backend.common
# (removed in modern torchaudio), which breaks SpeechBrain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models"
VERSION="0.5.6"
BASE="https://github.com/Rikorose/DeepFilterNet/releases/download/v${VERSION}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  ASSET="deep-filter-${VERSION}-aarch64-apple-darwin" ;;
  Darwin-x86_64) ASSET="deep-filter-${VERSION}-x86_64-apple-darwin" ;;
  Linux-aarch64) ASSET="deep-filter-${VERSION}-aarch64-unknown-linux-gnu" ;;
  Linux-x86_64)  ASSET="deep-filter-${VERSION}-x86_64-unknown-linux-musl" ;;
  *) echo "No prebuilt binary for $(uname -s)-$(uname -m)." >&2
     echo "Build from source: https://github.com/Rikorose/DeepFilterNet" >&2; exit 1 ;;
esac

mkdir -p "$DEST"
OUT="$DEST/deep-filter"

if [ -f "$OUT" ] && "$OUT" --version >/dev/null 2>&1; then
  echo "  ✓ already installed: $("$OUT" --version 2>&1 | head -1)"
  exit 0
fi

echo "  ↓ $ASSET"
curl -fL --retry 3 --progress-bar -o "$OUT.part" "$BASE/$ASSET"
chmod +x "$OUT.part"
xattr -d com.apple.quarantine "$OUT.part" 2>/dev/null || true   # macOS Gatekeeper

if ! "$OUT.part" --version >/dev/null 2>&1; then
  rm -f "$OUT.part"
  echo "  ✗ the downloaded binary would not run — not installing." >&2
  exit 1
fi
mv "$OUT.part" "$OUT"

echo
echo "Done: $("$OUT" --version 2>&1 | head -1)"
echo "Restart the server, then use the 'DeepFilter' toggle in the UI to A/B it live."
