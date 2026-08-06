#!/usr/bin/env bash
# Vendor Silero VAD so the browser can tell speech from noise OFFLINE.
#
# This is OPTIONAL. Without it the UI uses an adaptive noise-floor energy VAD, which
# is already far better than the fixed threshold it replaced. With it you get a real
# speech/non-speech classifier, which is what stops a fan, a TV or a nearby
# conversation from opening a turn and feeding Whisper garbage.
#
# Downloads ~13 MB into web/vendor/ (git-ignored). Run once:
#     bash scripts/download_vad.sh
#
# Everything is served from your own machine afterwards — no CDN at runtime, so the
# "works with no internet" promise holds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/web/vendor"
ORT_VERSION="1.19.2"
CDN="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"
VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"

mkdir -p "$DEST"

fetch() {   # fetch <url> <filename> <min-bytes>
  local url="$1" name="$2" min="$3" out="$DEST/$2"
  if [ -f "$out" ] && [ "$(wc -c <"$out")" -ge "$min" ]; then
    echo "  ✓ $name (already present)"
    return
  fi
  echo "  ↓ $name"
  curl -fsSL --retry 3 -o "$out.part" "$url"
  local got; got="$(wc -c <"$out.part")"
  if [ "$got" -lt "$min" ]; then
    rm -f "$out.part"
    echo "  ✗ $name looked truncated ($got bytes) — not installing." >&2
    exit 1
  fi
  mv "$out.part" "$out"
}

echo "Vendoring Silero VAD + onnxruntime-web into web/vendor/ …"
# wasm-only ORT build (46 KB) — the full ort.min.js also drags in WebGL/WebGPU
# backends we never use.
fetch "$CDN/ort.wasm.min.js"              ort.wasm.min.js              20000
fetch "$CDN/ort-wasm-simd-threaded.mjs"   ort-wasm-simd-threaded.mjs   10000
fetch "$CDN/ort-wasm-simd-threaded.wasm"  ort-wasm-simd-threaded.wasm  5000000
fetch "$VAD_URL"                          silero_vad.onnx              1000000

echo
echo "Done. Restart the server and reload the page."
echo "The status line under the orb will read 'Silero VAD' once it's active."
