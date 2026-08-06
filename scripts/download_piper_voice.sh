#!/usr/bin/env bash
# Download a free Piper neural voice into ./models.
# Usage:  bash scripts/download_piper_voice.sh [voice]
#   default voice: en_US-amy-medium
#   browse voices: https://huggingface.co/rhasspy/piper-voices
set -e
VOICE="${1:-en_US-amy-medium}"

# voice id format: <lang>_<REGION>-<name>-<quality>   e.g. en_US-amy-medium
LANG_REGION="${VOICE%%-*}"        # en_US
LANG="${LANG_REGION%_*}"          # en
REST="${VOICE#*-}"                # amy-medium
NAME="${REST%-*}"                 # amy
QUAL="${VOICE##*-}"               # medium

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
DIR="$BASE/$LANG/$LANG_REGION/$NAME/$QUAL"

mkdir -p models
echo "› Downloading Piper voice '$VOICE' …"
curl -L --fail -o "models/$VOICE.onnx"      "$DIR/$VOICE.onnx"
curl -L --fail -o "models/$VOICE.onnx.json" "$DIR/$VOICE.onnx.json"
echo "✓ Saved to models/$VOICE.onnx"
echo "  Now set in config.yaml →  tts.provider: piper"
