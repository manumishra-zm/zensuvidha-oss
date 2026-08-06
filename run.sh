#!/usr/bin/env bash
# One-shot setup + run for ZenSuvidha OSS. Free, offline, CPU-only.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "› Creating virtualenv…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "› Installing dependencies…"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo
echo "› Checking Ollama…"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "  ⚠  Ollama not reachable at localhost:11434."
  echo "     Install from https://ollama.com  then run:  ollama pull qwen3:4b"
else
  echo "  ✓ Ollama is up. (If you haven't yet:  ollama pull qwen3:4b )"
fi

echo
echo "──────────────────────────────────────────────"
echo " Text CLI :  python -m zensuvidha.cli --pack clinic"
echo " Web voice:  uvicorn zensuvidha.server:app --port 8000   → http://localhost:8000"
echo "──────────────────────────────────────────────"
