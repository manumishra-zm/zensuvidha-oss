#!/usr/bin/env bash
# Wait for Ollama, pull the model once, then start the server.
set -e
: "${OLLAMA_URL:=http://ollama:11434}"
: "${ZS_LLM_MODEL:=qwen3:4b}"

echo "› Waiting for Ollama at $OLLAMA_URL …"
until curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; do sleep 2; done

echo "› Ensuring model '$ZS_LLM_MODEL' is present (first run downloads it)…"
curl -s "$OLLAMA_URL/api/pull" -d "{\"name\":\"$ZS_LLM_MODEL\"}" >/dev/null || \
  echo "  (pull request failed — the app will still start; pull manually if needed)"

echo "› Starting ZenSuvidha OSS on :8000"
exec uvicorn zensuvidha.server:app --host 0.0.0.0 --port 8000
