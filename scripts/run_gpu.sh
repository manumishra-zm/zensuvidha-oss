#!/usr/bin/env bash
# Run ZenSuvidha on a Linux GPU box with config.gpu.yaml.
#
# Serves HTTPS by default: browsers refuse microphone access (getUserMedia) on a plain
# http:// public IP, so voice simply cannot be tested over http. A self-signed cert is
# generated once — you'll get a browser warning, which is fine for testing. For anything
# shareable, front this with Caddy on a real domain and use --http.
#
#   bash scripts/run_gpu.sh            # HTTPS on :8000 (self-signed)
#   bash scripts/run_gpu.sh --http     # plain HTTP — typing works, the mic will NOT
set -euo pipefail
cd "$(dirname "$0")/.."

export ZS_CONFIG="${ZS_CONFIG:-config.gpu.yaml}"
PORT="${PORT:-8000}"
MODE="${1:-}"
CERTDIR=data/cert
KEY="$CERTDIR/key.pem"
CERT="$CERTDIR/cert.pem"

[ -d .venv ] || { echo "No .venv — run: bash scripts/setup_gpu.sh"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

# Ollama must be up before the app warms the model at startup.
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "› starting ollama…"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 \
  || { echo "✗ ollama is not responding on :11434 — see /tmp/ollama.log"; exit 1; }

MODEL=$(python - <<'EOF'
import os, yaml
print((yaml.safe_load(open(os.environ["ZS_CONFIG"])) or {}).get("llm", {}).get("model", ""))
EOF
)
if [ -n "$MODEL" ] && ! ollama list | grep -q "${MODEL%%:*}"; then
  echo "✗ model '$MODEL' is not pulled. Run:  ollama pull $MODEL"; exit 1
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
IP=${IP:-<server-ip>}

if [ "$MODE" = "--http" ]; then
  echo "──────────────────────────────────────────────────────────"
  echo "  http://$IP:$PORT     (typing only — the MIC WILL NOT WORK over http)"
  echo "──────────────────────────────────────────────────────────"
  exec uvicorn zensuvidha.server:app --host 0.0.0.0 --port "$PORT"
fi

if [ ! -f "$CERT" ]; then
  echo "› generating a one-time self-signed certificate…"
  mkdir -p "$CERTDIR"
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" \
    -days 365 -subj "/CN=$IP" >/dev/null 2>&1
fi

echo "──────────────────────────────────────────────────────────"
echo "  Open:  https://$IP:$PORT"
echo ""
echo "  Accept the browser warning (it's your own cert), then"
echo "  'Start call' and allow the microphone."
echo ""
echo "  Open port $PORT in the E2E firewall / security group first."
echo "──────────────────────────────────────────────────────────"
exec uvicorn zensuvidha.server:app --host 0.0.0.0 --port "$PORT" \
  --ssl-keyfile "$KEY" --ssl-certfile "$CERT"
