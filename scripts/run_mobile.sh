#!/usr/bin/env bash
# Run ZenSuvidha so you can use it on your PHONE (same Wi-Fi).
# Uses HTTPS with a self-signed cert — mobile browsers require a secure context
# for the microphone (getUserMedia), so plain http://<ip> won't allow voice.
set -e
cd "$(dirname "$0")/.."

CERTDIR=data/cert
KEY="$CERTDIR/key.pem"
CERT="$CERTDIR/cert.pem"
mkdir -p "$CERTDIR"

if [ ! -f "$CERT" ]; then
  echo "› Generating a one-time self-signed certificate…"
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" \
    -days 365 -subj "/CN=zensuvidha.local" >/dev/null 2>&1
fi

# find this machine's LAN IP
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || \
     hostname -I 2>/dev/null | awk '{print $1}')
IP=${IP:-<your-computer-ip>}

echo "──────────────────────────────────────────────────────────"
echo "  On your phone (same Wi-Fi), open:"
echo ""
echo "      https://$IP:8000"
echo ""
echo "  Accept the browser security warning (it's your own cert),"
echo "  then tap 'Start call' and allow the microphone."
echo "──────────────────────────────────────────────────────────"

exec .venv/bin/uvicorn zensuvidha.server:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile "$KEY" --ssl-certfile "$CERT"
