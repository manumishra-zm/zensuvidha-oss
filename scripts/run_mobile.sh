#!/usr/bin/env bash
# Run ZenSuvidha so you can use it on your PHONE (same Wi-Fi).
# Uses HTTPS with a self-signed cert — mobile browsers require a secure context
# for the microphone (getUserMedia), so plain http://<ip> won't allow voice.
#
# THE CERTIFICATE HAS TO CARRY A subjectAltName, and it has to name the IP you actually
# type. This script used to issue `-subj "/CN=zensuvidha.local"` with no SAN at all,
# which no modern phone will accept:
#
#   * iOS/macOS have REQUIRED subjectAltName since iOS 13 — the Common Name is ignored
#     outright, so a CN-only cert is not "warn and let the user through", it is refused.
#   * Chrome has ignored CN since 58 (ERR_CERT_COMMON_NAME_INVALID).
#   * And the name has to be the LAN IP, because that is what you open. A cert for
#     `zensuvidha.local` does not match `https://192.168.1.7:8000` even if it is trusted.
#
# Without a secure context getUserMedia is not merely blocked — `navigator.mediaDevices`
# is undefined, so the page fails before it can explain why. Hence: find the IP FIRST,
# then issue the cert for it, and re-issue whenever the IP changes (a laptop that moves
# between home and office Wi-Fi gets a new one, and the cached cert would name the old).
set -e
cd "$(dirname "$0")/.."

CERTDIR=data/cert
KEY="$CERTDIR/key.pem"
CERT="$CERTDIR/cert.pem"
mkdir -p "$CERTDIR"

# find this machine's LAN IP, before the certificate is issued for it
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || \
     hostname -I 2>/dev/null | awk '{print $1}')

if [ -z "$IP" ]; then
  echo "!  Could not work out this machine's LAN IP."
  echo "   Find it (System Settings › Wi-Fi › Details, or \`ifconfig\`) and re-run as:"
  echo "       ZS_IP=192.168.x.x bash scripts/run_mobile.sh"
  IP="${ZS_IP:-}"
  [ -z "$IP" ] && exit 1
fi
IP="${ZS_IP:-$IP}"

# Re-issue when the cert is missing, expired, or was issued for a DIFFERENT address —
# the last one is the case that silently breaks a laptop which changed networks.
NEED_CERT=1
if [ -f "$CERT" ] && [ -f "$KEY" ]; then
  if openssl x509 -in "$CERT" -noout -checkend 86400 >/dev/null 2>&1 && \
     openssl x509 -in "$CERT" -noout -ext subjectAltName 2>/dev/null \
       | grep -Eq "IP Address:$IP(,|$|[[:space:]])"; then
    NEED_CERT=0
  fi
fi

if [ "$NEED_CERT" = "1" ]; then
  echo "› Issuing a self-signed certificate for $IP …"
  rm -f "$CERT" "$KEY"
  # -addext needs OpenSSL 1.1.1+ (macOS ships LibreSSL, which also accepts it). The
  # fallback writes a temporary config for anything older, so this never silently
  # produces the CN-only certificate the header above describes.
  if ! openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" \
        -days 365 -subj "/CN=$IP" \
        -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost" \
        -addext "basicConstraints=critical,CA:FALSE" \
        -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
        -addext "extendedKeyUsage=serverAuth" >/dev/null 2>&1; then
    CONF=$(mktemp)
    cat >"$CONF" <<EOF
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = $IP
[v3]
subjectAltName = IP:$IP,IP:127.0.0.1,DNS:localhost
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
EOF
    openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" \
      -days 365 -config "$CONF" >/dev/null 2>&1
    rm -f "$CONF"
  fi
  chmod 600 "$KEY"
  # Prove it rather than assume it: a cert without the IP in its SAN looks fine here and
  # fails on the phone with an error that says nothing about certificates.
  if ! openssl x509 -in "$CERT" -noout -ext subjectAltName 2>/dev/null | grep -q "IP Address:$IP"; then
    echo "!  The certificate was issued WITHOUT $IP in its subjectAltName."
    echo "   Your phone will refuse it. Check your openssl version (need 1.1.1+):"
    openssl version
    exit 1
  fi
fi

echo "──────────────────────────────────────────────────────────"
echo "  On your phone (same Wi-Fi), open:"
echo ""
echo "      https://$IP:8000"
echo ""
echo "  Accept the security warning — it is your own certificate."
echo "  On iPhone: tap 'Show Details' › 'visit this website'."
echo "  Then tap 'Start call' and allow the microphone."
echo ""
echo "  Keep the phone screen on the page; the call holds a wake"
echo "  lock, but switching apps pauses the microphone until you"
echo "  come back."
echo "──────────────────────────────────────────────────────────"

exec .venv/bin/uvicorn zensuvidha.server:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile "$KEY" --ssl-certfile "$CERT"
