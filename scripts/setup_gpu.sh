#!/usr/bin/env bash
# One-time setup for a Linux GPU box (E2E L40S / A100 / H100) — Phase 1: Ollama on the GPU.
#
# Checks the machine BEFORE installing anything, so a missing driver or a full disk
# fails in ten seconds instead of forty minutes into a model download.
#
#   bash scripts/setup_gpu.sh          # then: bash scripts/run_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${ZS_LLM_MODEL:-qwen3:14b}"
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
die()  { printf "  \033[31m✗\033[0m %s\n" "$1"; exit 1; }

echo
echo "── Preflight ───────────────────────────────────────────────"

# ---- GPU ---------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — this box has no usable NVIDIA driver.
     On E2E, pick a GPU image with CUDA preinstalled, or install the driver first."
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
ok "GPU: $GPU"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 20000 ]; then
  warn "under 20GB VRAM — use qwen3:8b instead of $MODEL (edit config.gpu.yaml)"
fi

# ---- disk (weights are big: LLM ~9GB + Whisper ~3GB + Parler ~2GB) -----------
FREE=$(df -Pk . | awk 'NR==2{print int($4/1048576)}')
[ "$FREE" -ge 30 ] || die "only ${FREE}GB free — need ~30GB for the model weights."
ok "disk: ${FREE}GB free"

# ---- python ------------------------------------------------------------------
PY=$(command -v python3 || true)
[ -n "$PY" ] || die "python3 not found."
PYV=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')
case "$PYV" in
  3.10|3.11|3.12) ok "python $PYV" ;;
  *) die "python $PYV — the voice stack needs 3.10–3.12 (faster-whisper/pyttsx3 wheels)." ;;
esac

# ---- ollama ------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  warn "ollama not found — installing"
  curl -fsSL https://ollama.com/install.sh | sh
fi
ok "ollama: $(ollama --version 2>/dev/null | head -1)"

echo
echo "── Installing ──────────────────────────────────────────────"
[ -d .venv ] || $PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
ok "core dependencies"

# Indic-Parler (real Indian-language speech) needs torch/transformers. On a Linux GPU
# box there is no macOS `say`, so without this there is effectively no usable voice.
echo "  … installing GPU voice stack (torch + Indic-Parler, several GB)"
pip install -q -r requirements-voice.txt
ok "voice stack"

# faster-whisper on CUDA needs cuDNN; the pip wheel is the reliable way to get it.
pip install -q nvidia-cudnn-cu12 nvidia-cublas-cu12 2>/dev/null && ok "cuDNN/cuBLAS" || \
  warn "could not install nvidia-cudnn-cu12 — if STT fails on CUDA, install it manually"

echo
echo "── Models ──────────────────────────────────────────────────"
pgrep -x ollama >/dev/null 2>&1 || { nohup ollama serve >/tmp/ollama.log 2>&1 & sleep 5; }
if ollama list 2>/dev/null | grep -q "^${MODEL%%:*}.*${MODEL##*:}"; then
  ok "$MODEL already pulled"
else
  echo "  … pulling $MODEL (~9GB, one time)"
  ollama pull "$MODEL"
  ok "$MODEL"
fi

# Indic-Parler is a GATED repo — surface it now, not at the first spoken word.
if $PY - <<'EOF' 2>/dev/null
import os, pathlib, sys
tok = pathlib.Path.home()/".cache/huggingface/token"
sys.exit(0 if (tok.exists() or os.environ.get("HF_TOKEN")) else 1)
EOF
then ok "Hugging Face credentials present"
else
  warn "no Hugging Face token. Indic-Parler is gated — run 'huggingface-cli login' and"
  warn "accept the terms at huggingface.co/ai4bharat/indic-parler-tts, or Indian-language"
  warn "speech will fail at the first turn."
fi

echo
echo "────────────────────────────────────────────────────────────"
echo " Setup complete. Start the app with:"
echo ""
echo "     bash scripts/run_gpu.sh"
echo ""
echo "────────────────────────────────────────────────────────────"
