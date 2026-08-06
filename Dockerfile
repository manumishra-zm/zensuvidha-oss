FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_URL=http://ollama:11434 \
    ZS_LLM_MODEL=qwen3:4b

# espeak-ng → pyttsx3 voices; libsndfile1 → soundfile; curl → healthcheck/entrypoint.
# (faster-whisper bundles ffmpeg via PyAV wheels, so no system ffmpeg needed.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      espeak-ng libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x scripts/*.sh || true

# run as a non-root user (never run the service as root)
RUN useradd -m -u 10001 app && mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "zensuvidha.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ── GPU / full multilingual TTS (AI4Bharat Indic-Parler) ────────────────────────
# The base image ships only the CPU/system-TTS stack. For proper Indian-language
# speech (indic_parler, config.gpu.yaml) build a GPU variant that ALSO runs
# `pip install -r requirements-voice.txt` and provides a Hugging Face token
# (accept terms at huggingface.co/ai4bharat/indic-parler-tts).
