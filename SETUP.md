# ZenSuvidha OSS — Full Setup (every command)

Zero → running, with every command. Pick **Docker** (easiest) or **Local**.

---

## 0. Prerequisites

| Need | How |
|---|---|
| **Python 3.10–3.12** | `python3 --version` (voice needs 3.10–3.12; text works on any 3.10+) |
| **Ollama** (local LLM) | Install from <https://ollama.com>, then `ollama pull qwen3:4b` |
| **openssl** (only for phone/HTTPS) | Preinstalled on macOS/Linux |

> The STT models (faster-whisper `small`) download automatically on first run (~460MB).

---

## 1. Docker (one command — brings up Ollama + the app)

```bash
cd zensuvidha-oss
docker compose up --build
# open http://localhost:8000
```
That's it — the container waits for Ollama, pulls the model, and starts the server.

---

## 2. Local install & run

```bash
# from the zensuvidha-oss/ folder
python3.12 -m venv .venv            # use a 3.10–3.12 interpreter
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt     # core (STT + LLM client + TTS + denoise)

# one-time: the local model
ollama pull qwen3:4b
```

### Run it
```bash
# Web UI (desktop) — http://localhost:8000
uvicorn zensuvidha.server:app --host 127.0.0.1 --port 8000
#   …or:  make web

# Text CLI (no mic needed)
python -m zensuvidha.cli --pack clinic
python -m zensuvidha.cli --pack restaurant --speak     # + spoken replies

# Run the tests
pip install pytest && pytest -q                        #   …or: make test
```

### Run on your PHONE (same Wi-Fi)
Mobile browsers need HTTPS for the mic, so use the mobile runner (self-signed cert):
```bash
bash scripts/run_mobile.sh
# → open the printed  https://<your-ip>:8000  on the phone, accept the warning, tap Start call
```

---

## 3. Optional add-ons

### A. Better neural voices (Piper / Kokoro) — free
```bash
pip install -r requirements-voice.txt
bash scripts/download_piper_voice.sh          # downloads en_US-amy-medium into models/
# then set  tts.provider: piper  in config.yaml
```
(macOS already has good built-in `say` voices shown in the Voice picker — this is mainly for Linux/Docker.)

### B. Indian-language voices — AI4Bharat Indic-Parler-TTS — free, ~21 Indian languages
Speaks Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, … Pick a voice in the UI's
**Voice** dropdown (the "AI4Bharat · Indian voices" group), e.g. *Hindi · Divya*.
```bash
# 1) install the engine
pip install -r requirements-voice.txt
pip install --no-deps "parler-tts @ git+https://github.com/huggingface/parler-tts.git"
pip install descript-audio-codec sentencepiece

# 2) the model is GATED on Hugging Face — one-time free access:
#    a) make a free account at https://huggingface.co
#    b) open https://huggingface.co/ai4bharat/indic-parler-tts and click "Agree and access"
#    c) create a token at https://huggingface.co/settings/tokens
pip install -U "huggingface_hub[cli]"
hf auth login          # paste your token   (older CLIs: huggingface-cli login)
```
Then pick an AI4Bharat voice in the UI. First use downloads the ~2GB model. **Real-time on GPU;
very slow (~10–30s/sentence) on CPU** — best on a GPU box.

### C. Voice cloning ("Brand Voice") — free, heavy (~4GB), best on GPU
```bash
pip install -r requirements-clone.txt
# if pip complains about torch, install torch first per https://pytorch.org, then re-run.
#   …or:  make clone
```
Then in the UI: **★ Brand voice → Record my voice (15s)**. First clone downloads the XTTS model (~1.8GB).
> XTTS-v2 is non-commercial; swap to OpenVoice (MIT) for a commercial product.

---

## 3b. GPU server (E2E L40S / A100 / H100) — the real Indian-language setup

On a laptop, Indian languages are the weak spot: a native-script Telugu turn takes
**24–32s** (unusable for a call), and macOS `say` cannot pronounce Telugu at all. Both
are GPU problems, so this is not an optimisation — it's what makes Indic work.

### Provision

An **L40S (48GB)** is comfortable: Qwen3-14B AWQ ~10GB + Whisper `large-v3` ~3GB +
Indic-Parler ~3GB ≈ **16GB**, leaving ~30GB of headroom. Pick an image with **CUDA
drivers preinstalled**, give it **≥60GB disk** (model weights), and open port **8000**
in the E2E firewall / security group.

### Install and run

```bash
git clone <your repo> && cd zensuvidha-oss
bash scripts/setup_gpu.sh      # preflight (driver/disk/python) → deps → pulls qwen3:14b
bash scripts/run_gpu.sh        # HTTPS on :8000
```

`setup_gpu.sh` checks the machine *before* downloading anything, so a missing driver
fails in ten seconds rather than forty minutes into a pull.

### Three things that will bite you

1. **The microphone needs HTTPS.** Browsers refuse `getUserMedia` on a plain
   `http://<public-ip>` — the localhost exemption does not apply to a remote server, so
   voice cannot be tested over http at all. `run_gpu.sh` serves HTTPS with a self-signed
   cert (accept the warning). For anything shareable, front it with Caddy on a real
   domain. `run_gpu.sh --http` gives you plain http for typing only.

2. **Model family beats model size.** Qwen2.5 supports 29 languages and **Telugu is not
   one of them**; Qwen3 supports 119 and includes Telugu, Tamil, Kannada and Malayalam.
   A 14B Qwen2.5 is *worse* at Telugu than a 4B Qwen3. Stay on Qwen3.

3. **Indic-Parler is a gated Hugging Face repo.** Run `huggingface-cli login` and accept
   the terms (§3B above) *before* the first call, or Indian-language speech fails on the
   first turn. `setup_gpu.sh` warns if no token is present.

### Phase 2 — vLLM (only once quality is confirmed)

Ollama is single-stream; vLLM batches and is what you want for concurrent calls. Launch
it first, on its own port, and **cap its VRAM** — vLLM defaults to
`--gpu-memory-utilization 0.9`, which grabs 43GB of the L40S and leaves Whisper and
Indic-Parler unable to allocate on the same card:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-AWQ --port 8001 \
  --max-model-len 6144 --gpu-memory-utilization 0.55
```

Then switch the `llm:` block in `config.gpu.yaml` to the commented vLLM variant.

---

## 4. Makefile shortcuts

```bash
make install      # venv + core deps
make web          # web UI on :8000
make run PACK=clinic
make test         # engine tests
make voice        # install Piper + Kokoro
make clone        # install the cloning stack
make docker       # docker compose up
make clean        # remove venv + data
```

---

## 5. Configuration — `config.yaml`

| Setting | Default | Notes |
|---|---|---|
| `stt.model` | `small` | `tiny`/`base` = faster, `small` = most accurate |
| `stt.beam_size` | `5` | lower = faster, higher = more accurate |
| `stt.denoise` | `true` | background-noise removal before STT |
| `stt.language` | `en` | `hi` for Hindi; `null` for auto (mis-fires on noise) |
| `llm.model` | `qwen3:4b` | any Ollama model; `qwen3:8b` = smarter |
| `tts.provider` | `system` | `piper` / `kokoro` / `indic_parler` (Indian langs) / `clone` / `none` |
| `server.streaming` | `true` | sentence-by-sentence low-latency replies |
| `default_pack` | `clinic` | clinic · salon · restaurant · laundry · gym · hotel |

Env overrides: `OLLAMA_URL`, `ZS_LLM_MODEL`, `ZS_TTS`, `ZS_DEFAULT_PACK`, `ZS_CONFIG`.

---

## 6. Add a new industry

Copy any `packs/<name>.yaml`, edit the fields (persona, business, services, policies, scenarios,
vocabulary, escalation, knowledge), and restart. It appears in the CLI and UI automatically —
no code changes.

---

## Troubleshooting
- **`torch` errors during clone install** → install torch first per pytorch.org, then re-run `requirements-clone.txt`.
- **Mic blocked on phone** → you must use HTTPS: run `scripts/run_mobile.sh`.
- **Orb/audio silent** → click once (browser autoplay); *Start call* covers it.
- **Weak recognition** → keep `stt.model: small`; use headphones; speak close to the mic.
- **Ollama not reachable** → start Ollama, `ollama pull qwen3:4b`, check `/health`.
- **Python 3.13/3.14** → voice wheels aren't published yet; use 3.10–3.12 or Docker.
