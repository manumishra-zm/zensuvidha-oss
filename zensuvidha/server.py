"""FastAPI voice server.

Latency features:
  * warm-pooling — STT/TTS/LLM are loaded and warmed at startup (lifespan), so the
    first real turn is not cold.
  * streaming turns — the LLM is streamed and each finished sentence is synthesised
    and sent immediately, so the caller hears audio while the rest is still being
    generated (low time-to-first-audio).
  * per-turn metrics — STT/LLM-first-token/TTS timings are logged and returned.

Robustness:
  * one asyncio.Lock per session (serialises turns on a call).
  * per-turn try/except keeps the socket alive through provider errors.
  * automatic fallback to a non-streaming turn if streaming fails or is disabled.
"""
import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .booking import list_bookings, log_turn, list_turns
from .config import load_config
from .guard import (GuardConfig, is_degenerate, looks_complete, looks_incomplete,
                    looks_like_echo,
                    ungrounded_numbers)
from .llm import get_llm, OllamaError
from .logs import setup_logging, set_call_id, call_id_var
from .orchestrator import _KIND_TO_SAFE, Session, extract_kind, extract_say, next_chunk
from .packs import list_packs, load_pack

ROOT = Path(__file__).resolve().parent.parent
cfg = load_config()
setup_logging(cfg)              # configurable level / rotating file / JSON / call-id
log = logging.getLogger("zensuvidha.server")

_llm = get_llm(cfg["llm"])          # ollama (dev) or vllm/openai (GPU) per cfg.llm.provider
_stt = None
_tts = None
_brand = {"tts": None}          # owner's cloned "brand voice" (set via /voice/clone)
_speaker_gate = None            # optional: ignore voices other than the caller's
_diarizer = None                # optional: keep only the caller's segments in a turn
GUARD = GuardConfig(cfg.get("guard"))   # grounding/language checks — see guard.py
# The denoise router is off by default on measurement (see pipeline.AUTO_DENOISE); this
# is the one knob that puts it back.
from . import pipeline as _pipeline  # noqa: E402
_pipeline.AUTO_DENOISE = bool((cfg.get("stt") or {}).get("auto_denoise", False))
STREAMING = bool(cfg.get("server", {}).get("streaming", True))
LOG_TIMINGS = bool(cfg.get("server", {}).get("log_timings", True))
LOG_TRANSCRIPTS = bool(cfg.get("server", {}).get("log_transcripts", True))
import secrets  # noqa: E402  (short per-call correlation ids)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- warm-pool providers so the first call is fast ----
    global _stt, _tts, _brand_lock
    from .stt import get_stt
    from .tts import get_tts
    _brand_lock = asyncio.Lock()
    log.info("Warming up providers (in parallel)…")

    async def _warm_stt():
        global _stt, _speaker_gate
        _stt = await run_in_threadpool(get_stt, cfg.get("stt", {}))
        from .speaker import get_speaker_gate
        _speaker_gate = await run_in_threadpool(get_speaker_gate, cfg.get("stt", {}))
        from .diarize import get_diarizer
        global _diarizer
        _diarizer = await run_in_threadpool(get_diarizer, cfg.get("stt", {}))
        if _diarizer and not _speaker_gate:
            log.warning("stt.diarize is on but stt.speaker_gate is off — segment gating "
                        "needs an enrolled voiceprint, so it will stay inactive.")

    async def _warm_tts():
        global _tts
        _tts = await run_in_threadpool(get_tts, cfg.get("tts", {}))
        if _tts:
            try:
                await run_in_threadpool(_tts.synth, "Hello.")   # prime TTS
            except Exception:  # noqa: BLE001
                pass

    # STT load, TTS load+prime, and LLM warmup all run concurrently → faster boot.
    # The LLM warmup sends the REAL system prompt of the pack this server will serve,
    # not "hi": loading the weights is the cheap half, and evaluating a ~6,000-token
    # prompt with no cached prefix is what made the first caller wait 23 seconds.
    def _prime_prompt():
        try:
            sess = Session(load_pack(cfg.get("default_pack", "clinic")), _llm)
            return sess.call_messages(), sess.context_budget("")
        except Exception as e:  # noqa: BLE001
            log.debug("could not build the warmup prompt (%s) — warming with a stub", e)
            return None, None

    prime_msgs, prime_ctx = _prime_prompt()
    warmups = [_warm_stt(), _warm_tts(),
               run_in_threadpool(_llm.warmup, None, prime_msgs)]
    if FAST_MODEL:                                    # keep the routing model hot too
        warmups.append(run_in_threadpool(_llm.warmup, FAST_MODEL, prime_msgs))
    await asyncio.gather(*warmups)
    # Pre-cache greetings + fillers so the FIRST call's greeting and any slow-reply
    # filler are instant (no cold-start TTS latency). The TTS cache is keyed by voice,
    # so we prime through the same path a real session uses.
    async def _precache():
        if not _tts:
            return
        try:
            for pk in list_packs():
                sess = Session(load_pack(pk), _llm, tts=_tts)
                await run_in_threadpool(sess.tts_bytes, sess.greeting())
            # Fillers must be cached under the voice they will be SPOKEN with — the
            # cache key is "voice|text", so warming voice=None while real turns use
            # session.voice meant every filler was synthesised cold on the critical
            # path (~1.12s): the line whose entire job is to prevent dead air WAS the
            # dead air.
            #
            # But warming every phrase × every pack's voice is 209 synth calls, ~84s of
            # CPU, and it competes with the first real callers for exactly as long —
            # measured: the first two turns were seconds slower than the fourth. Most of
            # it was waste anyway, because a provider ignores voice ids that are not its
            # own (Kokoro maps all seven macOS names onto one voice), so six of every
            # seven calls re-synthesised identical audio.
            #
            # Warm the voice this server will actually greet people with, one phrase per
            # language, and stop. That is the case that matters and it costs ~13 calls.
            default_voice = Session(load_pack(cfg.get("default_pack", "clinic")),
                                    _llm, tts=_tts).voice
            for phrases in _FILLERS.values():
                await run_in_threadpool(_tts.synth, phrases[0], default_voice)
                await asyncio.sleep(0)          # yield: a real turn always outranks this
        except Exception as e:  # noqa: BLE001
            log.debug("greeting/filler precache skipped: %s", e)
    _spawn(_precache())                  # background — don't delay readiness

    # Which caller languages can this deployment actually SPEAK? A script no configured
    # voice can pronounce produces a silent reply — correct behaviour (the UI explains
    # it) but a bad thing to discover from a real caller. Measured on macOS: every Indian
    # script speaks except Gujarati. Say it once at boot so whoever deploys this knows
    # before anybody rings.
    async def _voice_coverage():
        if not _tts:
            return
        probes = {"Hindi": "नमस्ते", "Telugu": "నమస్కారం", "Tamil": "வணக்கம்",
                  "Kannada": "ನಮಸ್ಕಾರ", "Malayalam": "നമസ്കാരം", "Bengali": "নমস্কার",
                  "Marathi": "नमस्कार", "Gujarati": "નમસ્તે", "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
                  "Odia": "ନମସ୍କାର", "Urdu": "السلام علیکم"}
        mute = []
        for name, hello in probes.items():
            try:
                if not await run_in_threadpool(_tts.synth, hello, None):
                    mute.append(name)
            except Exception:  # noqa: BLE001
                mute.append(name)
        if mute:
            log.warning("NO VOICE for: %s — a caller speaking one of those will SEE the "
                        "reply and hear nothing. Install an OS voice for it, or run "
                        "tts.provider: indic_parler on a GPU.", ", ".join(mute))
        else:
            log.info("voice coverage: every probed Indian language speaks")
    _spawn(_voice_coverage())

    ok, info = _llm.health()
    log.info("Ready. Ollama=%s model=%s stt=%s tts=%s streaming=%s sessions<=%d",
             ok, cfg["llm"]["model"], cfg.get("stt", {}).get("provider"),
             cfg.get("tts", {}).get("provider"), STREAMING, MAX_SESSIONS)
    if ok and cfg["llm"]["model"] not in info:
        log.warning("Model '%s' not pulled — run: ollama pull %s", cfg["llm"]["model"], cfg["llm"]["model"])
    yield
    # ---- graceful shutdown: cancel any outstanding background tasks ----
    for t in list(_bg_tasks):
        t.cancel()


app = FastAPI(title="ZenSuvidha OSS", lifespan=lifespan)


def _new_session(pack_name: str) -> Session:
    s = Session(load_pack(pack_name), _llm, tts=_tts, stt=_stt, guard_cfg=GUARD,
                speaker_gate=_speaker_gate)
    s.diarizer = _diarizer
    s.clone = _brand["tts"]
    return s


def _b64(audio) -> str | None:
    return base64.b64encode(audio).decode() if audio else None


SYNTH_WORKERS = max(1, int(cfg.get("server", {}).get("synth_workers", 3)))
FILLER = cfg.get("server", {}).get("filler") or None
FAST_MODEL = cfg["llm"].get("fast_model") or None
# If a reply is slow (LLM cold/thinking), speak a short natural filler after this many
# seconds so the caller isn't left in dead silence. 0 disables.
SLOW_FILLER_S = float(cfg.get("server", {}).get("slow_filler_seconds", 2.0))
# Ship TTS audio frame-by-frame for providers that can render progressively (Kokoro).
# Self-gating: providers without .synth_stream keep the whole-clip path, so turning
# this on cannot slow down macOS `say`, Piper or XTTS.
PCM_STREAMING = bool(cfg.get("server", {}).get("pcm_streaming", True))
STREAM_QUEUE = max(2, int(cfg.get("server", {}).get("stream_queue", 24)))
# Transcribe an utterance while the caller may still be mid-pause, so the transcript is
# already in hand when the endpoint confirms. See the `stt_hint`/`commit` protocol.
SPECULATIVE_STT = bool(cfg.get("server", {}).get("speculative_stt", True))
# A clip at least this long that yields no transcript gets a spoken "say that again"
# rather than silence. Below it, the audio is a cough or a chair and silence is right.
REPEAT_ASK_S = float(cfg.get("server", {}).get("repeat_ask_seconds", 2.0))
# A caller who says nothing at all. Every real receptionist re-prompts and then lets the
# line go; without this the call sits open forever holding a session slot, and the caller
# — who may simply have been distracted — gets no cue that anyone is still waiting.
IDLE_PROMPT_S = float(cfg.get("server", {}).get("idle_prompt_seconds", 20.0))
IDLE_HANGUP_S = float(cfg.get("server", {}).get("idle_hangup_seconds", 45.0))

import random  # noqa: E402  (used only for filler variety; server runs in a normal process)

# language-matched "let me check…" fillers, so the filler stays in the caller's language
_FILLERS = {
    "English":  ["One moment, let me check that.", "Just checking your details…",
                 "Let me look that up for you.", "Give me a second…"],
    "Hindi":    ["एक मिनट, मैं देख लेती हूँ…", "ज़रा आपकी जानकारी देख लूँ…", "एक सेकंड रुकिए…"],
    "Telugu":   ["ఒక్క నిమిషం, చూస్తాను…", "మీ వివరాలు చూస్తున్నాను…"],
    "Kannada":  ["ಒಂದು ಕ್ಷಣ, ನೋಡುತ್ತೇನೆ…", "ನಿಮ್ಮ ವಿವರಗಳನ್ನು ನೋಡುತ್ತಿದ್ದೇನೆ…"],
    "Tamil":    ["ஒரு நிமிடம், பார்க்கிறேன்…", "உங்கள் விவரங்களைப் பார்க்கிறேன்…"],
    "Bengali":  ["এক মিনিট, দেখে নিচ্ছি…", "আপনার তথ্য দেখে নিই…"],
    "Marathi":  ["एक मिनिट, बघते…", "तुमची माहिती बघते…"],
    "Gujarati": ["એક મિનિટ, જોઈ લઉં…", "તમારી વિગતો જોઉં છું…"],
    "Malayalam":["ഒരു നിമിഷം, നോക്കട്ടെ…", "നിങ്ങളുടെ വിവരങ്ങൾ നോക്കുന്നു…"],
    "Punjabi":  ["ਇੱਕ ਮਿੰਟ, ਵੇਖਦੀ ਹਾਂ…", "ਤੁਹਾਡੀ ਜਾਣਕਾਰੀ ਵੇਖਦੀ ਹਾਂ…"],
    "Urdu":     ["ایک منٹ، دیکھتی ہوں…", "آپ کی تفصیلات دیکھ رہی ہوں…"],
    "Odia":     ["ଏକ ମିନିଟ୍, ଦେଖୁଛି…", "ଆପଣଙ୍କ ବିବରଣୀ ଦେଖୁଛି…"],
    "Assamese": ["এক মিনিট, চাই লওঁ…", "আপোনাৰ তথ্য চাই আছোঁ…"],
}


# A BACKCHANNEL is not a filler. A filler covers OUR silence while we think; a
# backchannel is what a listener says while the OTHER person is still talking — the
# "mm-hm" that tells them you are still there. Its absence is the most machine-like
# thing about a long turn: the agent is completely silent, then abruptly speaks.
#
# Deliberately one short word. Anything longer stops being a listening noise and starts
# being an interruption, and a caller mid-sentence will stop to let you finish.
_BACKCHANNELS = {
    "English":  "Mm-hm.",
    "Hindi":    "जी…",
    "Telugu":   "అలాగే…",
    "Kannada":  "ಹೌದು…",
    "Tamil":    "ஆமாம்…",
    "Bengali":  "হ্যাঁ…",
    "Marathi":  "हो…",
    "Gujarati": "હા…",
    "Malayalam": "ശരി…",
    "Punjabi":  "ਹਾਂ ਜੀ…",
    "Urdu":     "جی…",
    "Odia":     "ହଁ…",
}
BACKCHANNEL_ON = bool(cfg.get("server", {}).get("backchannel", True))


def _backchannel_for(lang_name):
    """The listening noise for this call's language, or None to stay silent.

    Never falls back to English: dropping an English "mm-hm" into a Telugu call is
    exactly the single-language rule the fillers already observe.
    """
    if not BACKCHANNEL_ON:
        return None
    return _BACKCHANNELS.get(lang_name or "English")


_filler_used: set = set()


def _pick_filler(lang_name):
    """A short filler in the caller's language. Returns None (→ no filler) for a known
    non-English call with no matching phrase, so we never inject English mid-call.

    The FIRST filler in each language is always phrases[0], because that is the one
    the startup precache warmed. Picking at random meant the line whose whole purpose
    is to cover a slow reply was itself synthesised cold — roughly a second of the very
    silence it exists to fill. Later fillers vary, by which time the cache is warm and
    a long call does not sound like a stuck record.
    """
    key = lang_name if lang_name in _FILLERS else "English"
    if lang_name and lang_name != "English" and key == "English":
        return None                                  # don't break the single-language rule
    phrases = _FILLERS[key]
    if key not in _filler_used:
        _filler_used.add(key)
        return phrases[0]                            # the precached one
    return random.choice(phrases)


# ---- deploy limits ---------------------------------------------------------- #
MAX_SESSIONS = int(cfg.get("server", {}).get("max_sessions", 24))      # concurrent WS calls
ALLOW_CLONE = bool(cfg.get("server", {}).get("allow_clone", True))     # /voice/clone enabled
MAX_UPLOAD = int(cfg.get("server", {}).get("max_upload_bytes", 6_000_000))
RESUME_TTL = float(cfg.get("server", {}).get("resume_ttl_seconds", 90))  # keep a dropped call this long
# Reconnecting callers may exceed max_sessions by this much — mid-call beats brand-new,
# but the overflow is bounded rather than unlimited.
RESUME_HEADROOM = max(0, int(cfg.get("server", {}).get("resume_headroom", 4)))
_sessions = {"n": 0}

# On disconnect we STASH the Session by its call id so a reconnect (Wi-Fi blip) resumes the
# SAME conversation — history, slots, latched language, voice — instead of a fresh amnesiac one.
_stash: dict = {}   # sid -> {"session": Session, "expires": ts}

def _prune_stash():
    now = time.time()
    for k in [k for k, v in _stash.items() if v["expires"] < now]:
        _stash.pop(k, None)
    if len(_stash) > 200:                                 # hard cap: drop the oldest
        for k in sorted(_stash, key=lambda k: _stash[k]["expires"])[:len(_stash) - 200]:
            _stash.pop(k, None)
_brand_lock = None   # set in lifespan (asyncio.Lock; created inside the running loop)

# ---- tracked background tasks (so exceptions surface + we can cancel on shutdown) ---
_bg_tasks: set = set()

def _spawn(coro):
    t = asyncio.ensure_future(coro)
    _bg_tasks.add(t)
    def _done(task):
        _bg_tasks.discard(task)
        if not task.cancelled() and task.exception():
            log.warning("background task error: %s", task.exception())
    t.add_done_callback(_done)
    return t


def _record_turn(cid, pack, user_text, reply, lang, latency_ms):
    """Persist a completed turn (user + assistant) off the event loop, if enabled."""
    if not LOG_TRANSCRIPTS:
        return
    async def _write():
        try:
            await run_in_threadpool(log_turn, cid, pack, "user", user_text, lang, None)
            await run_in_threadpool(log_turn, cid, pack, "assistant", reply, lang, latency_ms)
        except Exception as e:  # noqa: BLE001
            log.debug("transcript write skipped: %s", e)
    _spawn(_write())


# ---- binary audio wire format --------------------------------------------- #
# One self-contained WS binary frame carries metadata + raw WAV, avoiding the
# ~33% base64 bloat and a JSON encode/parse per audio chunk:
#   [4-byte big-endian header length][UTF-8 JSON metadata][raw WAV bytes]
# Messages WITHOUT audio (reply_start/reply_end/error/voiceinfo) stay JSON text.
def _audio_frame(meta: dict, audio: bytes | None) -> bytes:
    header = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return len(header).to_bytes(4, "big") + header + (audio or b"")


async def _dropped(sock, why: str):
    """Tell the client a turn it sent will not be answered.

    The client goes to "thinking" the instant it ships a recording and only leaves
    that state when a reply arrives. Every server path that decides NOT to answer
    therefore has to say so, or the caller sits watching a spinner with a live
    microphone they believe is switched off. Silence is often the right REPLY; it is
    never the right protocol.
    """
    try:
        await sock.send_json({"type": "turn_dropped", "reason": why})
    except Exception:  # noqa: BLE001  (socket already gone — nothing to tell)
        pass


# OFF by default. MEASURED on one local Ollama, which is what this project targets:
#
#   a turn alone                    2339 ms
#   the same turn, spec in flight   6044 ms      +3704 ms
#
# Ollama serialises requests per model, so the real generation QUEUES BEHIND the guess
# instead of overlapping it — the turn gets 2.6x slower, which is the exact opposite of
# what this was built for. The same shape as a bug already in this codebase's history:
# a degeneracy retry fired from inside an open stream and was queued behind the
# generation it had already abandoned.
#
# It only pays where the model server can genuinely run concurrent requests — a GPU with
# spare capacity, or OLLAMA_NUM_PARALLEL > 1 on hardware that can absorb it. On a laptop
# it is a straight loss.
SPECULATIVE_REPLY = bool(cfg.get("server", {}).get("speculative_reply", False))


def _void_speculation(spec: dict) -> None:
    """Throw away a speculative reply and the work still producing it.

    Called from every path that invalidates the guess — the caller resumed, they barged
    in, a newer guess arrived. Missing one of them is how a reply to a half-finished
    sentence reaches the caller, so this is one function rather than three copies.
    """
    t = spec.get("gen")
    if t is not None and not t.done():
        t.cancel()
    spec["gen"] = None
    spec["gen_for"] = None


async def _speculate_reply(session, guess: str) -> str:
    """Generate the reply to a transcript the caller has not yet confirmed.

    The saving is the gap between speculating (450ms of silence) and the endpoint
    (800-1200ms) — 350-750ms off time-to-first-audio on a long turn, which is the
    dominant remaining wait now that STT is already off the critical path.

    Strictly READ-ONLY on the session. `begin_user` appends history and folds the
    caller's numbers into the grounding set, and this system has already been broken
    once by letting a guess mutate state — a voiceprint enrolled from a speculative
    fragment locked the caller out for a whole call. So this builds the message list by
    hand, streams into a buffer, and touches nothing. If the guess is wrong the buffer
    is dropped and the only cost is compute nobody was using.
    """
    msgs = session.call_messages() + [{"role": "user", "content": guess}]
    buf = []
    async for delta in session.llm.astream(
            msgs, force_json=True, model=session.model,
            num_predict=session.reply_budget(guess), meta={},
            num_ctx=session.context_budget(guess)):
        buf.append(delta)
    return "".join(buf)


async def _send_audio(sock, meta: dict, audio: bytes | None, session=None):
    """Ship an audio frame, and remember it as echo reference on the way out.

    Every byte we play leaves through here, which makes it the one place that cannot
    miss a frame — and a reference with holes in it is worse than none, because the
    frames it fails to explain are exactly the ones that get answered as if the caller
    had said them.
    """
    if session is not None and audio:
        try:
            session.note_played(audio)
        except Exception as e:  # noqa: BLE001
            log.debug("echo: could not record our own output (%s)", e)
    await sock.send_bytes(_audio_frame(meta, audio))


# ---- simple-turn model routing -------------------------------------------- #
_SIMPLE_RE = re.compile(
    r"^(hi|hello|hey|namaste|namaskar|yes|no|ok|okay|thanks|thank you|thank u|bye|"
    r"goodbye|good morning|good evening|good afternoon|sure|hmm|great|perfect|"
    r"nice|cool|got it|alright)[\s!.,]*$",
    re.IGNORECASE)


def _is_simple_turn(text: str) -> bool:
    """Greetings / acknowledgements / yes-no — cheap turns a tiny model handles fine.
    Anything with digits (times, phone, party size) or >6 words is NOT simple."""
    t = (text or "").strip()
    if not t or any(c.isdigit() for c in t):
        return False
    if _SIMPLE_RE.match(t):
        return True
    return len(t.split()) <= 3


def _route_model(session, user_text: str):
    """Which LLM to use for this turn. A caller-pinned model always wins; otherwise trivial
    ENGLISH turns go to the (English-only) fast_model. Non-English turns always use the main
    multilingual model. None → OllamaLLM's config default."""
    if session.model:
        return session.model
    if FAST_MODEL and _is_simple_turn(user_text):
        lang = session.reply_language(user_text)
        if lang in (None, "English"):          # only fast-route English (fast_model is English-only)
            return FAST_MODEL
    return None


# --------------------------------------------------------------------------- #
# HTTP routes
# --------------------------------------------------------------------------- #
_INDEX_HTML = None

@app.get("/", response_class=HTMLResponse)
def index():
    global _INDEX_HTML
    if _INDEX_HTML is None:
        try:
            _INDEX_HTML = (ROOT / "web" / "index.html").read_text()
        except OSError:
            return HTMLResponse("<h1>UI not found</h1>", status_code=500)
    return _INDEX_HTML


# The mic framer runs as an AudioWorklet, which the browser can only load from a URL.
@app.get("/vad-worklet.js")
def vad_worklet():
    try:
        return HTMLResponse((ROOT / "web" / "vad-worklet.js").read_text(),
                            media_type="application/javascript")
    except OSError:
        return HTMLResponse("", status_code=404)


# Optional Silero VAD assets (scripts/download_vad.sh). Served from disk so the app
# stays fully offline — nothing here is fetched from a CDN at runtime.
_VENDOR_TYPES = {".js": "application/javascript", ".mjs": "application/javascript",
                 ".wasm": "application/wasm", ".onnx": "application/octet-stream"}


# HEAD too: the browser probes this endpoint with HEAD before loading the VAD, and a
# GET-only route answered 405 — a real console error on every page load, and a probe
# that reports "missing" for a file that is right there.
@app.head("/vendor/{name}")
@app.get("/vendor/{name}")
def vendor(name: str):
    from fastapi.responses import FileResponse
    suffix = Path(name).suffix
    # Serve only known asset types by exact filename — no user-controlled path parts.
    if suffix not in _VENDOR_TYPES or "/" in name or "\\" in name or name.startswith("."):
        return JSONResponse({"error": "not found"}, status_code=404)
    path = ROOT / "web" / "vendor" / name
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type=_VENDOR_TYPES[suffix],
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/packs")
def packs():
    return {"packs": list_packs(), "default": cfg.get("default_pack")}


@app.get("/voices")
def voices():
    """Natural voices available for the TTS provider (macOS `say` for now).
    Indian voices are listed first. Empty list on non-macOS / other providers."""
    if sys.platform != "darwin" or cfg.get("tts", {}).get("provider") != "system":
        return {"voices": []}
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return {"voices": []}
    keep = {"en_IN", "hi_IN", "en_US", "en_GB", "en_AU", "en_IE"}
    novelty = {"Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Good News",
               "Jester", "Junior", "Kathy", "Organ", "Superstar", "Ralph", "Trinoids", "Whisper",
               "Wobble", "Zarvox", "Fred", "Deranged", "Hysterical", "Pipe Organ",
               "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"}
    seen, good = set(), []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\b", line)
        if not m:
            continue
        name = m.group(1).split(" (")[0].strip()   # "Aman (English (India))" → "Aman"
        loc = m.group(2)
        if loc in keep and name not in novelty and name not in seen:
            seen.add(name)
            good.append({"name": name, "locale": loc})
    good.sort(key=lambda v: (0 if v["locale"] in ("en_IN", "hi_IN") else 1, v["name"]))
    return {"voices": good[:24]}


# ---- PII endpoints ---------------------------------------------------------- #
# /transcripts and /bookings return every caller's NAME, PHONE NUMBER and the full text
# of their call. That is harmless while the server is bound to 127.0.0.1, but
# config.gpu.yaml binds 0.0.0.0 — so on the GPU box these were open to anyone who could
# reach the port. Off-loopback they now require a token.
_ADMIN_TOKEN = (cfg.get("server", {}) or {}).get("admin_token") or None


def _authorised(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    if host in ("127.0.0.1", "::1", "localhost"):
        return True                                   # a local operator, as before
    if not _ADMIN_TOKEN:
        return False                                  # remote + no token configured = no
    sent = request.headers.get("x-admin-token") or request.query_params.get("token")
    return bool(sent) and secrets.compare_digest(sent, _ADMIN_TOKEN)


def _denied():
    return JSONResponse(
        {"error": "This endpoint returns caller PII. Set server.admin_token in config "
                  "and send it as the X-Admin-Token header, or access from localhost."},
        status_code=403)


@app.get("/bookings")
def bookings(request: Request, pack: str | None = None):
    if not _authorised(request):
        log.warning("blocked unauthenticated /bookings from %s",
                    request.client.host if request.client else "?")
        return _denied()
    return {"bookings": list_bookings(pack)}


@app.get("/transcripts")
def transcripts(request: Request, call_id: str | None = None,
                pack: str | None = None, limit: int = 200):
    """Recent conversation turns (QA / audit / call history). Contains caller PII."""
    if not _authorised(request):
        log.warning("blocked unauthenticated /transcripts from %s",
                    request.client.host if request.client else "?")
        return _denied()
    # min(limit, 1000) does not bound a NEGATIVE limit, and SQLite reads LIMIT -1 as
    # "no limit" — `?limit=-1` dumped the entire turns table, every caller's transcript.
    return {"turns": list_turns(call_id=call_id, pack=pack,
                                limit=max(1, min(int(limit), 1000)))}


def _load_clone(ref_path: str):
    from .tts import CloneTTS
    return CloneTTS({"provider": "clone", "reference": ref_path,
                     "clone_language": cfg.get("tts", {}).get("clone_language", "en")})


_indic_cache = {}

def _indic_voice(speaker: str):
    """Build (and cache) an AI4Bharat Indic-Parler voice for a named speaker.
    The model itself is a class-level singleton — only the description varies."""
    if speaker in _indic_cache:
        return _indic_cache[speaker]
    from .tts import IndicParlerTTS
    desc = (f"{speaker} speaks in a clear, warm, natural voice at a moderate pace with "
            f"expressive intonation. The recording is very high quality with no background noise.")
    tts = IndicParlerTTS({"indic_description": desc, "device": cfg.get("tts", {}).get("device")})
    _indic_cache[speaker] = tts
    return tts


@app.post("/voice/clone")
async def voice_clone(request: Request):
    """Owner records/uploads a ~10s WAV; we clone it and use it as the agent's voice.
    Gated by server.allow_clone; size-capped; serialised so concurrent uploads don't race."""
    if not ALLOW_CLONE:
        return JSONResponse({"error": "Voice cloning is disabled on this server."}, status_code=403)
    # Check the DECLARED size first. `await request.body()` reads the entire upload into
    # memory before this function can object, so the 413 below was decorative: one
    # unauthenticated POST with a multi-GB body could exhaust the server's memory.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD:
        return JSONResponse({"error": "Recording too large."}, status_code=413)
    # A missing/lying content-length still has to be bounded, so read in chunks and stop
    # the moment the cap is passed rather than after the sender has finished.
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD:
            return JSONResponse({"error": "Recording too large."}, status_code=413)
        chunks.append(chunk)
    data = b"".join(chunks)
    if len(data) < 8000:
        return JSONResponse({"error": "Recording too short — please record ~10–15 seconds of clear speech."},
                            status_code=400)
    async with _brand_lock:               # serialise: fixed data/ paths + global _brand
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        ref_raw, ref_clean = data_dir / "brandvoice_raw.wav", data_dir / "brandvoice.wav"
        ref_raw.write_bytes(data)
        try:
            from .tts import clean_reference
            dur = await run_in_threadpool(clean_reference, str(ref_raw), str(ref_clean))
            if dur < 4:
                return JSONResponse({"error": "Not enough clear speech — record ~10–15s in a quiet spot and speak steadily."},
                                    status_code=400)
            clone = await run_in_threadpool(_load_clone, str(ref_clean))
            sample = "Hello, this is your brand voice. Thank you for calling — how can I help you today?"
            audio = await run_in_threadpool(clone.synth, sample)
        except Exception:  # noqa: BLE001
            log.exception("clone failed")
            return JSONResponse({"error": "Cloning failed — please try a cleaner recording."}, status_code=500)
        _brand["tts"] = clone
    log.info("brand voice cloned from %.1fs of cleaned reference", dur)
    return {"ok": True, "text": sample, "audio": _b64(audio)}


@app.get("/health")
def health():
    ok, info = _llm.health()
    return JSONResponse({
        "ollama": ok, "models": info if ok else [], "error": None if ok else info,
        "model": cfg["llm"]["model"], "streaming": STREAMING,
        "stt": cfg.get("stt", {}).get("provider"), "tts": cfg.get("tts", {}).get("provider"),
        "stt_loaded": _stt is not None, "tts_loaded": _tts is not None,   # actually loaded, not just configured
        "sessions": _sessions["n"], "max_sessions": MAX_SESSIONS,
        "packs": list_packs(),
    })


# --------------------------------------------------------------------------- #
# WebSocket voice loop
# --------------------------------------------------------------------------- #
async def _preload_backchannel(sock, session):
    """Ship the listening noise to the client once, at the start of the call.

    It has to play the instant the caller pauses mid-sentence, so asking the server for
    it at that moment would arrive after the gap it exists to fill. Best-effort: a call
    with no backchannel is exactly today's behaviour, so any failure here is silent.
    """
    try:
        phrase = _backchannel_for(session.reply_language(""))
        if not phrase:
            return
        audio = await run_in_threadpool(session.tts_bytes, phrase)
        if audio:
            # RECURRING, not rolling: the client holds this and plays it whenever the
            # caller pauses mid-sentence, which may be minutes from now.
            session.note_recurring_audio(audio)
            await _send_audio(sock, {"type": "backchannel", "text": phrase}, audio)
    except Exception as e:  # noqa: BLE001
        log.debug("backchannel preload skipped: %s", e)


async def _send_greeting(sock, session, pack_name):
    greet = session.greeting()
    audio = await run_in_threadpool(session.tts_bytes, greet)
    await _send_audio(sock, {"type": "greeting", "pack": pack_name, "text": greet}, audio, session=session)
    await _preload_backchannel(sock, session)


async def _speak_chunk(sock, session, text, seq):
    audio = await run_in_threadpool(session.tts_bytes, text)
    await _send_audio(sock, {"type": "chunk", "seq": seq, "text": text}, audio, session=session)


async def _stream_turn(sock, session, user_text, heard, precomputed=None):
    """Stream the LLM and speak each finished sentence, with THREE stages pipelined:
      * generate — consume the LLM token stream, cut it into sentences;
      * synthesise — each sentence is TTS'd in its own task (up to SYNTH_WORKERS at
        once), so several render concurrently instead of one-at-a-time;
      * send — a single sender awaits the synth tasks IN ORDER and ships each frame,
        so audio always reaches the caller in the right sequence.
    While TTS renders sentence N the LLM is already generating N+1 and N+2 are queued
    for synthesis — TTS time is off the critical path for all but the last sentence.
    """
    t0 = time.perf_counter()
    await sock.send_json({"type": "reply_start", "heard": heard})

    # deterministic safety escalation — no model call
    session.filled_last_turn, session.filled_this_turn = session.filled_this_turn, False
    sc = session.begin_user(user_text)

    # THE FAST PATH. Some questions are asked almost word-for-word, constantly —
    # "what are your timings", "what is the consultation fee" — and the pack already
    # carries the answer written out. Speaking it skips the model entirely: it cannot
    # invent, so the grounding guard has nothing to catch; it costs no generation, which
    # matters most in the languages that inflate 6.2×; and it is deterministic, so the
    # same question gets the same answer every time.
    #
    # It DECLINES unless the match is unmistakable. Answering the neighbouring question
    # confidently is worse than taking the slower path — see semantic.DIRECT_SCORE.
    if not sc:
        quick, why = await run_in_threadpool(session.quick_answer, user_text)
        if quick:
            log.info("fast path: answered from the pack, no model call (%s)", why)
            # Speak it SENTENCE BY SENTENCE, the same way a generated reply is spoken.
            # The first version handed the whole answer to TTS in one call and waited
            # 1724ms before the caller heard anything — the pack's answers carry real
            # detail ("open Monday to Saturday, 9am to 8pm, with a lunch break…") and
            # Kokoro scales with the text. Skipping the model is worthless if the saving
            # is spent waiting for one long synthesis. The first chunk breaks on clauses
            # too, which is what makes the opening arrive fast.
            pos, seq = 0, 0
            while pos < len(quick):
                chunk, pos = next_chunk(quick, pos, clause=(seq == 0), final=True)
                if not chunk:
                    break
                seq += 1
                await _speak_chunk(sock, session, chunk, seq)
            session.finalize(json.dumps({"kind": "answer", "say": quick,
                                         "action": {"type": "none"}},
                                        ensure_ascii=False))
            await sock.send_json({"type": "reply_end", "text": quick,
                                  "action": {"type": "none"}, "escalated": False,
                                  "timings": {"total_ms": _ms(t0), "fast_path": True}})
            _record_turn(call_id_var.get(), session.pack.get("id"), user_text, quick,
                         session.reply_language(user_text), _ms(t0))
            return
    if sc:
        await _speak_chunk(sock, session, sc["say"], 1)
        await sock.send_json({"type": "reply_end", "text": sc["say"], "action": sc["action"],
                              "escalated": True, "timings": {"total_ms": _ms(t0)}})
        _record_turn(call_id_var.get(), session.pack.get("id"), user_text, sc["say"],
                     session.reply_language(user_text), _ms(t0))
        return

    # optional speculative filler played IMMEDIATELY (opt-in; off by default)
    if FILLER and heard:
        fill = await run_in_threadpool(session.tts_bytes, FILLER)
        await _send_audio(sock, {"type": "chunk", "seq": 0, "text": FILLER, "filler": True}, fill, session=session)

    model = _route_model(session, user_text)
    task_q: asyncio.Queue = asyncio.Queue()
    st = {"buf": "", "t_first": None}
    sem = asyncio.Semaphore(SYNTH_WORKERS)
    synth_tasks: list = []
    stream_producers: list = []     # (task, stop-event) for progressive renderers
    spoke = {"yet": False}          # true once the FIRST real audio chunk has been sent

    async def slow_filler():
        """If the reply is slow (cold model / knowledge lookup), speak a short natural
        'let me check…' after SLOW_FILLER_S so the caller never hears dead air.

        Skipped when the PREVIOUS turn already used one. On a CPU box every reply is
        slower than the threshold, so without this the caller hears "let me check your
        details" before every single answer — which stops sounding like thinking and
        starts sounding like a stuck record.
        """
        try:
            await asyncio.sleep(SLOW_FILLER_S)
            if spoke["yet"] or session.filled_last_turn:
                return
            phrase = _pick_filler(session.reply_language(user_text))
            audio = await run_in_threadpool(session.tts_bytes, phrase)
            if not spoke["yet"]:        # re-check: real audio may have raced in during synth
                session.filled_this_turn = True
                await _send_audio(sock, {"type": "chunk", "seq": 0, "text": phrase, "filler": True}, audio, session=session)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("filler skipped: %s", e)

    filler_task = asyncio.create_task(slow_filler()) if SLOW_FILLER_S > 0 else None

    async def synth_one(seq, text):
        """Render one approved chunk.

        A provider that can render progressively (see Session.tts_stream) hands back a
        QUEUE the sender drains frame-by-frame, so audio starts leaving the server
        before the sentence has finished rendering. Everything else returns one
        finished WAV exactly as before — no provider is made slower by this path.
        """
        await sem.acquire()                               # bound concurrent synthesis
        stream = session.tts_stream(text) if PCM_STREAMING else None
        if stream is None:
            try:
                audio = await run_in_threadpool(session.tts_bytes, text)
            finally:
                sem.release()
            return seq, text, audio, None

        q: asyncio.Queue = asyncio.Queue(maxsize=STREAM_QUEUE)
        stop = threading.Event()
        loop = asyncio.get_running_loop()

        def produce():
            # Bounded queue = backpressure: a provider faster than the socket blocks
            # here instead of buffering the whole reply in memory.
            try:
                for pcm, sr in stream:
                    if stop.is_set():
                        return
                    asyncio.run_coroutine_threadsafe(q.put((pcm, sr)), loop).result()
            except Exception as e:  # noqa: BLE001
                log.warning("tts stream failed mid-render (%.40s): %s", text, e)
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop).result(timeout=5)
                except Exception:  # noqa: BLE001
                    pass

        prod = asyncio.create_task(run_in_threadpool(produce))
        prod.add_done_callback(lambda _t: sem.release())
        stream_producers.append((prod, stop))
        return seq, text, None, q

    def launch_synth(seq, text):
        t = asyncio.create_task(synth_one(seq, text))
        synth_tasks.append(t)
        return t

    def abort_synth():
        """Barge-in / error: stop every renderer, including the ones inside threads."""
        for t in synth_tasks:
            t.cancel()
        for prod, stop in stream_producers:
            stop.set()
            prod.cancel()

    # Nothing is synthesised until it has been checked. `allowed` is fixed for the turn
    # (the caller can't tell us a new number mid-reply), so build it once.
    allowed = session.allowed_numbers()
    # blocked = stop synthesising this reply; frozen = stop collecting it too (a retry
    # has already replaced the buffer that `finalize` will read).
    gate = {"kind": None, "blocked": False, "frozen": False, "spoken": None}
    llm_meta: dict = {}          # filled with the model's finish_reason when the stream ends

    def hallucinated(chunk: str) -> bool:
        """A number in this sentence that appears in neither the pack nor the call."""
        if not (GUARD.enabled and GUARD.check_numbers and not GUARD.log_only):
            return False
        bad = ungrounded_numbers(chunk, allowed)
        if bad:
            log.warning("guard: ungrounded number(s) %s — not speaking: %s",
                        ",".join(bad[:4]), chunk[:100])
        return bool(bad)

    async def generate():
        """LLM stream → ordered synth tasks pushed onto task_q as sentences complete.

        Every sentence passes the guard BEFORE it is handed to TTS: on a voice call an
        invented price cannot be taken back once it has been spoken. When a turn is
        blocked we speak one pre-written line in the caller's language and then drain
        the rest of the stream in silence, so `finalize` still sees the whole reply and
        the transcript matches the audio exactly.
        """
        sent_upto, first_done, seq = 0, False, 0

        async def block_with(kind: str):
            # recovery_line, not safe_say: mid-booking the caller is answering OUR
            # questions, so "I'll check with the desk" abandons the booking. Ask for
            # what's still missing instead. Same decision as the blocking path.
            nonlocal seq
            gate["blocked"] = True
            line = session.recovery_line(kind)
            gate["spoken"] = line          # what the caller hears — see finalize(spoken=)
            seq += 1
            await task_q.put(launch_synth(seq, line))

        retry_after_stream = {"want": False}

        async def _tokens():
            """The reply, token by token — from the model, or from a guess we already
            generated while the caller was still pausing.

            Replayed through the SAME loop rather than short-circuiting it, so the guard,
            the sentence splitter, the TTS pipeline and the history all behave
            identically. A second code path for the fast case is how the fast case ends
            up subtly different from the slow one.
            """
            if precomputed:
                llm_meta.setdefault("done_reason", "stop")
                llm_meta["speculative"] = True
                yield precomputed
                return
            async for d in session.llm.astream(
                    session.call_messages(), force_json=True, model=model,
                    num_predict=session.reply_budget(user_text), meta=llm_meta,
                    num_ctx=session.context_budget(user_text)):
                yield d

        async for delta in _tokens():
            if st["t_first"] is None:
                st["t_first"] = time.perf_counter()
            if not gate["frozen"]:
                st["buf"] += delta
            if gate["blocked"]:
                continue                       # drain silently — its words are not spoken
            # The model's own verdict arrives FIRST (the prompt puts "kind" ahead of
            # "say"), so a turn it can't answer honestly is intercepted before it has
            # written a single word — and answered faster than a normal reply.
            if gate["kind"] is None:
                k = extract_kind(st["buf"])
                if k:
                    gate["kind"] = k
                    if k in _KIND_TO_SAFE:
                        log.info("guard: model self-reported '%s' — speaking the safe line", k)
                        await block_with(k)
                        continue
            say, _done = extract_say(st["buf"])
            if say is None:
                continue
            # Parroting the PREVIOUS turn. Caught from the opening words, before any of
            # it is synthesised — on a call you cannot un-say it.
            if GUARD.enabled and GUARD.check_repetition and not GUARD.log_only \
                    and not spoke["yet"] and session.looks_like_repeat_start(say):
                log.warning("guard: reply repeats the previous turn — cutting it off")
                await block_with("unclear")
                continue
            # Answering AS the caller ("मेरा नाम मिश्रा है"). Caught while the sentence is
            # still forming, before any of it is synthesised — on a call you cannot
            # un-say it. Waits for enough words that a real reply can't trip it.
            if GUARD.enabled and not GUARD.log_only and not spoke["yet"] \
                    and len(say) >= 12 and looks_like_echo(say, user_text,
                                                           session.allowed_caller_numbers()):
                log.warning("guard: reply echoes the caller in their own voice — cutting it off")
                await block_with("unclear")
                continue
            # Watch the reply take shape: a model looping on one clause is caught as the
            # SECOND copy appears, so the caller never hears the repetition.
            if GUARD.enabled and GUARD.check_repetition and not GUARD.log_only \
                    and is_degenerate(say):
                log.warning("guard: reply degenerated into repetition — cutting it off")
                # Nothing has reached the caller's ear yet, so we can still try again
                # properly instead of retreating to "I'll check with the desk".
                #
                # The retry MUST NOT run here. Issuing a second, blocking request while
                # this stream is still open makes it queue behind the very generation we
                # have already given up on — Ollama serialises requests to one model, so
                # the "fast" recovery could not even start until the doomed reply had
                # finished looping. Leave the loop, close the stream, then retry.
                gate["blocked"] = True
                if not spoke["yet"]:
                    retry_after_stream["want"] = True
                break
            # first audio out fast (break on a comma); natural sentences after
            chunk, sent_upto = next_chunk(say, sent_upto, clause=not first_done, final=_done)
            if chunk:
                if hallucinated(chunk):
                    await block_with("")       # substitute the safe line, stop speaking
                    continue
                first_done = True
                seq += 1
                await task_q.put(launch_synth(seq, chunk))
        if retry_after_stream["want"]:
            # The generator above has been left; ask again on a clean connection.
            retry = await run_in_threadpool(session.retry_short, llm_meta)
            r_say, _ = extract_say(retry)
            if r_say and not is_degenerate(r_say):
                st["buf"] = retry            # what finalize reads = what we speak
                gate["frozen"] = True
                seq += 1
                await task_q.put(launch_synth(seq, r_say))
            else:
                await block_with("")
            return
        # flush any tail after the last boundary
        say, _ = extract_say(st["buf"])
        if say and sent_upto < len(say) and not gate["blocked"]:
            tail = say[sent_upto:].strip()
            if tail and hallucinated(tail):
                # the safe line here too, so the spoken audio still matches the
                # transcript `finalize` is about to produce
                await block_with("")
            elif tail:
                seq += 1
                await task_q.put(launch_synth(seq, tail))

    async def generate_guarded():
        """Guarantee the sentinel.

        `generate` pushes its end-of-stream sentinel on its LAST line, so any failure
        before that — Ollama down, a read timeout, malformed JSON mid-stream — left the
        consumer blocked on task_q.get() FOREVER: the turn never finished, reply_end was
        never sent, and the caller heard nothing for the rest of the call. A blocked
        get() never raises, so none of the except handlers below could fire either.
        The sentinel now goes out on every path; `await gen` then re-raises the real
        error into the handler that knows how to fall back.
        """
        try:
            await generate()
        finally:
            task_q.put_nowait(None)

    gen = asyncio.create_task(generate_guarded())
    seq = 0
    try:
        while True:
            t = await task_q.get()
            if t is None:
                break
            seq, text, audio, q = await t                 # await synth results IN ORDER
            if filler_task:
                filler_task.cancel()
            if q is None:                                 # whole-clip provider
                spoke["yet"] = True                       # real audio → filler not needed
                await _send_audio(sock, {"type": "chunk", "seq": seq, "text": text}, audio, session=session)
                continue
            # Progressive provider: ship frames as they render. Still strictly ordered —
            # this chunk is drained to completion before the next one is awaited.
            first = True
            while True:
                item = await q.get()
                if item is None:
                    break
                pcm, sr = item
                spoke["yet"] = True
                await _send_audio(sock, {"type": "pcm", "seq": seq, "sr": sr,
                                         "text": text if first else None,
                                         "start": first}, pcm, session=session)
                first = False
            if first:      # yielded nothing — synthesise normally so the turn isn't silent
                audio = await run_in_threadpool(session.tts_bytes, text)
                spoke["yet"] = True
                await _send_audio(sock, {"type": "chunk", "seq": seq, "text": text}, audio, session=session)
        await gen
    except (asyncio.CancelledError, WebSocketDisconnect):
        gen.cancel()
        abort_synth()
        raise                                             # barge-in / client gone
    except Exception as e:  # noqa: BLE001
        gen.cancel()
        abort_synth()
        if not spoke["yet"]:
            raise                                         # nothing spoken yet → let caller fall back
        # already spoke part of the reply → do NOT replay the whole thing via fallback;
        # gracefully finish with whatever was generated.
        log.warning("stream errored after partial audio; finishing with partial reply: %s", e)
    finally:
        if filler_task:
            filler_task.cancel()

    # finalize() writes to SQLite (booking) — run off the event loop so a slow/locked
    # DB write can't stall every other active call.
    # finalize() writes the booking to SQLite. A locked DB or a full disk raised HERE,
    # AFTER audio had already been sent — the blanket handler then re-ran the whole turn
    # through _fallback_turn and the caller heard the SAME reply twice. If the condition
    # persisted the retry raised too and the turn ended with no reply_end at all, leaving
    # the browser bubble open forever. Never replay spoken audio because a DB write failed.
    try:
        result = await run_in_threadpool(session.finalize, st["buf"], llm_meta,
                                         gate.get("spoken"))
    except Exception as e:  # noqa: BLE001
        log.exception("finalize failed (%s) — closing the turn without replaying audio", e)
        say, _ = extract_say(st["buf"])
        if not (say or "").strip():
            say = session.safe_say("unknown")
        if not spoke["yet"]:
            seq += 1
            await _speak_chunk(sock, session, say, seq)
        await sock.send_json({"type": "reply_end", "text": say,
                              "action": {"type": "none"}, "escalated": session.escalated,
                              "timings": {"total_ms": _ms(t0)},
                              "mute_reason": session.mute_reason})
        return
    # Nothing was ever spoken, but finalize still produced words — the model returned
    # something with no parseable "say", so the stream had nothing to synthesise and
    # finalize substituted a safe line. Without this the caller sat in SILENCE while the
    # UI showed text. Say it.
    if not spoke["yet"] and (result.get("say") or "").strip():
        seq += 1
        log.info("nothing was spoken during the stream — voicing the finalised reply")
        await _speak_chunk(sock, session, result["say"], seq)
        spoke["yet"] = True
    # finalize() may append a booking reference to the reply after streaming — voice it.
    say, _ = extract_say(st["buf"])
    bid = (result.get("action") or {}).get("booking_id")
    if bid and f"#{bid}" not in (say or ""):
        seq += 1
        await _speak_chunk(sock, session, session.booking_ref(bid), seq)   # caller's language

    timings = {"llm_first_token_ms": _ms(t0, st["t_first"]) if st["t_first"] else None,
               "total_ms": _ms(t0)}
    if LOG_TIMINGS:
        log.info("turn: first_token=%sms total=%sms | %s",
                 timings["llm_first_token_ms"], timings["total_ms"], result["say"][:60])
    await sock.send_json({"type": "reply_end", "text": result["say"], "action": result["action"],
                          "escalated": result["escalated"], "timings": timings,
                          "mute_reason": session.mute_reason})
    _record_turn(call_id_var.get(), session.pack.get("id"), user_text, result["say"],
                 session.reply_language(user_text), timings["total_ms"])


async def _fallback_turn(sock, session, text=None, heard=""):
    """Robust non-streaming turn (used when streaming is off or errors out)."""
    t0 = time.perf_counter()
    res = await run_in_threadpool(session.turn, text, None)
    res.setdefault("timings", {})["total_ms"] = _ms(t0)
    # echo `heard` only for voice turns (heard set); never for typed text
    audio = base64.b64decode(res["audio_b64"]) if res.get("audio_b64") else None
    await _send_audio(sock, {"type": "reply", "heard": heard, "text": res["say"],
                             "action": res.get("action", {}),
                             "escalated": res.get("escalated", False),
                             "timings": res["timings"]}, audio, session=session)
    _record_turn(call_id_var.get(), session.pack.get("id"), text or heard, res["say"],
                 session.reply_language(text or heard or ""), res["timings"].get("total_ms"))


async def _turn_guarded(sock, session, text, heard, precomputed=None):
    """Run a turn as a cancellable task. On barge-in the task is cancelled;
    we keep the conversation history paired so the next turn stays coherent."""
    try:
        await _run_turn(sock, session, text=text, heard=heard,
                        precomputed=precomputed)
    except asyncio.CancelledError:
        session.note_interrupted()
        return


@app.websocket("/ws")
async def ws(sock: WebSocket, pack: str = Query(None), sid: str = Query(None)):
    await sock.accept()
    pack_name = pack or cfg.get("default_pack", "clinic")
    _prune_stash()
    stashed = _stash.pop(sid, None) if sid else None
    resumed = bool(stashed and stashed["expires"] >= time.time())

    if resumed:                                      # reconnect → restore the SAME call
        # A resume still occupies a slot, so it has to be counted. Skipping the check
        # here let the cap be walked straight past: reconnect with a stashed sid and you
        # are admitted no matter how full the server is. The allowance is deliberate but
        # BOUNDED — a caller mid-conversation should beat a brand-new one, not everyone.
        if _sessions["n"] >= MAX_SESSIONS + RESUME_HEADROOM:
            await sock.send_json({"type": "error",
                                  "error": "We're at capacity right now — please try again shortly."})
            await sock.close()
            return
        session = stashed["session"]
        set_call_id(sid[:6])
    else:
        if _sessions["n"] >= MAX_SESSIONS:           # capacity applies only to NEW calls
            await sock.send_json({"type": "error", "error": "We're at capacity right now — please try again shortly."})
            await sock.close()
            return
        sid = sid or secrets.token_hex(6)
        set_call_id(sid[:6])
        try:
            session = _new_session(pack_name)
        except Exception as e:  # noqa: BLE001
            log.warning("session init failed: %s", e)
            await sock.send_json({"type": "error", "error": "That business isn't available."})
            await sock.close()
            return
    _sessions["n"] += 1
    log.info("call %s (pack=%s, active=%d)", "resumed" if resumed else "started", pack_name, _sessions["n"])

    task = {"cur": None}
    # Speculative transcription state (see the `stt_hint` / `commit` messages below).
    # `armed`: the NEXT binary frame is a guess, not a turn. `text`: its transcript,
    # held until the client confirms the caller really had finished.
    spec = {"armed": False, "text": None, "incomplete": False,
        "settled": False, "prompted": False,
        # a reply generated against a guess, plus the guess it answers. Adopted only
        # if the committed transcript is IDENTICAL — a reply to a sentence the caller
        # had not finished is worse than waiting for the one they did.
        "gen": None, "gen_for": None,
        # the next frame is a latched salvage — see the stt_hint handler
        "latched": False}

    async def cancel_current():
        t = task["cur"]
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        task["cur"] = None
        # We have stopped talking. Whatever we were half-way through saying is still in
        # the echo reference, and the caller's interrupting words OVERLAP it — so a
        # stale tail could explain away the very turn that interrupted us. `reset()`
        # was written for exactly this and nothing was calling it.
        try:
            session.echo.reset()
        except Exception:  # noqa: BLE001
            pass

    async def launch(text, heard, precomputed=None):
        await cancel_current()   # barge-in: abandon any in-flight reply
        task["cur"] = asyncio.create_task(
            _turn_guarded(sock, session, text, heard, precomputed))

    idle_task = None                     # declared here so `finally` can always stop it
    try:
        # greeting is inside the try so a client that disconnects immediately
        # (before/while it's sent) exits quietly instead of logging a traceback.
        await sock.send_json({"type": "session", "sid": sid, "resumed": resumed,
                              "denoise_available": bool(getattr(_stt, "denoiser", None)),
                              "isolate_available": bool(_diarizer and _speaker_gate)})
        await sock.send_json({"type": "voiceinfo", "voice": session.voice})
        if not resumed:                     # a resumed call already has its history — don't re-greet
            await _send_greeting(sock, session, pack_name)

        # ---- the caller who says nothing ------------------------------------
        # Anything arriving from the client counts as presence — audio, a control
        # frame, even a toggle. Only real silence advances these.
        idle = {"since": time.monotonic(), "prompted": False}

        async def _watch_idle():
            while True:
                await asyncio.sleep(1.0)
                if not IDLE_PROMPT_S and not IDLE_HANGUP_S:
                    return
                # A turn in flight IS the call being alive. The timer only reset on
                # messages FROM the client, so while the agent was thinking or speaking
                # — 20s+ on an Indic reply — the clock kept running and it hung up on
                # itself mid-sentence. Caught by the concurrency sweep, where turns are
                # slow enough to trip it every time.
                if task["cur"] and not task["cur"].done():
                    idle["since"] = time.monotonic()
                    idle["prompted"] = False
                    continue
                quiet = time.monotonic() - idle["since"]
                try:
                    if IDLE_HANGUP_S and quiet >= IDLE_HANGUP_S:
                        line = session.safe_say("goodbye")
                        log.info("call idle %.0fs — saying goodbye and closing", quiet)
                        audio = await run_in_threadpool(session.tts_bytes, line)
                        await _send_audio(sock, {"type": "reply", "text": line, "heard": "",
                                                 "action": {"type": "none"},
                                                 "timings": {}}, audio, session=session)
                        await asyncio.sleep(2.5)      # let it actually play
                        await sock.close()
                        return
                    if (IDLE_PROMPT_S and not idle["prompted"]
                            and quiet >= IDLE_PROMPT_S):
                        idle["prompted"] = True
                        line = session.safe_say("still_there")
                        log.info("call idle %.0fs — asking if the caller is there", quiet)
                        audio = await run_in_threadpool(session.tts_bytes, line)
                        await _send_audio(sock, {"type": "reply", "text": line, "heard": "",
                                                 "action": {"type": "none"},
                                                 "timings": {}}, audio, session=session)
                except Exception:  # noqa: BLE001  (socket gone, or shutting down)
                    return

        idle_task = _spawn(_watch_idle()) if (IDLE_PROMPT_S or IDLE_HANGUP_S) else None

        while True:
            msg = await sock.receive()
            idle["since"] = time.monotonic()   # they are still with us
            idle["prompted"] = False
            if msg.get("type") == "websocket.disconnect":   # client closed — exit cleanly
                await cancel_current()
                return

            # A single malformed frame must never kill the call — guard the whole body.
            try:
                # ---- text / control JSON ----
                if msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    mtype = data.get("type")
                    if mtype == "text":
                        await launch(str(data.get("text", ""))[:2000], None)
                    elif mtype == "lang":          # set STT + reply language (null = auto-detect)
                        session.lang = data.get("lang")
                        session.lang_lock = None    # re-detect fresh when the caller changes this
                        session._lang_cand = None
                        session._lang_cand_n = 0
                        session._last_det = None
                    elif mtype == "stt_mode":       # "fast" = greedy+no-denoise; else accurate
                        session.stt_fast = (data.get("mode") == "fast")
                    elif mtype == "isolate_mode":
                        # Split the turn by speaker and transcribe only the caller's parts.
                        # Measured: rescues turns the whole-utterance gate would discard
                        # (similarity 0.364 -> 0.872) and removes the intruder's words.
                        # `on` may be null, meaning AUTO — let the pipeline decide per
                        # turn. bool(None) is False, which silently turned the UI's
                        # default state into a hard OFF.
                        v = data.get("on")
                        session.isolate_on = None if v is None else bool(v)
                        log.info("voice isolation %s for this call",
                                 "AUTO" if session.isolate_on is None
                                 else ("ON" if session.isolate_on else "off"))
                    elif mtype == "denoise_mode":
                        # Live A/B of DeepFilterNet in front of Whisper. Measured to make
                        # transcription worse (see zensuvidha/denoise.py); exposed so the
                        # trade-off can be heard rather than argued about.
                        v = data.get("on")
                        session.stt_denoise = None if v is None else bool(v)
                        log.info("DeepFilterNet %s for this call",
                                 "AUTO (per-turn, on the room's noise floor)"
                                 if session.stt_denoise is None
                                 else ("ON" if session.stt_denoise else "off"))
                    elif mtype == "model":          # switch LLM (e.g. qwen2.5:3b = smaller/faster)
                        was = session.model
                        session.model = data.get("model") or None
                        # Warm ONLY on a genuine change, and with THIS call's real prompt.
                        # Re-warming the same model with a stub prompt evicted the cached
                        # prefix every real turn depends on, so the next caller paid the
                        # full ~20s prompt evaluation again. A browser reconnect sends
                        # this message, which is how it happened in practice.
                        # ...and never for the model the server already has loaded and
                        # prompt-cached at startup. A browser sends `model` on every
                        # connect with whatever is selected in the dropdown — usually the
                        # default — so this fired a redundant 2.6s warmup that competed
                        # with the caller's own first turn for CPU.
                        if (session.model and session.model != was
                                and session.model != getattr(_llm, "model", None)):
                            _spawn(run_in_threadpool(_llm.warmup, session.model,
                                                     session.call_messages()))
                    elif mtype == "cancel":         # pure barge-in signal (user started speaking)
                        spec["armed"] = False
                        spec["text"] = None
                        _void_speculation(spec)
                        await cancel_current()
                    elif mtype == "stt_hint":
                        # "spec": the next audio frame is a GUESS — the caller has gone
                        # quiet but may only be pausing. Transcribe it now so the words
                        # are ready if they really were done; say nothing yet.
                        # anything else: they resumed, so the guess is void.
                        if SPECULATIVE_STT and data.get("mode") == "spec":
                            spec["armed"] = True
                        elif data.get("mode") == "latched":
                            # The client hit its latch guard — 7s with no pause — and is
                            # offering the audio instead of discarding it. A caller
                            # talking OVER continuous noise never gives a clean pause
                            # either, so they land here too and used to be thrown away
                            # and asked to repeat, into the same noise. Isolation exists
                            # to pull one voice out of exactly that.
                            spec["armed"] = False
                            spec["text"] = None
                            spec["latched"] = True
                            _void_speculation(spec)
                        else:
                            spec["armed"] = False
                            spec["text"] = None
                            _void_speculation(spec)
                    elif mtype == "commit":
                        # Endpoint confirmed. If a speculative transcript is waiting, the
                        # turn starts with ZERO STT on the clock.
                        text, spec["text"], spec["armed"] = spec["text"], None, False
                        unfinished = spec.pop("incomplete", False)
                        # Only nudge when we have something too SHORT to act on. A long
                        # transcript that merely trails off ("...my number, I will give you
                        # later, list me out all the...") is a truncated sentence, not a
                        # caller who stopped: answering it beats asking them to repeat.
                        # Observed live: three nudges in a row, then "sorry, I didn't catch
                        # that" when the repeat-guard fired on our own nudge.
                        brief = len((text or "").split()) <= 6
                        if text and unfinished and brief and not spec.get("prompted"):
                            # They stopped mid-sentence. Answering the fragment is how a
                            # phone number arrived without its digits — invite them to
                            # finish instead. Once only, so a misread can't start a loop.
                            spec["prompted"] = True
                            # Every other speaking path in this loop cancels the in-flight
                            # turn first (launch, cancel, switch, voice, and the sibling
                            # `elif text` below). This one did not, so a commit arriving
                            # while a reply was still streaming produced TWO overlapping
                            # audio streams into the same socket.
                            await cancel_current()
                            # begin_user may answer deterministically (an escalation
                            # keyword, a service we don't offer). That outranks a nudge
                            # to keep talking, so honour it instead of speaking over it.
                            det = session.begin_user(text)
                            line = det["say"] if det else session.safe_say("go_on")
                            if not det:
                                # Pair the history ourselves. begin_user appended the
                                # caller's turn; leaving it without a reply gives the
                                # model two user messages in a row on the next call.
                                session._append("assistant",
                                                json.dumps({"say": line,
                                                            "action": {"type": "none"}},
                                                           ensure_ascii=False))
                                log.info("commit: transcript sounds unfinished — prompting to continue")
                            audio = await run_in_threadpool(session.tts_bytes, line)
                            # ONE frame carrying both text and audio: a separate
                            # _speak_chunk would render the line twice in the transcript.
                            await _send_audio(sock, {
                                "type": "reply", "text": line, "heard": text,
                                "action": det["action"] if det else {"type": "none"},
                                "escalated": bool(det and det.get("escalated")),
                                "timings": {}}, audio, session=session)
                            _record_turn(call_id_var.get(), session.pack.get("id"),
                                         text, line, session.reply_language(text), None)
                        elif text:
                            spec["prompted"] = False
                            # Was this exact sentence already answered while they paused?
                            pre = None
                            gen, gen_for = spec.get("gen"), spec.get("gen_for")
                            if gen is not None and gen_for == text:
                                try:
                                    pre = await gen
                                except asyncio.CancelledError:
                                    pre = None
                                except Exception as e:  # noqa: BLE001
                                    log.debug("speculative reply failed (%s) — "
                                              "generating normally", e)
                                    pre = None
                                if pre:
                                    log.info("commit: the reply was already generated "
                                             "while the caller paused")
                            elif gen is not None:
                                # A DIFFERENT sentence arrived. The work is void, and
                                # keeping it would answer words nobody said.
                                log.info("commit: transcript changed after the guess — "
                                         "discarding the speculative reply")
                            _void_speculation(spec)
                            log.info("commit: speculative transcript used (%r)",
                                     text[:60] if LOG_TRANSCRIPTS else f"<{len(text)} chars>")
                            await launch(text, text, precomputed=pre)
                        else:
                            # NO usable guess — and the client took `commit` to mean the
                            # words were already here, so it discarded the recording. Both
                            # sides then wait for the other and the call hangs on
                            # "thinking" with no reply, no error and no way out. Observed
                            # live with music playing: `spec STT → heard=''` fifteen times
                            # in one call, every commit after one of them a lost turn.
                            # Ask for the audio back instead of dropping it.
                            log.info("commit: no speculative transcript — asking the "
                                     "client to send the recording")
                            await sock.send_json({"type": "commit_miss"})
                    elif mtype == "switch":
                        await cancel_current()
                        pack_name = data.get("pack", pack_name)
                        session = _new_session(pack_name)
                        await sock.send_json({"type": "voiceinfo", "voice": session.voice})
                        await _send_greeting(sock, session, pack_name)
                    elif mtype == "voice":
                        await cancel_current()
                        v = data.get("voice")
                        if v == "__brand__":
                            session.clone = _brand["tts"]
                            if not session.clone:
                                await sock.send_json({"type": "error",
                                                      "error": "No brand voice yet — record one first."})
                                continue
                            session.voice = "__brand__"
                        elif isinstance(v, str) and v.startswith("parler:"):
                            try:                    # AI4Bharat Indic-Parler, e.g. "parler:Hindi:Divya"
                                parts = v.split(":")
                                speaker = parts[2] if len(parts) > 2 else parts[-1]
                                await sock.send_json({"type": "voiceinfo", "voice": v, "loading": True})
                                session.indic = await run_in_threadpool(_indic_voice, speaker)
                                session.voice = v
                            except Exception as e:  # noqa: BLE001
                                log.warning("indic voice load failed: %s", e)
                                gated = "gated" in str(e).lower() or "401" in str(e)
                                await sock.send_json({"type": "error", "error": (
                                    "AI4Bharat's model is a gated Hugging Face repo — make a free HF "
                                    "account, accept the terms at huggingface.co/ai4bharat/indic-parler-tts, "
                                    "run `huggingface-cli login`, and restart." if gated
                                    else "That voice couldn't be loaded. Please try another.")})
                                continue
                        else:
                            session.voice = v or session.pack.get("voice")
                        sample = "Hello, this is how I'll sound. How can I help you today?"
                        a = await run_in_threadpool(session.tts_bytes, sample)
                        await _send_audio(sock, {"type": "voice_sample", "voice": session.voice,
                                                 "text": sample}, a, session=session)

                # ---- binary mic audio (fed straight to STT — no disk round-trip) ----
                elif msg.get("bytes") is not None:
                    raw = msg["bytes"]
                    speculative = spec["armed"]
                    # Reset BEFORE any early exit: leaving it armed made the next real
                    # turn get held as a guess and never answered.
                    spec["armed"] = False
                    if len(raw) > 8_000_000:        # ~8 MB ≈ 4 min of 16k mono
                        # Dropping this silently made the caller's turn vanish with no
                        # reply at all. Say something rather than nothing.
                        log.warning("mic frame too large (%s bytes) — asking for a repeat", len(raw))
                        if speculative:
                            await _dropped(sock, "frame too large")
                        if not speculative:
                            line = session.safe_say(session.repair_kind())
                            audio = await run_in_threadpool(session.tts_bytes, line)
                            await _send_audio(sock, {"type": "reply", "text": line, "heard": "",
                                                     "action": {"type": "none"},
                                                     "timings": {}}, audio, session=session)
                        continue
                    # Split the turn by speaker and keep only the caller's parts BEFORE
                    # transcription, so a colleague talking in a gap never reaches Whisper.
                    # Returns the original bytes whenever anything is uncertain.
                    salvage, spec["latched"] = spec.get("latched", False), False
                    if salvage and session.voiceprint is None:
                        # Nothing to trim against. Isolation cannot run, so passing it on
                        # would just hand Whisper the noise the guard exists for — and
                        # worse, ENROL it as the caller.
                        log.info("latched turn: no voiceprint yet, cannot isolate — "
                                 "asking the caller to repeat")
                        line = session.safe_say(session.repair_kind())
                        out = await run_in_threadpool(session.tts_bytes, line)
                        await _send_audio(sock, {"type": "reply", "text": line,
                                                 "heard": "", "action": {"type": "none"},
                                                 "timings": {}}, out, session=session)
                        continue

                    # Is this us? Judged on RAW audio, before isolation or denoising —
                    # a denoiser reshapes the echo enough to break the correlation, and
                    # isolation would happily trim our own voice into something
                    # unrecognisable. Browser calls have real AEC underneath, so this
                    # almost never fires there; a phone line has nothing else at all.
                    own, corr = await run_in_threadpool(session.is_own_echo, raw)
                    if own:
                        log.info("mic turn ignored: our own audio coming back "
                                 "(correlation %.2f)", corr)
                        # No _insight here: it is defined further down this handler and
                        # calling it from above would be a NameError on the one path
                        # that only fires in production. The drop is what the client
                        # needs — it entered "thinking" the moment it sent this.
                        await _dropped(sock, "our own echo")
                        continue

                    audio, dia = await run_in_threadpool(session.clean_audio, raw)
                    heard = ""
                    try:
                        heard = await run_in_threadpool(session.transcribe, audio)
                    except Exception as e:  # noqa: BLE001  (bad/short/garbled audio)
                        log.warning("STT decode failed (%s bytes): %s", len(audio), e)
                    # Tell the UI what each filtering stage actually did, so the
                    # inspector can show what reached Whisper and what was thrown away
                    # rather than asking anyone to take it on trust.
                    async def _insight(**kw):
                        if not (session.diarizer or session.speaker_gate
                                or session.stt_denoise):
                            return
                        d = dia or {}
                        await sock.send_json({
                            "type": "audio_insight",
                            "speculative": speculative,
                            # What the pipeline ACTUALLY did to this turn, not what
                            # the toggles allow it to do — the router decides per
                            # turn, so "enabled" and "used" are different answers.
                            "denoise": bool(d.get("denoised")),
                            "isolate": bool(d.get("isolated")),
                            "snr_db": d.get("snr_db"),
                            "speakers": d.get("speakers", 1),
                            "kept_s": d.get("kept_s") or 0.0,
                            "total_s": d.get("total_s") or 0.0,
                            "dia_reason": d.get("reason"),
                            "filter_ms": round(d.get("iso_ms", 0.0) + d.get("den_ms", 0.0)),
                            "enrolled": session.voiceprint is not None,
                            **kw})

                    # Whose voice was that? The first speaker on the call is enrolled;
                    # anyone else near the mic — a colleague, a television — is dropped
                    # here, before a transcript can become a turn. Off unless
                    # stt.speaker_gate is set, and it accepts whenever it can't judge.
                    # A SPECULATIVE frame is a 450ms partial snapshot taken while the
                    # caller may still be mid-sentence — it is a guess for the UI, not a
                    # turn. Enrolling on one produced a voiceprint from a fragment, after
                    # which the caller's own full sentences scored 0.01-0.08 against
                    # themselves and every turn was refused. Judge and enrol on COMMITTED
                    # audio only; a guess never mutates session state.
                    if heard and not speculative:
                        same, sim = await run_in_threadpool(
                            session.check_speaker, audio, (dia or {}).get("speakers"),
                            heard, not salvage)
                        if not same:
                            log.info("mic turn ignored: different speaker (sim=%.2f) %r",
                                     sim if sim is not None else -1,
                                     heard[:40] if LOG_TRANSCRIPTS else f"<{len(heard)} chars>")
                            await _insight(accepted=False, similarity=sim, heard=heard)
                            spec["text"] = None
                            await _dropped(sock, "different speaker")
                            continue
                        # `rescued` is set only when the gate refused on the audio and the
                        # TURN gave it back. The operator needs to see that, or a call
                        # that works for a reason they cannot observe looks like luck.
                        # Consumed here so it describes THIS turn and never leaks to the
                        # next one.
                        rescued, session._last_rescue = session._last_rescue, None
                        await _insight(accepted=True, similarity=sim, heard=heard,
                                       rescued=rescued)
                    if speculative:
                        # Hold it. Nothing reaches the LLM until the endpoint is
                        # confirmed — the guard grounds numbers against the caller's
                        # COMPLETE words, so a half-spoken phone number must never
                        # start a turn. The client gets it only to show live text.
                        spec["text"] = heard
                        # Now that there are WORDS, we can do better than counting
                        # silence: "मेरा मोबाइल नंबर" is a finished noun phrase and an
                        # unfinished sentence. Tell the client to keep listening.
                        want_phone = (session.pending_slot
                                      or session.last_asked_slot) == "phone"
                        spec["incomplete"] = bool(heard) and looks_incomplete(
                            heard, expect_phone=want_phone)
                        # …and the inverse. Silence cannot tell "finished" from
                        # "thinking", but a completed phone number or a bare "haan" can:
                        # nothing follows either, so waiting the full window for a word
                        # that is already done is most of what makes a call feel slow.
                        spec["settled"] = bool(heard) and looks_complete(
                            heard, expect_phone=want_phone)
                        log.info("spec STT: %s bytes → heard=%r%s", len(raw),
                                 heard if LOG_TRANSCRIPTS else f"<{len(heard)} chars>",
                                 " [sounds unfinished]" if spec["incomplete"] else "")
                        if heard:
                            await sock.send_json({"type": "partial", "text": heard,
                                                  "complete": not spec["incomplete"],
                                                  "settled": spec["settled"],
                                                  # what we ASKED for. People pause
                                                  # differently mid-phone-number than
                                                  # mid-sentence, and the window has
                                                  # been one number for everyone.
                                                  "expect": (session.pending_slot
                                                             or session.last_asked_slot)})
                        # Start answering it NOW, while they may still be pausing. Only
                        # when the words look FINISHED: replying to half a sentence is
                        # worse than waiting for the whole one, and a half-spoken phone
                        # number must never reach the guard, which grounds numbers
                        # against the caller's complete words.
                        _void_speculation(spec)
                        if (SPECULATIVE_REPLY and heard and not spec["incomplete"]
                                and session.llm is not None):
                            spec["gen_for"] = heard
                            spec["gen"] = asyncio.create_task(
                                _speculate_reply(session, heard))
                        continue
                    spec["text"] = None            # a full frame supersedes any guess
                    if heard:
                        spec["asked_repeat"] = False   # they got through — arm it again
                    # log_transcripts:false exists precisely so caller words stay OUT of
                    # the rotating log file. This line ignored it and wrote every turn —
                    # names, phone numbers, medical complaints — to disk regardless.
                    log.info("mic turn: %s bytes → heard=%r", len(raw),
                             heard if LOG_TRANSCRIPTS else f"<{len(heard)} chars>")
                    if heard:
                        await launch(heard, heard)
                    elif _stt is None:
                        # No STT at all (model missing or failed to load). Every turn of
                        # the whole call would otherwise be swallowed and the caller would
                        # talk into silence forever, with nothing anywhere saying why.
                        line = session.safe_say("repeat")
                        audio_out = await run_in_threadpool(session.tts_bytes, line)
                        await _send_audio(sock, {"type": "reply", "text": line, "heard": "",
                                                 "action": {"type": "none"},
                                                 "timings": {}}, audio_out, session=session)
                    else:
                        # Nothing recognisable in it. A short clip is a cough or a door
                        # and deserves silence — but a LONG one was somebody really
                        # trying to say something, and dropping it without a word leaves
                        # them repeating themselves into a machine that never reacts.
                        # Observed live with loud music playing: 12s and 15s clips both
                        # came back empty and the caller got nothing at all.
                        # Say so in the inspector too. It only ever reported turns that
                        # HAD a transcript, so the one case a caller most needs
                        # explained — "I spoke and nothing happened" — produced no row
                        # at all and the panel sat blank. Reported live with loud audio
                        # playing: every empty turn was invisible.
                        await _insight(accepted=False, similarity=None, heard="")
                        secs = len(raw) / 32000.0          # 16k mono PCM16
                        if secs >= REPEAT_ASK_S and not spec.get("asked_repeat"):
                            spec["asked_repeat"] = True    # once per run of failures
                            line = session.safe_say(session.repair_kind())
                            log.info("mic turn: %.1fs of audio, nothing recognisable — "
                                     "asking the caller to repeat", secs)
                            audio_out = await run_in_threadpool(session.tts_bytes, line)
                            await _send_audio(sock, {"type": "reply", "text": line,
                                                     "heard": "",
                                                     "action": {"type": "none"},
                                                     "timings": {}}, audio_out, session=session)
                        else:
                            # The client set itself to "thinking" the moment it sent this
                            # audio and only leaves that state on a reply — tell it the
                            # turn is over so the caller gets the microphone back.
                            await _dropped(sock, "nothing recognisable")

            except WebSocketDisconnect:
                raise
            except Exception:  # noqa: BLE001  (a bad control frame — keep the call alive)
                log.exception("control-message error (ignored)")
                continue

    except WebSocketDisconnect:
        await cancel_current()
        return
    except Exception:  # noqa: BLE001
        log.exception("ws loop error")
        try:
            await sock.send_json({"type": "error", "error": "Something went wrong — please try again."})
        except Exception:  # noqa: BLE001
            pass
    finally:
        # The idle watcher outlives the socket unless it is stopped — one leaked task
        # per call, each holding a Session and firing TTS into a closed connection.
        if idle_task:
            idle_task.cancel()
        # …and so does a speculative reply. Same failure class as the line above: a
        # client that drops mid-guess left an Ollama request running for a call that had
        # ended, competing with live callers for the model.
        _void_speculation(spec)
        await cancel_current()
        _sessions["n"] = max(0, _sessions["n"] - 1)
        # keep the session alive briefly so a reconnect resumes it (no mid-call amnesia)
        if sid and not session.escalated:
            _stash[sid] = {"session": session, "expires": time.time() + RESUME_TTL}


async def _run_turn(sock, session, text, heard=None, precomputed=None):
    """Dispatch to streaming, with a safe fallback on any error.
    `heard` is the transcript for VOICE turns (echoed to the UI) and None/"" for
    typed text (the client already shows the typed bubble — don't echo it back)."""
    heard = heard or ""
    if STREAMING:
        try:
            await _stream_turn(sock, session, text, heard, precomputed)
            return
        except (asyncio.CancelledError, WebSocketDisconnect):
            raise                         # barge-in / client gone — do NOT re-run the turn
        except OllamaError as e:
            log.warning("stream failed, falling back: %s", e)
        except Exception as e:  # noqa: BLE001
            log.warning("stream error, falling back: %s", e)
        # The model is unreachable or broke mid-stream. Before dropping to the blocking
        # path — which needs the same model — see whether the pack simply contains the
        # answer. "I don't have that detail" is the wrong reply to a question written
        # down three lines away in the knowledge base.
        try:
            quick, why = await run_in_threadpool(session.quick_answer, text)
        except Exception:  # noqa: BLE001
            quick = None
        if quick:
            log.info("rescue: the model failed but the pack answers this (%s)", why)
            audio = await run_in_threadpool(session.tts_bytes, quick)
            await _send_audio(sock, {"type": "reply", "heard": heard, "text": quick,
                                     "action": {"type": "none"},
                                     "timings": {"rescued": True}}, audio,
                              session=session)
            return
    await _fallback_turn(sock, session, text=text, heard=heard)


def _ms(t0, t1=None):
    return round(((t1 or time.perf_counter()) - t0) * 1000)
