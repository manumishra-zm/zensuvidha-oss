"""Text-to-speech, provider-pluggable. All options are free and offline.

  system        — pyttsx3 (OS voices), zero download. Normalised to 16-bit WAV.  [default]
  piper         — small neural voice (auto-discovers a .onnx model in ./models).
  kokoro        — Apache-2.0 high quality (a little more compute).
  indic_parler  — AI4Bharat Indic-Parler-TTS: ~21 Indian languages + English, open-
                  source (Apache-2.0), offline. Voice/style set by a text description.
  clone         — Coqui XTTS zero-shot voice cloning (non-commercial license).
  none          — text only (no audio).

Robustness:
  * every provider's .synth(text) returns 16-bit WAV bytes (or None on failure).
  * results are LRU-cached (greetings/confirmations become instant on repeat).
  * pyttsx3 (not thread-safe) is serialised behind a lock.

Incremental synthesis (optional):
  * a provider MAY also expose .synth_stream(text, voice) -> iterator of
    (pcm_int16_bytes, sample_rate), yielding audio while the sentence is still
    rendering. The server ships those frames as they arrive, so the caller hears the
    first words instead of waiting for the whole clip.
  * this is strictly additive. A provider without it (macOS `say`, Piper, XTTS) keeps
    the whole-clip path unchanged, so nothing that works today gets slower.
"""
import hashlib
import io
import re
import logging
import os
import tempfile
import threading
from collections import OrderedDict

log = logging.getLogger("zensuvidha.tts")


class FallbackTTS:
    """Speak with `primary`; hand anything it cannot pronounce to `secondary`.

    No single open-source voice covers both "fast" and "all of India". Measured on
    one machine, same sentences:

        provider     English   Hindi    Telugu / Tamil / Kannada …
        macOS say     1065ms   ~880ms   silently skipped
        Kokoro         385ms    583ms   DECLINED (an English pipeline fed Telugu
                                        produced 6.5s of confident nonsense)

    So route rather than choose: the fast voice takes what it genuinely speaks, and
    everything else falls through to whatever the deployment does have — macOS `say`
    in development, Indic-Parler on a GPU. A provider signals "not mine" by returning
    None with last_skipped_script set, which is the flag the UI already uses to
    explain silence.
    """

    def __init__(self, primary, secondary):
        self.primary, self.secondary = primary, secondary
        self.last_skipped_script = False

    def _declined(self, provider, audio):
        return audio is None and getattr(provider, "last_skipped_script", False)

    def synth(self, text: str, voice: str | None = None):
        audio = self.primary.synth(text, voice)
        if not self._declined(self.primary, audio):
            self.last_skipped_script = getattr(self.primary, "last_skipped_script", False)
            return audio
        audio = self.secondary.synth(text, voice)
        self.last_skipped_script = getattr(self.secondary, "last_skipped_script", False)
        if audio is None:
            log.info("no configured voice can speak this script")
        return audio

    def synth_stream(self, text: str, voice: str | None = None):
        """Only the primary streams. If it declines there is nothing to stream, and
        the caller falls back to synth() — which is what the capability check in
        server.py already does when a provider has no synth_stream."""
        stream = getattr(self.primary, "synth_stream", None)
        if stream is None:
            raise AttributeError("synth_stream")
        yielded = False
        for frame in stream(text, voice):
            yielded = True
            yield frame
        if not yielded and getattr(self.primary, "last_skipped_script", False):
            # Nothing came out and the primary said it cannot speak this. Render the
            # whole clip through the secondary rather than leaving the caller in
            # silence — a stream that yields nothing is indistinguishable from a
            # broken turn.
            audio = self.secondary.synth(text, voice)
            self.last_skipped_script = getattr(self.secondary, "last_skipped_script", False)
            if audio:
                import numpy as np
                import soundfile as sf
                data, sr = sf.read(io.BytesIO(audio), dtype="float32")
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=1)
                pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                yield pcm, sr

    def __getattr__(self, name):
        return getattr(self.primary, name)


def get_tts(cfg: dict):
    provider = (cfg or {}).get("provider", "system")
    if provider in (None, "none"):
        return None
    try:
        if provider == "piper":
            inner = PiperTTS(cfg)
            serialise = False
        elif provider == "kokoro":
            inner = KokoroTTS(cfg)
            serialise = False
            # Kokoro speaks English and Hindi here and declines every other Indian
            # script. Without a fallback those callers get silence; with one they get
            # whatever this deployment can actually manage.
            fb = (cfg or {}).get("fallback", "system")
            if fb and fb not in ("none", provider):
                try:
                    inner = FallbackTTS(inner, get_tts({**(cfg or {}), "provider": fb,
                                                        "cache": False, "fallback": "none"}))
                except Exception as e:  # noqa: BLE001
                    log.warning("TTS fallback '%s' unavailable (%s) — unsupported "
                                "scripts will be silent", fb, e)
        elif provider == "indic_parler":
            inner = IndicParlerTTS(cfg)
            serialise = True   # heavy neural model — serialise inference
        elif provider == "clone":
            # Out of process when a worker is configured, which is the only way this
            # works on the shipping install — the in-process cloner cannot import at
            # all here (coqui-tts needs a newer transformers than parler-tts allows).
            # The in-process path stays for environments where it does load.
            inner = SubprocessCloneTTS(cfg) if (cfg or {}).get("clone_command") \
                else CloneTTS(cfg)
            serialise = True   # heavy model — serialise inference
        else:
            import sys
            inner = SystemTTS(cfg)
            # macOS `say` spawns an independent subprocess per call → safe to run in
            # parallel (real speedup for multi-sentence replies). pyttsx3 (Linux/Win)
            # drives one shared engine and must be serialised.
            serialise = sys.platform != "darwin"
    except Exception as e:  # noqa: BLE001
        log.warning("TTS provider '%s' unavailable (%s); replies will be text-only.", provider, e)
        return None

    if (cfg or {}).get("cache", True):
        return CachedTTS(inner, serialise=serialise, maxsize=int((cfg or {}).get("cache_size", 256)))
    return inner


# --------------------------------------------------------------------------- #
class CachedTTS:
    """Wraps a provider with an LRU cache and optional synth-serialisation."""

    def __init__(self, inner, serialise=False, maxsize=256):
        self.inner = inner
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        # Pre-rendered lines live SEPARATELY and are never evicted. Without this the
        # feature defeats itself: the clinic pack has 318 fixed lines against a default
        # maxsize of 256, so a fifth of the owner's cloned voice would be thrown away
        # the moment it was made, and ordinary traffic would grind away the rest.
        # They are a fixed, known set — bounded by the pack, not by the call — so an
        # LRU is the wrong structure for them.
        self._pinned: dict[str, bytes] = {}
        self._cache_lock = threading.Lock()
        self._synth_lock = threading.Lock() if serialise else None
        self.last_skipped_script = False

    def synth(self, text: str, voice: str | None = None, pin: bool = False):
        key = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()
        with self._cache_lock:
            hit = self._pinned.get(key)
            if hit is not None:
                self.last_skipped_script = False
                return hit
            if key in self._cache:
                self._cache.move_to_end(key)
                # A cached entry is audio that WAS produced, so nothing was skipped.
                # Leaving a stale True here would make the UI claim a voice can't speak
                # a script it had just spoken.
                self.last_skipped_script = False
                return self._cache[key]

        def _do():
            audio = self.inner.synth(text, voice)
            # Forward the provider's "this voice cannot speak that script" flag. Without
            # it mute_reason was ALWAYS None, so a caller whose language the voice cannot
            # pronounce got silence with no explanation anywhere in the UI.
            self.last_skipped_script = getattr(self.inner, "last_skipped_script", False)
            # Never cache a NON-result. A provider that declined the script returns None
            # with the flag set; filing that meant the next request for the same line
            # was served from the cache — which clears the flag on a hit — so the UI
            # stopped being able to explain why the caller heard nothing. Re-asking a
            # provider that will decline again is cheap; being unable to say why is not.
            if not audio:
                return audio
            with self._cache_lock:
                if pin and not self.last_skipped_script:
                    self._pinned[key] = audio
                else:
                    self._cache[key] = audio
                    if len(self._cache) > self.maxsize:
                        self._cache.popitem(last=False)
            return audio

        if self._synth_lock:
            with self._synth_lock:
                with self._cache_lock:            # re-check after acquiring
                    if key in self._cache:
                        return self._cache[key]
                return _do()
        return _do()

    def unpin(self, text: str, voice: str | None = None) -> bool:
        """Drop a pinned line. Used when a pre-rendered clip fails verification —
        the render has already happened by then, and leaving a bad one pinned is worse
        than never having rendered it."""
        key = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()
        with self._cache_lock:
            return self._pinned.pop(key, None) is not None

    def pinned_bytes(self) -> int:
        """How much memory the pre-rendered set is holding. Reported rather than
        capped: it is a fixed set decided by the pack, and silently dropping half of it
        would put the owner's voice on some lines and not others for no visible reason.
        """
        with self._cache_lock:
            return sum(len(v) for v in self._pinned.values())

    def synth_stream(self, text: str, voice: str | None = None):
        """Incremental synthesis, cache-aware.

        Returns None when this text is ALREADY cached — a cache hit is instant, so
        streaming it frame-by-frame would only add work. Otherwise it streams from the
        provider and, on completion, files the assembled clip in the cache so the next
        identical phrase is a hit. Repeated confirmations stay as cheap as before.
        """
        inner_stream = getattr(self.inner, "synth_stream", None)
        if inner_stream is None:
            return None
        key = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()
        with self._cache_lock:
            if key in self._cache:
                return None                       # already instant — take the WAV path

        def _gen():
            parts, rate = [], None
            for pcm, sr in inner_stream(text, voice):
                if not pcm:
                    continue
                rate = rate or sr
                parts.append(pcm)
                yield pcm, sr
            if parts and rate:
                wav = _wav_from_pcm(b"".join(parts), rate)
                with self._cache_lock:
                    self._cache[key] = wav
                    if len(self._cache) > self.maxsize:
                        self._cache.popitem(last=False)

        return _gen()


def _wav_from_pcm(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header (for caching a streamed clip)."""
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_bytes_from_file(path: str) -> bytes:
    """Read any audio file and re-emit as 16-bit PCM WAV (browser-safe)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


SAY_TIMEOUT_S = 20.0     # a wedged speech daemon must never hang a turn


class SystemTTS:
    """Offline OS TTS, normalised through libsndfile to valid 16-bit WAV.

    On macOS we shell out to the `say` command: it renders reliably from any
    thread (pyttsx3's NSSpeechSynthesizer does NOT render off the main thread,
    which breaks it inside the server's threadpool). Elsewhere we use pyttsx3
    (espeak on Linux, SAPI5 on Windows), both of which render off-thread fine.
    """

    def __init__(self, cfg: dict):
        import sys
        self.cfg = cfg or {}
        self._mac = sys.platform == "darwin"
        self._installed = self._list_installed_voices() if self._mac else set()
        if not self._mac:
            import pyttsx3  # noqa: F401  fail fast if missing

    @staticmethod
    def _list_installed_voices() -> set:
        """Which macOS `say` voices are ACTUALLY installed (many Indian voices aren't by
        default). We only pick from these — otherwise `say` falls back to an English voice
        that garbles native-script text (reads it as gibberish/spelling)."""
        import re as _re
        import subprocess
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
        except Exception:  # noqa: BLE001
            return set()
        names = set()
        for line in out.splitlines():
            m = _re.match(r"^(.+?)\s+[a-z]{2}_[A-Z]{2}\b", line)
            if m:
                names.add(m.group(1).split(" (")[0].strip())
        return names

    # Unicode ranges → a language-appropriate macOS voice, so replies are SPOKEN
    # in the caller's language (not just written). Extend as your OS has voices.
    _SCRIPT_VOICE = [
        (0x0900, 0x097F, "voice_hi", "Lekha"),      # Devanagari (Hindi/Marathi)
        (0x0980, 0x09FF, "voice_bn", "Piya"),       # Bengali/Assamese
        (0x0A00, 0x0A7F, "voice_pa", None),         # Gurmukhi / Punjabi (no default macOS voice)
        (0x0A80, 0x0AFF, "voice_gu", None),         # Gujarati (no default macOS voice)
        (0x0B00, 0x0B7F, "voice_or", None),         # Odia (no default macOS voice)
        (0x0B80, 0x0BFF, "voice_ta", "Vani"),       # Tamil
        (0x0C00, 0x0C7F, "voice_te", "Geeta"),      # Telugu
        (0x0C80, 0x0CFF, "voice_kn", "Soumya"),     # Kannada
        (0x0D00, 0x0D7F, "voice_ml", None),         # Malayalam (no default macOS voice)
        (0x0600, 0x06FF, "voice_ur", None),         # Urdu (no default macOS voice)
    ]

    def _voice_for(self, text: str, fallback: str | None):
        """Return (voice, is_indic_without_voice). Picks the voice for the DOMINANT script of
        the reply (not the first stray character) so it matches the orchestrator's language
        decision. Returns a voice only if it's actually installed; else None so the caller
        skips audio instead of garbling native script with an English voice."""
        counts, letters = {}, 0
        for c in text:
            if not c.isalpha():
                continue
            letters += 1
            for lo, hi, key, default in self._SCRIPT_VOICE:
                if lo <= ord(c) <= hi:
                    counts[(key, default)] = counts.get((key, default), 0) + 1
                    break
        if not counts:
            return fallback, False         # Latin/other → default voice
        (key, default), n = max(counts.items(), key=lambda kv: kv[1])
        if n / max(1, letters) < 0.3:      # a stray script char, not the reply's language
            return fallback, False
        v = self.cfg.get(key, default)
        if v and v in self._installed:
            return v, False
        return None, True                  # dominant Indic script but no installed voice → skip

    def synth(self, text: str, voice: str | None = None):
        if self._mac:
            v, indic_no_voice = self._voice_for(text, voice)
            if indic_no_voice:             # don't speak native script with an English voice (garbled)
                log.info("no installed macOS voice for this reply's script — sending text only. "
                         "Use tts.provider: indic_parler for spoken Indian languages.")
                # Flag it so the UI can SAY so. Returning None silently looked exactly
                # like the agent cutting out mid-reply.
                self.last_skipped_script = True
                return None
            self.last_skipped_script = False
            return self._mac_say(text, v)
        # non-mac (pyttsx3/espeak) has no per-script voice map — warn once for Indic scripts
        if any(ord(c) >= 0x0900 for c in text) and not getattr(self, "_warned_pyttsx", False):
            log.warning("Indian-language reply on non-macOS `system` TTS — espeak will mispronounce it. "
                        "Use tts.provider: indic_parler for proper Indian-language speech.")
            self._warned_pyttsx = True
        return self._pyttsx3(text, voice)

    def _mac_say(self, text: str, voice: str | None = None) -> bytes:
        import subprocess
        aiff = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
        aiff.close()
        cmd = ["say", "-o", aiff.name]
        v = voice or self.cfg.get("voice")
        if v:
            cmd += ["-v", str(v)]
        if self.cfg.get("rate"):
            cmd += ["-r", str(int(self.cfg["rate"]))]
        cmd.append(text)
        try:
            # A timeout is not optional here. `say` talks to a system speech daemon that
            # can wedge; with no timeout the call never returns, the caller hears nothing
            # for the rest of the call, and the threadpool worker is consumed
            # PERMANENTLY — enough wedged turns and the whole server stops answering
            # anyone. 20s is far longer than any real line takes to render.
            subprocess.run(cmd, check=True, timeout=SAY_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return _wav_bytes_from_file(aiff.name)
        except subprocess.TimeoutExpired:
            log.error("macOS `say` did not return within %ss — skipping this line "
                      "(the speech daemon may be wedged)", SAY_TIMEOUT_S)
            return None
        finally:
            try:
                os.unlink(aiff.name)
            except OSError:
                pass

    def _pyttsx3(self, text: str, voice: str | None = None) -> bytes:
        import pyttsx3
        eng = pyttsx3.init()
        if self.cfg.get("rate"):
            eng.setProperty("rate", int(self.cfg["rate"]))
        v = voice or self.cfg.get("voice")
        if v:
            eng.setProperty("voice", v)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            eng.save_to_file(text, tmp.name)
            eng.runAndWait()
            return _wav_bytes_from_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            try:
                eng.stop()
            except Exception:  # noqa: BLE001
                pass


class PiperTTS:
    def __init__(self, cfg: dict):
        from piper.voice import PiperVoice
        self.cfg = cfg or {}
        model = self.cfg.get("piper_model_path") or _find_piper_model(
            self.cfg.get("piper_model", "en_US-amy-medium"))
        if not model or not os.path.exists(model):
            raise FileNotFoundError(
                "Piper voice not found. Run  bash scripts/download_piper_voice.sh  "
                "or set tts.piper_model_path in config.yaml.")
        self.voice = PiperVoice.load(model)
        log.info("Piper voice loaded: %s", os.path.basename(model))

    def synth(self, text: str, voice: str | None = None) -> bytes:
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf)   # piper-tts 1.2.x API (one model = one voice)
        return buf.getvalue()


def _find_piper_model(name: str):
    import glob
    for base in ["models", "./models", os.path.expanduser("~/.local/share/piper")]:
        hits = glob.glob(os.path.join(base, f"{name}.onnx"))
        if hits:
            return hits[0]
    # fall back to ANY onnx voice present
    for base in ["models", "./models"]:
        hits = glob.glob(os.path.join(base, "*.onnx"))
        if hits:
            return hits[0]
    return None


def clean_reference(in_path: str, out_path: str) -> float:
    """Denoise, trim silence, and normalise a reference clip so the cloned voice
    doesn't inherit background noise. Returns the cleaned duration in seconds."""
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(in_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32")

    # spectral-gate denoise (removes steady background hiss/hum/fan noise)
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.9)
    except Exception as e:  # noqa: BLE001
        log.warning("denoise skipped: %s", e)

    # trim leading/trailing silence to keep mostly speech
    peak = float(np.max(np.abs(audio))) or 1.0
    thr = peak * 0.03
    idx = np.where(np.abs(audio) > thr)[0]
    if len(idx) > sr // 10:
        pad = int(sr * 0.1)
        audio = audio[max(0, idx[0] - pad): min(len(audio), idx[-1] + pad)]

    # normalise level
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * 0.95

    sf.write(out_path, audio, sr, subtype="PCM_16")
    return len(audio) / sr


class CloneTTS:
    """Zero-shot voice cloning via Coqui XTTS-v2. Clones any voice from a short
    reference clip (~10s). Real-time on GPU; slow (a few seconds/sentence) on CPU.
    License: XTTS is non-commercial (fine for demos/pilots)."""

    _model = None   # class-level: the model is huge — load exactly once
    _gen_lock = threading.Lock()   # the singleton model is shared across sessions — serialise generate()

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.reference = self.cfg.get("reference")        # path to reference wav
        self.language = self.cfg.get("clone_language", "en")
        self._ensure_model()

    @classmethod
    def _ensure_model(cls):
        if cls._model is None:
            import os
            os.environ.setdefault("COQUI_TOS_AGREED", "1")   # accept license, non-interactive
            from TTS.api import TTS
            log.info("Loading XTTS-v2 cloning model (first time downloads ~1.8GB)…")
            cls._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            log.info("XTTS-v2 loaded.")

    def set_reference(self, path: str):
        self.reference = path

    def synth(self, text: str, voice: str | None = None) -> bytes:
        import numpy as np
        import soundfile as sf
        ref = voice if (voice and os.path.exists(str(voice))) else self.reference
        if not ref or not os.path.exists(ref):
            raise FileNotFoundError("No reference voice clip set for cloning.")
        with CloneTTS._gen_lock:            # shared singleton model — serialise inference
            wav = self._model.tts(text=text, speaker_wav=ref, language=self.language)
        arr = np.asarray(wav, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, arr, 24000, format="WAV", subtype="PCM_16")
        return buf.getvalue()


class KokoroTTS:
    """Kokoro-82M. Fast — measured against macOS `say` on the same machine:

        text                                    say      kokoro
        "Hi."                                  1065ms     239ms
        50-char sentence                       1083ms     526ms

    `say` is ~1064ms of FIXED per-call overhead plus 0.4ms/char, so a three-word
    opener costs the same as a paragraph and the caller waits a full second before
    ANY audio starts. Kokoro runs in-process and scales with the text, which is what
    makes shortening the first chunk worth doing at all.

    It speaks only the languages it was trained on, and it does NOT fail loudly on
    the others: fed Telugu, an English pipeline produced 2.1MB and 6.5 SECONDS of
    audio for one short sentence — the caller hears confident nonsense, which is
    considerably worse than silence. So the script is checked before anything is
    synthesised, and an unsupported one is declined so the session can fall back or
    explain itself.
    """

    # The language dominant_script_lang() reports -> the Kokoro lang_code that can
    # speak it. Everything absent from this map is DECLINED. Kokoro covers en/es/fr/
    # hi/it/ja/pt/zh, which among Indian languages is Hindi alone — Telugu, Tamil,
    # Kannada, Malayalam, Bengali, Gujarati, Odia, Punjabi and Urdu all need
    # Indic-Parler (GPU) and must never be handed to an English pipeline.
    _CAN_SPEAK = {"Hindi": "h"}

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.voice = self.cfg.get("kokoro_voice", "af_heart")
        self.lang = self.cfg.get("kokoro_lang", "a")
        self._pipes = {}
        self.last_skipped_script = False
        self._pipe(self.lang)                    # fail fast if kokoro is unusable

    def _pipe(self, lang_code: str):
        """One pipeline per language, built on first use and kept."""
        if lang_code not in self._pipes:
            from kokoro import KPipeline
            self._pipes[lang_code] = KPipeline(lang_code=lang_code)
        return self._pipes[lang_code]

    # Hindi needs a Hindi voice as well as a Hindi pipeline; an English voice id on
    # the 'h' pipeline is a 404 and a silent reply.
    _LANG_VOICE = {"h": "hf_alpha"}

    def _route(self, text: str):
        """(lang_code, voice) — or None when Kokoro must not attempt this text."""
        from .orchestrator import dominant_script_lang
        script = dominant_script_lang(text or "")
        if script in (None, "Latin"):
            return self.lang, None               # ASCII/Latin → the configured language
        want = self._CAN_SPEAK.get(script)
        if want is None:
            return None                          # a script Kokoro cannot speak
        return want, self._LANG_VOICE.get(want)

    SR = 24000
    # Kokoro voice ids look like af_heart / am_michael / bf_emma: <lang><gender>_<name>.
    # Every pack sets `voice:` to an OS voice ("Tara"), which is handed to whichever
    # provider is configured — Kokoro then tries to download a voice by that name and
    # gets a 404, so the reply is silent. A provider must ignore an id that isn't its own.
    _VOICE_RE = re.compile(r"^[abefhijpz][fm]_\w+$")

    def _voice(self, voice: str | None) -> str:
        if voice and self._VOICE_RE.match(voice):
            return voice
        if voice:
            log.debug("kokoro: ignoring non-Kokoro voice %r, using %s", voice, self.voice)
        return self.voice

    def synth(self, text: str, voice: str | None = None) -> bytes:
        import numpy as np
        import soundfile as sf
        route = self._route(text)
        if route is None:
            # Say nothing rather than something wrong. `last_skipped_script` is what
            # lets the session set mute_reason and the UI explain the silence.
            self.last_skipped_script = True
            log.info("kokoro: cannot speak this script — declining (%r)", (text or "")[:30])
            return None
        self.last_skipped_script = False
        lang, lang_voice = route
        pipe = self._pipe(lang)
        chunks = [audio for _gs, _ps, audio in
                  pipe(text, voice=lang_voice or self._voice(voice))]
        wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, wav, self.SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # Kokoro's default split is on NEWLINES, so a whole sentence renders as one segment
    # and there is nothing to stream — measured 1 frame, no gain over the whole-clip
    # path. Splitting on clause punctuation instead gets the first words out roughly
    # twice as fast on a long sentence (1344ms → 738ms, 3 frames), at the cost of
    # generating each clause independently. Streaming only; `synth` keeps one-shot
    # prosody for anything cached or spoken as a whole.
    STREAM_SPLIT = r"[,;:]\s*"

    def synth_stream(self, text: str, voice: str | None = None):
        """Yield each segment the moment Kokoro finishes it.

        The pipeline already hands back audio segment-by-segment — `synth` above just
        concatenates them and throws the incremental delivery away. Here we forward
        each one instead, so the caller starts hearing the sentence while its tail is
        still being generated.
        """
        import numpy as np
        route = self._route(text)
        if route is None:
            self.last_skipped_script = True
            log.info("kokoro: cannot speak this script — declining (%r)", (text or "")[:30])
            return
        self.last_skipped_script = False
        lang, lang_voice = route
        for _gs, _ps, audio in self._pipe(lang)(text, voice=lang_voice or self._voice(voice),
                                                split_pattern=self.STREAM_SPLIT):
            arr = np.asarray(audio, dtype="float32").reshape(-1)
            if not arr.size:
                continue
            pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            yield pcm, self.SR


class IndicParlerTTS:
    """AI4Bharat Indic-Parler-TTS — open-source, ~21 Indian languages + English.

    Free & offline: weights download once from Hugging Face (`ai4bharat/indic-
    parler-tts`, ~2GB). NO api key. The spoken *language* is auto-detected from the
    text, so the same voice handles Hindi, Tamil, Telugu, Bengali, Marathi, etc.
    The *voice/style* (gender, pace, clarity) is set by a natural-language
    `description` string — override it per pack via `tts.indic_description`.

    Real-time on GPU; slow (a few seconds/sentence) on CPU — cache + streaming hide
    most of that for repeated greetings/confirmations. License: Apache-2.0
    (commercial-safe, unlike XTTS clone).
    """

    _model = None          # class-level: model is large — load exactly once
    _tok = None
    _desc_tok = None
    _sr = 44100
    _device = None
    _gen_lock = threading.Lock()   # shared singleton model — serialise generate() across sessions

    _DEFAULT_DESC = ("A female speaker with a clear, warm, natural voice speaks at a "
                     "moderate pace with expressive intonation. The recording is very "
                     "high quality with no background noise.")

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.description = self.cfg.get("indic_description", self._DEFAULT_DESC)
        self._ensure_model(self.cfg.get("device"))

    @classmethod
    def _ensure_model(cls, device_pref=None):
        if cls._model is not None:
            return
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        # Default cuda -> cpu. MPS (Apple) is opt-in only: Parler's DAC decoder has
        # unsupported ops on MPS, so we don't auto-select it — set tts.device: mps
        # in config.yaml to try it.
        if device_pref:
            cls._device = device_pref
        elif torch.cuda.is_available():
            cls._device = "cuda"
        else:
            cls._device = "cpu"
        model_id = "ai4bharat/indic-parler-tts"
        log.info("Loading Indic-Parler-TTS (first run downloads ~2GB)… device=%s", cls._device)
        # Build EVERYTHING into locals first and publish cls._model LAST. Assigning the
        # model before the tokenizers meant a failure on any later line (a network blip
        # fetching the second tokenizer, an OOM) left cls._model set with cls._tok still
        # None — and the `if cls._model is not None: return` guard above then made every
        # subsequent load a no-op. The singleton was permanently poisoned and every reply
        # for the rest of the process was silent, with the real error long gone.
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(cls._device)
        tok = AutoTokenizer.from_pretrained(model_id)
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        sr = int(model.config.sampling_rate)
        cls._tok, cls._desc_tok, cls._sr = tok, desc_tok, sr
        cls._model = model                       # published last — see above
        log.info("Indic-Parler-TTS loaded (sr=%d).", cls._sr)

    def synth(self, text: str, voice: str | None = None) -> bytes:
        import numpy as np
        import soundfile as sf
        # a `voice` override string is treated as a style description
        description = voice if (voice and not os.path.exists(str(voice))) else self.description
        with IndicParlerTTS._gen_lock:      # shared singleton model — serialise inference
            desc = self._desc_tok(description, return_tensors="pt").to(self._device)
            prompt = self._tok(text, return_tensors="pt").to(self._device)
            generation = self._model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask)
        arr = generation.cpu().numpy().squeeze().astype("float32")
        if arr.ndim == 0 or arr.size == 0:
            arr = np.zeros(1, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, arr, self._sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()


class SubprocessCloneTTS:
    """Voice cloning in a SEPARATE process, with its own dependencies.

    WHY THIS EXISTS, and it is not a style preference.

    The in-process cloner does not work. `get_tts({"provider": "clone"})` returns None
    on this install, because coqui-tts needs a newer `transformers` than the 4.46.1 this
    venv is pinned to — and that pin is deliberate: parler-tts, the Indic voice, requires
    exactly 4.46.1. So the environment can host the Indic voice or the cloner, and it
    chose the voice, correctly. The brand-voice feature has been silently unavailable
    ever since.

    That is the third time a pip install has broken this venv (deepfilternet forced
    numpy<2 and broke SpeechBrain; Qwen3-Embedding's transformers bump broke parler),
    and the codebase already has the answer both other times: DO NOT IMPORT IT. The
    DeepFilterNet path shells out to a Rust binary; whisper.cpp shells out to
    `whisper-server`. A subprocess cannot break this venv, and its dependency conflicts
    are its own business.

    Cloning is the ideal candidate for it. It is measured in SECONDS per sentence
    (VoxCPM-0.5B: RTF 2.89 English on this M1; XTTS documents "a few seconds/sentence on
    CPU"), so it can never be on a turn's critical path anyway — it runs offline, in
    `prerender`, filling a cache. Against a 5-second synthesis, process spawn is noise.

    The worker contract is deliberately tiny, so any cloner can satisfy it:

        <python> <script> --ref <reference.wav> --text <text> --out <out.wav>

    exit 0 and a readable wav at --out means success. Anything else is a decline, and
    the caller falls back exactly as it does for any provider that cannot speak.
    """

    def __init__(self, cfg: dict):
        import os
        import shutil
        cfg = cfg or {}
        self.reference = cfg.get("reference")
        cmd = cfg.get("clone_command")
        if not cmd:
            raise RuntimeError(
                "tts.clone_command is not set. Point it at a worker in its own venv, "
                "e.g. ['/path/to/vox-venv/bin/python', 'scripts/clone_worker.py'] — "
                "see scripts/clone_worker.py for the contract.")
        self.cmd = list(cmd) if isinstance(cmd, (list, tuple)) else str(cmd).split()
        exe = self.cmd[0]
        if not (os.path.isfile(exe) or shutil.which(exe)):
            raise RuntimeError("clone worker not found: %s" % exe)
        if not self.reference or not os.path.isfile(self.reference):
            raise RuntimeError("clone reference wav missing: %s" % self.reference)
        self.timeout = float(cfg.get("clone_timeout", 120))
        self.language = cfg.get("clone_language", "en")
        self.last_skipped_script = False
        # One at a time. The worker loads a multi-gigabyte model; two of them at once on
        # a laptop is how you get an OOM instead of a voice.
        self._lock = threading.Lock()
        log.info("TTS: subprocess cloner (%s)", os.path.basename(exe))

    def synth(self, text: str, voice: str | None = None):
        import os
        import subprocess
        import tempfile
        self.last_skipped_script = False
        if not (text or "").strip():
            return None
        tmp = tempfile.mkdtemp(prefix="zs_clone_")
        out = os.path.join(tmp, "o.wav")
        try:
            cmd = self.cmd + ["--ref", self.reference, "--text", text, "--out", out,
                              "--language", self.language]
            with self._lock:
                r = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
            if r.returncode != 0:
                log.warning("clone worker failed (%s): %s", r.returncode,
                            (r.stderr or b"")[-200:])
                return None
            if not os.path.isfile(out) or os.path.getsize(out) < 64:
                return None
            data = open(out, "rb").read()
            # A worker that printed an error into --out must not be forwarded AS audio:
            # the client then fails to decode it and the turn is silent with nothing
            # recorded anywhere. `orchestrator._playable` already guards every clone
            # result, so this is the cheap structural half — importing that function
            # here would be a circular import, and copying it would be a second copy
            # to drift.
            if data[:4] not in (b"RIFF", b"FORM") and data[:3] != b"ID3":
                log.warning("clone worker wrote something that is not audio")
                return None
            return data
        except subprocess.TimeoutExpired:
            # A cloner that wedges must not hold a threadpool worker forever — the same
            # rule the `say` and whisper.cpp timeouts exist for.
            log.warning("clone worker timed out after %.0fs", self.timeout)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("clone worker error: %s", e)
            return None
        finally:
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
