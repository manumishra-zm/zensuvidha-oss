"""Speech-to-text via faster-whisper (CPU / int8). Accepts any audio file
(webm/opus, wav, mp3…) — PyAV bundled with faster-whisper decodes it, so no
system ffmpeg is required.

Silence hardening: VAD strips non-speech before transcription, and a degenerate-
output guard drops Whisper's classic silence hallucinations ("5, 5, 5, 5 …").
"""
import io
import logging
import re
import threading
import unicodedata

log = logging.getLogger("zensuvidha.stt")


def _looks_degenerate(text: str) -> bool:
    """True for repetitive Whisper hallucinations.

    Two shapes, because Whisper degenerates in two different ways:

      * word level — "5 5 5 5 5 5 5": many tokens, almost no distinct ones;
      * CHARACTER level — "तो अचाएज़़़़़़़़़़…", where it latches onto a single
        combining mark and emits it hundreds of times. That has no spaces at all, so
        it is one giant token and the word-level check never sees it. Real observed
        output: 129 characters, 9 distinct.
    """
    toks = re.findall(r"\w+", text.lower())
    if len(toks) >= 6 and len(set(toks)) <= max(2, len(toks) // 5):
        return True
    # A run of one character USED to condemn the whole turn at 6 repeats, anywhere in
    # the text. That is a real caller loss: "my number is 8888884321" has a six-run, and
    # so does a drawn-out "haaaaaan" or "हाँऽऽऽऽऽऽ". Whisper's actual character-level
    # degeneracy is not subtle — it emits the mark hundreds of times and little else — so
    # require the run to DOMINATE, not merely to occur.
    packed = re.sub(r"\s", "", text)
    longest = max((len(m.group(0)) for m in re.finditer(r"(.)\1+", packed)), default=0)
    if longest >= 20 or (longest >= 8 and packed and longest >= len(packed) * 0.25):
        return True
    if len(packed) >= 20 and len(set(packed)) <= max(3, len(packed) // 10):
        return True
    return False


def _trim_degenerate(text: str) -> str:
    """Collapse runaway character runs, keeping whatever the caller actually said.

    Whisper often produces a real transcript and THEN latches onto a combining mark:
    "मेरा नंबर है 8920429057ऽऽऽऽऽ…". Dropping the turn threw the phone number away with
    the artifact. Collapsing the run to a plausible length keeps the words.
    """
    return re.sub(r"(.)\1{7,}", lambda m: m.group(1) * 3, text or "")


# Phrases Whisper famously invents from silence / music / room-noise (it was trained on
# YouTube). These are NOT real caller utterances, so we drop them to stop the agent from
# "randomly speaking" when nobody actually said anything.
# Split by whether a real caller could ever say it. The distinction only started to
# matter with a second backend: whisper.cpp enforces its confidence thresholds inside
# the binary and does not report the resulting numbers, so a check written as "artifact
# AND no_speech > 0.5" silently becomes no check at all there. These never come out of
# a person's mouth on a phone call, so they need no corroboration.
_NEVER_SAID = {
    "thank you for watching", "thanks for watching", "thank you for watching!",
    "please subscribe", "like and subscribe", "please subscribe to my channel",
    "thanks for watching!", "see you in the next video",
    "www.mooji.org", "amara.org", "subtitles by the amara.org community",
    "i'll see you next time", "शुक्रिया देखने के लिए",
}
# …and these are Whisper artifacts that are ALSO things people say. Dropping one of
# these on the strength of the phrase alone would hang up on a caller saying goodbye.
_MAYBE_SAID = {
    "subscribe", "see you next time", "bye bye", "thank you.", "you", ".", "..", "...",
    "मुझे लगता है", " धन्यवाद",
}

_HALLUCINATION_PHRASES = _NEVER_SAID | _MAYBE_SAID     # kept: tests and callers use it


def _is_hallucination(text: str, *, only_certain: bool = False) -> bool:
    """True when the WHOLE transcript is a known silence/noise artifact.

    `only_certain` restricts it to phrases a caller would never utter, for the case
    where there is no confidence number to corroborate a judgement call.
    """
    t = text.strip().lower().strip(" .!?।-…")
    # Nothing left once punctuation is stripped. Nobody "says" "." or "…" or "।", so
    # this is certain whether or not a confidence number corroborates it — and it must
    # be, because `.` and `...` are in _MAYBE_SAID and would otherwise sail through the
    # only_certain branch on every whisper.cpp turn. A truthy "." becomes a real caller
    # turn and the agent answers silence, which is the exact failure this guard exists
    # to prevent.
    if not t:
        return True
    if t in _NEVER_SAID:
        return True
    # …and the same reasoning for a single stray character: it is noise in any language,
    # not a short answer.
    #
    # NOT `re.sub(r"\W", ...)`. This codebase has already been bitten by that once:
    # \w EXCLUDES Unicode combining marks, so "हाँ" reduces to "ह" — length 1 — and a
    # caller's "yes" in Devanagari is thrown away as noise. It only stayed hidden here
    # while this check sat behind a confidence gate that rarely fired; promoting it
    # above `only_certain` made it fire on every whisper.cpp turn. Counting Letters,
    # Numbers and Marks is what actually means "characters a person typed or said".
    if sum(1 for c in t if unicodedata.category(c)[0] in "LNM") <= 1:
        return True
    if only_certain:
        return False
    if t in _MAYBE_SAID:
        return True
    return False


def judge(text: str, *, no_speech: float | None, avg_logprob: float | None,
          reject_no_speech: float, reject_logprob: float) -> str:
    """Apply the hardening every backend must pass. Returns the text, or "" to drop it.

    Shared deliberately. Each of these guards exists because of a specific way a real
    call failed — a phantom transcription of silence answered nobody, a latched
    combining mark ate a phone number, "thank you for watching" was spoken back at a
    caller who said nothing. A second recogniser that skipped them would reintroduce
    all of it, and would do so quietly, because the failure looks like a bad model
    rather than a missing guard.

    `no_speech`/`avg_logprob` of None mean the backend enforced those thresholds
    itself and did not report the numbers — NOT that it is confident. Passing 0.0 for
    "unknown" would read as certainty and turn the artifact check off without saying so.
    """
    if not text:
        return ""
    known = no_speech is not None and avg_logprob is not None
    if known and no_speech > reject_no_speech:  # duration-weighted: mostly non-speech
        log.info("STT dropped (no_speech=%.2f): %r", no_speech, text[:40])
        return ""
    if known and avg_logprob < reject_logprob:  # duration-weighted low confidence
        log.info("STT dropped (avg_logprob=%.2f): %r", avg_logprob, text[:40])
        return ""
    if _looks_degenerate(text):
        # Salvage first. If trimming the runaway run leaves a real sentence, that
        # sentence is what the caller said and throwing it away loses their turn.
        trimmed = _trim_degenerate(text)
        if trimmed != text and not _looks_degenerate(trimmed) and len(trimmed.split()) >= 2:
            log.info("STT repaired (degenerate run trimmed): %r → %r",
                     text[:40], trimmed[:40])
            text = trimmed
        else:
            log.info("STT dropped (degenerate): %r", text[:40])
            return ""
    # a known Whisper artifact ("thank you for watching") — but "thank you"/"bye" are
    # also REAL utterances, so the ambiguous ones need the audio to look like non-speech
    # too. Without that evidence only the phrases nobody ever says are dropped.
    if _is_hallucination(text, only_certain=not known) and (not known or no_speech > 0.5):
        log.info("STT dropped (hallucination, no_speech=%s): %r",
                 "n/a" if not known else "%.2f" % no_speech, text[:40])
        return ""
    return text


# whisper.cpp's own name for a language, which it reports as a WORD ("english") where
# faster-whisper reports a code ("en"). Everything downstream — the language lock, the
# pack's a_hi/a_te answers, the reply-language guard — is keyed on codes, so handing it
# a name means the lock silently never matches anything.
# Kept here rather than imported from orchestrator: stt must not depend on the engine.
_WHISPER_LANG_CODES = {
    "english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta", "bengali": "bn",
    "marathi": "mr", "gujarati": "gu", "kannada": "kn", "malayalam": "ml",
    "punjabi": "pa", "urdu": "ur", "oriya": "or", "odia": "or", "assamese": "as",
    "nepali": "ne", "sanskrit": "sa", "sindhi": "sd",
}


def _lang_code(name):
    """'english' → 'en'. A two-letter value is already a code and passes through."""
    if not name:
        return None
    n = str(name).strip().lower()
    if len(n) <= 3 and n.isalpha():
        return n
    return _WHISPER_LANG_CODES.get(n)


def best_local_device(prefer: str | None = None) -> str:
    """What this machine should actually run on. 'auto' resolves here, once.

    The point is that a clone runs well on whatever box it lands on without anybody
    editing config: a CUDA card if there is one, Apple's GPU on a Mac, otherwise the
    CPU. `prefer` (anything but None/"auto") is returned untouched — somebody who
    pinned a device meant it.
    """
    if prefer and prefer != "auto":
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # NOT mps. Measured on this codebase: SpeechBrain's ECAPA fails outright on it
        # ("slow_conv2d_forward_mps: input(device='cpu')") because it keeps internal CPU
        # tensors, and Whisper via CTranslate2 does not use it at all. Apple's GPU is
        # reached here through whisper.cpp's Metal backend instead, which is a separate
        # process and needs no torch device.
    except Exception:  # noqa: BLE001  (torch is optional for the core install)
        pass
    return "cpu"


def best_thread_count(cfg: dict | None = None) -> int:
    """Threads for whisper.cpp. Deliberately not `all of them`.

    MEASURED on this M1 Pro with CTranslate2, and the shape is the same reason here:
    auto 1691ms, 4 threads 1601ms, 8 threads 2212ms, 10 threads 3798ms. Piling work onto
    the efficiency cores makes it WORSE. Performance-core count is the honest ceiling,
    and 4 is a safe floor on anything smaller.
    """
    explicit = (cfg or {}).get("whispercpp_threads")
    if explicit:
        return int(explicit)
    n = 4
    try:
        import subprocess as _sp
        out = _sp.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                      capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            n = int(out.stdout.strip())
    except Exception:  # noqa: BLE001  (not a Mac, or sysctl missing)
        try:
            import os as _os
            n = max(2, min(8, (_os.cpu_count() or 4) // 2))
        except Exception:  # noqa: BLE001
            n = 4
    return max(2, min(8, n))


def get_stt(cfg: dict):
    commit = _build_stt(cfg)
    if commit is None:
        return None
    # Two-pass: a tiny model answers the speculative frames, the accurate one answers
    # the turn. Off unless a partial model is configured AND actually present, because
    # a missing file must degrade to the single-model path rather than to no speech.
    pm = (cfg or {}).get("partial_model")
    if pm:
        sub = dict(cfg)
        sub["whispercpp_model"] = pm
        sub["provider"] = cfg.get("partial_provider") or "auto"
        sub["partial_model"] = None            # …or it would recurse
        try:
            part = _build_stt(sub)
            if part is not None and part is not commit:
                return TwoPassSTT(commit, part)
        except Exception as e:  # noqa: BLE001
            log.info("two-pass STT unavailable (%s) — one model for both passes", e)
    return commit


def _build_stt(cfg: dict):
    provider = (cfg or {}).get("provider", "faster_whisper")
    if provider in (None, "none"):
        return None
    # "auto" takes the faster recogniser when this machine actually has it. Measured
    # 1.73-1.87x on the dominant cost of the audio path, at slightly BETTER word error
    # rate — so on a box where it is installed there is no argument for the slower one,
    # and on a box where it is not, nothing changes. The constructor checks for the
    # binary and the model, so "installed" is a fact rather than a hope.
    # Best-first, each step measured against the one below it:
    #   whisper.cpp SERVER   model resident        621ms @3.5s   2.80x
    #   whisper.cpp CLI      model loaded per turn 841ms @3.5s   2.07x
    #   faster-whisper       CPU int8             1738ms @3.5s   baseline
    # The server is also the only one of the two whisper.cpp modes that reports
    # confidence numbers, so it is the more accurate as well as the faster.
    if provider == "auto":
        for build, why in ((WhisperCppServerSTT, "resident model, 2.8x"),
                           (WhisperCppSTT, "per-turn load, 2.1x")):
            try:
                stt = build(cfg)
                log.info("STT: auto → %s (%s)", build.__name__, why)
                return stt
            except Exception as e:  # noqa: BLE001
                log.info("STT: auto skipped %s (%s)", build.__name__, e)
        log.info("STT: auto → faster-whisper (no whisper.cpp on this machine)")
        provider = "faster_whisper"
    if provider in ("whisper_cpp_server", "whispercpp_server"):
        try:
            return WhisperCppServerSTT(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("whisper-server unavailable (%s) — falling back. "
                        "Run: bash scripts/download_whispercpp.sh", e)
    if provider in ("whisper_cpp", "whispercpp", "whisper.cpp"):
        try:
            return WhisperCppSTT(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("whisper.cpp unavailable (%s) — falling back to faster-whisper. "
                        "Run: bash scripts/download_whispercpp.sh", e)
    try:
        return FasterWhisperSTT(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("faster-whisper unavailable (%s); voice input disabled, text still works.", e)
        return None


# No caller says one sentence for five minutes. Anything longer is a decompression
# bomb or a latched VAD, and either way Whisper must not be handed it.
MAX_DECODE_S = 300


class FasterWhisperSTT:
    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel
        self.language = cfg.get("language")
        self.beam_size = int(cfg.get("beam_size", 1))
        self.fast_beam_size = int(cfg.get("fast_beam_size", 1))
        self.vad_filter = bool(cfg.get("vad_filter", False))
        self.denoise = bool(cfg.get("denoise", False))
        # Optional DeepFilterNet, switchable per turn from the UI. None = unavailable,
        # in which case the toggle simply has no effect. See zensuvidha/denoise.py for
        # the measurements that keep this off by default.
        self.denoiser = None
        if (cfg.get("denoise_backend") or "").lower() in ("deepfilternet", "deepfilter", "df"):
            from .denoise import get_denoiser
            self.denoiser = get_denoiser(cfg)
        self.reject_no_speech = float(cfg.get("reject_no_speech", 0.85))   # tuned to NOT drop real speech
        self.reject_logprob = float(cfg.get("reject_logprob", -1.6))
        # "auto" resolves here, once, rather than being handed to CTranslate2 — which
        # does not know the word and raises. On a CUDA box int8 would also be leaving
        # most of the card unused, so the compute type follows the device unless
        # somebody pinned one.
        device = best_local_device(cfg.get("device"))
        compute = cfg.get("compute_type")
        if not compute or compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(cfg.get("model", "tiny"), device=device,
                                  compute_type=compute)
        log.info("STT: faster-whisper %s on %s (%s)",
                 cfg.get("model", "tiny"), device, compute)
        # WhisperModel.transcribe is NOT safe to call concurrently — serialise it so
        # two simultaneous callers can't crash / corrupt each other's transcription.
        self._lock = threading.Lock()

    def _decode(self, audio):
        """Any input (path / bytes / file-like / ndarray) → 16kHz mono float32.
        Decode ONLY — no filtering, so a denoiser can be chosen separately."""
        import numpy as np
        import soundfile as sf
        if isinstance(audio, np.ndarray):
            return audio.astype("float32")
        src = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
        # Bound the DECODED length, not the transferred bytes. A 180KB FLAC or Opus frame
        # is well under any wire-size cap and still expands to hundreds of megabytes of
        # float32 — enough to stall STT for every concurrent call on the box while the
        # process thrashes. sf.read's `frames` argument stops the decoder rather than
        # trimming afterwards, so the memory is never allocated in the first place.
        try:
            info = sf.info(src)
            if isinstance(src, io.BytesIO):
                src.seek(0)
            limit = int(MAX_DECODE_S * (info.samplerate or 16000))
            if info.frames and info.frames > limit:
                log.warning("audio decodes to %.0fs — reading only the first %ss",
                            info.frames / max(1, info.samplerate), MAX_DECODE_S)
                data, sr = sf.read(src, frames=limit)
            else:
                data, sr = sf.read(src)
        except Exception:  # noqa: BLE001  (sf.info can't probe every stream — fall through)
            if isinstance(src, io.BytesIO):
                src.seek(0)
            data, sr = sf.read(src)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        data = data.astype("float32")
        if sr != 16000:
            n = int(len(data) * 16000 / sr)
            data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                             np.arange(len(data)), data).astype("float32")
        return data

    def _to_array(self, audio):
        """Decode, then reduce noise with noisereduce (spectral gating).

        Kept for `stt.denoise: true`. Measured to change word-error rate by 0.00, so
        it stays off by default — see the note in config.yaml."""
        data = self._decode(audio)
        try:
            import noisereduce as nr
            data = nr.reduce_noise(y=data, sr=16000, stationary=False,
                                   prop_decrease=0.8).astype("float32")
        except Exception as e:  # noqa: BLE001
            log.warning("denoise skipped: %s", e)
        return data

    _CFG = "__cfg__"

    def transcribe(self, audio, hint: str | None = None, language=_CFG, fast: bool | None = None,
                   denoise: bool | None = None):
        """Return (text, detected_lang_code, language_probability).

        `audio` may be a path, raw WAV bytes, a file-like, or an ndarray. When denoise
        is off we feed bytes straight to Whisper (no disk round-trip). Returning the
        detected language lets the caller latch it reliably (better than script guessing).
        """
        beam = self.fast_beam_size if fast else self.beam_size
        # `denoise` may be forced on/off per turn by the UI toggle; None = follow config.
        use_df = self.denoiser is not None and (
            self.denoise if denoise is None else bool(denoise))
        use_spectral = (self.denoise if denoise is None else bool(denoise)) and not use_df
        if use_df:
            audio_input = self.denoiser(self._decode(audio), 16000)
        elif use_spectral and not fast:
            audio_input = self._to_array(audio)
        elif isinstance(audio, (bytes, bytearray)):
            audio_input = io.BytesIO(audio)     # feed bytes directly — no temp file / disk I/O
        else:
            audio_input = audio                 # path / file-like / ndarray
        # language: "__cfg__" → config default; None → auto-detect; "en"/"hi"/… → pin it.
        lang = self.language if language == self._CFG else language
        # The vocabulary hint is ENGLISH; it biases Whisper toward English, so only use
        # it when the (resolved) language is English.
        initial_prompt = (hint or None) if lang == "en" else None
        with self._lock:                        # WhisperModel isn't thread-safe
            segments, info = self.model.transcribe(
                audio_input,
                task="transcribe",              # transcribe in the SPOKEN language, never translate
                language=lang,                  # None = auto-detect; pinned = reliable
                initial_prompt=initial_prompt,  # domain vocabulary boost (pinned-language only)
                beam_size=beam,                 # 1 = greedy = fastest
                vad_filter=self.vad_filter,     # strip silence → kills most hallucinations
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.2,
                temperature=0.0,
            )
            segs = list(segments)               # generator → run inference in-lock
        text = "".join(s.text for s in segs).strip()
        det_lang = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        if not text:
            return "", None, 0.0
        # ---- confidence gate: reject phantom transcriptions of silence/noise so the agent
        #      never "randomly speaks". Aggregate over segments by DURATION (not the single
        #      worst segment) so one brief pause can't drop a long, real sentence. ----
        tot = sum(max(0.05, s.end - s.start) for s in segs) or 1.0
        no_speech = sum(s.no_speech_prob * max(0.05, s.end - s.start) for s in segs) / tot
        avg_lp = sum(s.avg_logprob * max(0.05, s.end - s.start) for s in segs) / tot
        text = judge(text, no_speech=no_speech, avg_logprob=avg_lp,
                     reject_no_speech=self.reject_no_speech,
                     reject_logprob=self.reject_logprob)
        if not text:
            return "", None, 0.0
        return text, det_lang, prob


class WhisperCppSTT:
    """Whisper via the whisper.cpp binary — the same model, on the GPU.

    Why a second recogniser at all. STT is the dominant cost on the audio path
    (measured 1070ms at 3s of speech against 66ms for isolation and 34ms for the
    speaker gate) and it is CPU-bound with no knob left: more CTranslate2 threads
    measured WORSE on this M1 (auto 1691ms, 8 threads 2212ms, 10 threads 3798ms),
    and greedy decoding was not reliably faster. The one resource not being used is
    the GPU that is already in the machine, and whisper.cpp is the way to reach it
    without a CUDA box.

    Shelled out to, not imported, on the deep-filter precedent: a pip install into
    this venv has silently broken it twice (deepfilternet forced numpy<2 and broke
    SpeechBrain; Qwen3-Embedding's transformers bump broke parler-tts, the Indic
    voice). A binary cannot do that.

    Same interface as FasterWhisperSTT, and it goes through the same `judge()` — a
    backend that skipped those guards would answer silence again.
    """

    def __init__(self, cfg: dict):
        import os
        import shutil
        self.bin = cfg.get("whispercpp_bin") or shutil.which("whisper-cli") or "whisper-cli"
        if not (os.path.isfile(self.bin) or shutil.which(self.bin)):
            raise RuntimeError("whisper-cli not found (brew install whisper-cpp)")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = cfg.get("whispercpp_model") or "models/whispercpp/ggml-small.bin"
        self.model_path = model if os.path.isabs(model) else os.path.join(root, model)
        if not os.path.isfile(self.model_path):
            raise RuntimeError("ggml model missing: %s" % self.model_path)
        self.language = cfg.get("language")
        self.beam_size = int(cfg.get("beam_size", 3))
        self.fast_beam_size = int(cfg.get("fast_beam_size", 1))
        self.reject_no_speech = float(cfg.get("reject_no_speech", 0.85))
        self.reject_logprob = float(cfg.get("reject_logprob", -1.6))
        self.timeout = float(cfg.get("whispercpp_timeout", 60))
        self.threads = int(cfg.get("whispercpp_threads", 4))
        # whisper.cpp wants Silero in ITS OWN ggml format, not the ONNX this project
        # already has three copies of. Handing it `silero_vad_v6.onnx` does not fail
        # with a message — the process ABORTS (SIGABRT, a dyld exit trace), which from
        # Python looks like a broken recogniser rather than a wrong file. Verified by
        # running the binary by hand: identical command, ONNX swapped for the .bin.
        self.vad_model = None
        if cfg.get("vad_filter", False):
            for cand in self._vad_candidates(root, cfg):
                if os.path.isfile(cand):
                    self.vad_model = cand
                    break
            if self.vad_model is None:
                log.info("whisper.cpp: vad_filter is on but no ggml Silero model was "
                         "found — silence will reach the decoder. Run: "
                         "bash scripts/download_whispercpp.sh")
        self.denoise = bool(cfg.get("denoise", False))
        self.denoiser = None
        if (cfg.get("denoise_backend") or "").lower() in ("deepfilternet", "deepfilter", "df"):
            from .denoise import get_denoiser
            self.denoiser = get_denoiser(cfg)
        # One process at a time. The binary would happily run several, but they would
        # contend for the same GPU and the box already caps concurrent calls at 6.
        self._lock = threading.Lock()
        self._decode = FasterWhisperSTT._decode.__get__(self)
        log.info("STT: whisper.cpp (%s)", os.path.basename(self.model_path))

    @staticmethod
    def _vad_candidates(root, cfg):
        """Only ggml VAD models. An .onnx here aborts the binary — see __init__."""
        import glob
        import os
        explicit = cfg.get("whispercpp_vad_model")
        if explicit:
            yield explicit if os.path.isabs(explicit) else os.path.join(root, explicit)
        yield from sorted(glob.glob(os.path.join(root, "models", "whispercpp",
                                                 "ggml-silero*.bin")), reverse=True)

    _CFG = "__cfg__"

    def transcribe(self, audio, hint: str | None = None, language=_CFG,
                   fast: bool | None = None, denoise: bool | None = None):
        import json as _json
        import os
        import subprocess
        import tempfile

        import numpy as np
        import soundfile as sf

        data = self._decode(audio)
        if denoise is None:
            denoise = self.denoise
        if denoise and self.denoiser is not None:
            data = self.denoiser(data, 16000)
        data = np.asarray(data, dtype="float32").reshape(-1)
        if not data.size:
            return "", None, 0.0

        lang = self.language if language == self._CFG else language
        tmp = tempfile.mkdtemp(prefix="zs_wcpp_")
        # Serialise, as the comment on _lock promises. Six concurrent turns spawning six
        # whisper-cli processes contend for the same GPU, and the 1.8x that justifies
        # this backend disappears exactly when the box is busiest.
        wav = os.path.join(tmp, "a.wav")
        try:
            sf.write(wav, data, 16000, subtype="PCM_16")
            cmd = [self.bin, "-m", self.model_path, "-f", wav,
                   "-t", str(self.threads),
                   "-bs", str(self.fast_beam_size if fast else self.beam_size),
                   "-oj", "-of", os.path.join(tmp, "out"),
                   # The confidence half of the gate, enforced INSIDE the binary
                   # because it is the only place the numbers exist. Same values the
                   # Python path uses, so the two backends reject the same audio.
                   "-nth", str(self.reject_no_speech),
                   "-lpt", str(self.reject_logprob),
                   "--no-prints", "-np"]
            if self.vad_model:
                # `vad_filter: true` in config, honoured. Strips silence before the
                # decoder, which is what kills most hallucinations — the same job
                # faster-whisper's bundled copy of this exact model does.
                cmd += ["--vad", "-vm", self.vad_model]
            # None means auto-detect, exactly as it does for faster-whisper. whisper.cpp
            # spells that "auto" and treats an empty string as English, which would flip
            # every Indic call to the wrong language silently.
            cmd += ["-l", lang or "auto"]
            if hint and lang == "en":
                cmd += ["--prompt", hint]
            try:
                with self._lock:
                    subprocess.run(cmd, capture_output=True, timeout=self.timeout,
                                   check=True)
            except subprocess.TimeoutExpired:
                # Same reasoning as the `say` timeout: a wedged child must not hold a
                # threadpool worker forever, because enough of them stop the server
                # answering anybody.
                log.warning("whisper.cpp timed out after %.0fs — dropping the turn",
                            self.timeout)
                return "", None, 0.0
            except subprocess.CalledProcessError as e:
                log.warning("whisper.cpp failed (%s): %s", e.returncode,
                            (e.stderr or b"")[-200:])
                return "", None, 0.0
            with open(os.path.join(tmp, "out.json"), encoding="utf-8") as fh:
                out = _json.load(fh)
        finally:
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)

        segs = out.get("transcription") or []
        text = "".join(s.get("text", "") for s in segs).strip()
        det_lang = ((out.get("result") or {}).get("language")) or lang
        # None, not 0.0: the thresholds WERE applied, by the binary, using the same
        # numbers — but it does not report what it measured, and claiming certainty we
        # do not have would switch the ambiguous half of the artifact list off in
        # silence. judge() knows the difference.
        text = judge(text, no_speech=None, avg_logprob=None,
                     reject_no_speech=self.reject_no_speech,
                     reject_logprob=self.reject_logprob)
        if not text:
            return "", None, 0.0
        # None, not 0.0. whisper.cpp reports `result.language` with no confidence beside
        # it. Returning 0.0 made the orchestrator discard the detection and fall back to
        # guessing from the script — which cannot separate Marathi from Hindi, or
        # Assamese from Bengali. The same distinction judge() draws above.
        return text, det_lang, None


class WhisperCppServerSTT:
    """whisper.cpp with the model kept RESIDENT, instead of loaded once per turn.

    WHY THIS EXISTS. Timing the CLI against clip length showed the cost was not the
    audio at all:

        0.3s of speech → 644ms        10.5s of speech → 589ms

    Flat. Essentially all of it is process spawn plus reading a 490MB model off disk;
    the actual decode on the GPU is nearly free. Keeping the model loaded removes that
    floor — measured properly, all three backends warmed, best of 5 in one process:

        clip     faster-whisper   whisper-cli   whisper-server   vs faster-whisper
        0.3s             1420ms         895ms           615ms          2.31x
        3.5s             1738ms         841ms           621ms          2.80x
       10.5s             1674ms         903ms           670ms          2.50x

    A CORRECTION worth keeping, because the first number here was wrong: probing the
    server with curl suggested 4x. It is 2.3-2.8x. Two reasons the floor does not
    vanish entirely — `verbose_json` costs ~120ms more to serialise than plain text,
    and Whisper's encoder always runs over a padded 30-second window whatever the clip
    length, which is also why 0.3s of audio is not meaningfully cheaper than 3.5s.
    Measure the thing that ships, not a probe of it.

    It is also strictly more accurate than the CLI mode, which is the part that matters
    beyond speed: `verbose_json` returns per-segment `avg_logprob` and `no_speech_prob`,
    so the confidence gate and the full artifact list work here exactly as they do on
    faster-whisper. The CLI has to pass None for those and lose half the check.

    LOCAL BY DESIGN. The server is spawned as a CHILD of this process, bound to
    127.0.0.1 on a free port, and killed on exit. There is nothing to deploy, nothing
    to configure, and no port left listening after the app stops. Point
    `whispercpp_url` at an already-running instance if you would rather manage it
    yourself.
    """

    _CFG = "__cfg__"

    def __init__(self, cfg: dict):
        import atexit
        import os
        import shutil
        import socket
        import subprocess
        import time

        self.language = cfg.get("language")
        self.reject_no_speech = float(cfg.get("reject_no_speech", 0.85))
        self.reject_logprob = float(cfg.get("reject_logprob", -1.6))
        self.timeout = float(cfg.get("whispercpp_timeout", 60))
        self.denoise = bool(cfg.get("denoise", False))
        self.denoiser = None
        if (cfg.get("denoise_backend") or "").lower() in ("deepfilternet", "deepfilter", "df"):
            from .denoise import get_denoiser
            self.denoiser = get_denoiser(cfg)
        self._decode = FasterWhisperSTT._decode.__get__(self)
        # One request at a time — and MEASURED to cost nothing, so do not "optimise" it
        # away. Bypassing this lock and hitting the server with concurrent HTTP gives
        # 2.04x wall for 2 requests and 4.05x for 4: whisper-server serialises
        # internally on a single Metal context, exactly as this lock does. Removing it
        # would buy no throughput and lose the predictability.
        #     with the lock:    2 turns 2.07x · 4 turns 4.12x
        #     without the lock: 2 turns 2.04x · 4 turns 4.05x
        # Concurrency at `max_sessions: 6` is therefore a hardware property here, not a
        # code one — the answer is a second box or a GPU, not a different lock.
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None

        url = cfg.get("whispercpp_url")
        if url:
            self.url = url.rstrip("/")
            log.info("STT: whisper.cpp server (external, %s)", self.url)
            self._probe()
            return

        binary = cfg.get("whispercpp_server_bin") or shutil.which("whisper-server")
        if not binary:
            raise RuntimeError("whisper-server not found (brew install whisper-cpp)")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = cfg.get("whispercpp_model") or "models/whispercpp/ggml-small.bin"
        model = model if os.path.isabs(model) else os.path.join(root, model)
        if not os.path.isfile(model):
            raise RuntimeError("ggml model missing: %s" % model)

        # Ask the OS for a free port rather than picking one: a hard-coded port turns a
        # second instance, or anything else on 8080, into a confusing failure.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = "http://127.0.0.1:%d" % port

        cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port),
               "-t", str(best_thread_count(cfg)),
               "-bs", str(int(cfg.get("beam_size", 3))),
               "-nth", str(self.reject_no_speech), "-lpt", str(self.reject_logprob)]
        vad = None
        if cfg.get("vad_filter", False):
            for cand in WhisperCppSTT._vad_candidates(root, cfg):
                if os.path.isfile(cand):
                    vad = cand
                    break
        if vad:
            # Same trap as the CLI: an .onnx here ABORTS the process rather than
            # erroring, and the abort happens at startup where it looks like the binary
            # is simply broken.
            cmd += ["--vad", "-vm", vad]

        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        deadline = time.time() + float(cfg.get("whispercpp_boot_timeout", 90))
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("whisper-server exited during startup (rc=%s)"
                                   % self._proc.returncode)
            try:
                self._probe()
                log.info("STT: whisper.cpp server on %s (model resident — measured "
                         "2.3-2.8x faster than faster-whisper)", self.url)
                return
            except Exception:  # noqa: BLE001  (still loading the model)
                time.sleep(0.5)
        self.close()
        raise RuntimeError("whisper-server did not become ready in time")

    def _probe(self):
        import urllib.request
        with urllib.request.urlopen(self.url + "/", timeout=3):
            return True

    def close(self):
        """Stop the child. Registered with atexit so no port is left listening."""
        c, self._conn = self._conn, None
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        p, self._proc = self._proc, None
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass

    def _post(self, body: bytes, boundary: str) -> bytes:
        """POST on a KEPT-ALIVE connection.

        Measured: a fresh urllib request per turn cost ~90ms more than the same request
        from curl, all of it connection setup against a server on loopback. Holding one
        connection open is free here because `_lock` already serialises requests — there
        is never a second one in flight to contend for it. Rebuilt on any error, so a
        server restart costs one turn rather than every turn after it.
        """
        import http.client
        for attempt in (1, 2):
            try:
                if self._conn is None:
                    host = self.url.split("//", 1)[-1]
                    self._conn = http.client.HTTPConnection(host, timeout=self.timeout)
                self._conn.request(
                    "POST", "/inference", body,
                    {"Content-Type": "multipart/form-data; boundary=" + boundary,
                     "Connection": "keep-alive"})
                return self._conn.getresponse().read()
            except Exception:  # noqa: BLE001
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._conn = None
                if attempt == 2:
                    raise
        return b""

    def transcribe(self, audio, hint: str | None = None, language=_CFG,
                   fast: bool | None = None, denoise: bool | None = None):
        import io as _io
        import json as _json
        import uuid

        import numpy as np
        import soundfile as sf

        data = self._decode(audio)
        if denoise is None:
            denoise = self.denoise
        if denoise and self.denoiser is not None:
            data = self.denoiser(data, 16000)
        data = np.asarray(data, dtype="float32").reshape(-1)
        if not data.size:
            return "", None, 0.0

        wav = _io.BytesIO()
        sf.write(wav, data, 16000, format="WAV", subtype="PCM_16")
        lang = self.language if language == self._CFG else language

        boundary = "----zs" + uuid.uuid4().hex
        parts = []

        def field(name, value):
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                          % (boundary, name, value)).encode())

        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
                      "filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
                      % boundary).encode())
        parts.append(wav.getvalue())
        parts.append(b"\r\n")
        # verbose_json is what makes this backend fully hardened rather than merely
        # fast: it carries avg_logprob and no_speech_prob, so judge() gets real numbers.
        field("response_format", "verbose_json")
        field("language", lang or "auto")
        if hint and lang == "en":
            field("prompt", hint)
        parts.append(("--%s--\r\n" % boundary).encode())
        body = b"".join(parts)

        try:
            with self._lock:
                raw = self._post(body, boundary)
            out = _json.loads(raw.decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            # Same contract as every other failure on this path: drop the turn, never
            # raise into the websocket loop.
            log.warning("whisper-server request failed (%s) — dropping the turn", e)
            return "", None, 0.0

        text = (out.get("text") or "").strip()
        segs = out.get("segments") or []
        det = _lang_code(out.get("detected_language") or out.get("language"))
        prob = out.get("detected_language_probability")
        prob = float(prob) if isinstance(prob, (int, float)) else None

        # Duration-weighted, exactly as the faster-whisper path does it: one brief pause
        # must not be able to condemn a long real sentence.
        no_speech = avg_lp = None
        if segs:
            tot = sum(max(0.05, (s.get("end", 0) - s.get("start", 0))) for s in segs) or 1.0
            no_speech = sum(float(s.get("no_speech_prob", 0.0))
                            * max(0.05, s.get("end", 0) - s.get("start", 0))
                            for s in segs) / tot
            avg_lp = sum(float(s.get("avg_logprob", 0.0))
                         * max(0.05, s.get("end", 0) - s.get("start", 0))
                         for s in segs) / tot

        text = judge(text, no_speech=no_speech, avg_logprob=avg_lp,
                     reject_no_speech=self.reject_no_speech,
                     reject_logprob=self.reject_logprob)
        if not text:
            return "", None, 0.0
        return text, det, prob


class TwoPassSTT:
    """A tiny model for the partials, the accurate one for the turn.

    THE PATTERN, and where it comes from. Wispr Flow runs a ~120M realtime model for
    streaming partials and a large model for the final commit; the partials exist to
    make the interface feel live and to decide when the speaker has stopped, not to be
    the transcript. This codebase already had the SHAPE of that — speculative STT at
    ~450ms of silence, held and discarded if the caller resumes — but ran the SAME
    `small` model for both, so a guess cost exactly as much as an answer.

    That matters because of what the guess is FOR. Three turn-taking signals are
    computed from it and nothing else: `looks_incomplete` (the words), `looks_complete`
    (a finished phone number), and the pitch contour. All three are gated behind a full
    621ms recognition, so the endpointer cannot react until the caller has already been
    silent for most of the window it is trying to size.

    WHAT IT IS NOT FOR. The partial never reaches the LLM, never reaches the guard, and
    never becomes the transcript — `tiny` is measurably worse and would put its errors
    into bookings. `server.py` already enforces this ("Nothing reaches the LLM until the
    endpoint is confirmed"); this class only makes the guess cheaper.

    Falls back to the accurate model for everything if the partial one cannot be built,
    so a machine with one model still works exactly as before.
    """

    _CFG = "__cfg__"

    def __init__(self, commit, partial):
        self.commit = commit
        self.partial = partial
        # THE WHOLE SURFACE the engine reads off a provider, forwarded from the model
        # that produces the TRANSCRIPT. This is not decoration: `_decode` was missing in
        # the first version and `Session.clean_audio` passes `self.stt._decode` into the
        # audio pipeline, so EVERY microphone turn raised AttributeError before it
        # reached the recogniser — the agent simply stopped hearing anyone, with the
        # failure buried in a threadpool traceback. Discovered from the log, not the
        # tests, which is why test_two_pass_forwards_the_whole_provider_surface now
        # derives this list from the source instead of trusting this comment.
        self.language = getattr(commit, "language", None)
        self.denoiser = getattr(commit, "denoiser", None)
        # …and forwarded DEFENSIVELY. Requiring it at construction would break any
        # provider that does not happen to expose one, which is the same "a new required
        # thing broke every adapter" mistake the denoise kwarg made once. Falling back to
        # the canonical implementation keeps the attribute a guarantee rather than a hope.
        self._decode = getattr(commit, "_decode", None) \
            or FasterWhisperSTT._decode.__get__(self)
        log.info("STT: two-pass — %s for partials, %s for turns",
                 type(partial).__name__, type(commit).__name__)

    def close(self):
        for m in (self.partial, self.commit):
            c = getattr(m, "close", None)
            if c:
                c()

    def transcribe(self, audio, hint: str | None = None, language=_CFG,
                   fast: bool | None = None, denoise: bool | None = None,
                   partial: bool = False):
        model = self.partial if partial else self.commit
        kw = {"hint": hint, "language": language, "fast": fast, "denoise": denoise}
        try:
            return model.transcribe(audio, **kw)
        except TypeError:
            # Same contract the rest of this file uses: a provider that predates a
            # kwarg must keep working rather than taking voice input down.
            kw.pop("denoise", None)
            return model.transcribe(audio, **kw)
