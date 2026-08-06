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
_HALLUCINATION_PHRASES = {
    "thank you for watching", "thanks for watching", "thank you for watching!",
    "please subscribe", "subscribe", "like and subscribe", "please subscribe to my channel",
    "thanks for watching!", "see you next time", "see you in the next video",
    "www.mooji.org", "amara.org", "subtitles by the amara.org community",
    "i'll see you next time", "bye bye", "thank you.", "you", ".", "..", "...",
    "मुझे लगता है", " धन्यवाद", "शुक्रिया देखने के लिए",
}


def _is_hallucination(text: str) -> bool:
    """True when the WHOLE transcript is a known silence/noise artifact."""
    t = text.strip().lower().strip(" .!?।-…")
    if not t:
        return True
    if t in _HALLUCINATION_PHRASES:
        return True
    # a lone 1–2 char fragment with no real word is almost always noise
    if len(re.sub(r"\W", "", t)) <= 1:
        return True
    return False


def get_stt(cfg: dict):
    provider = (cfg or {}).get("provider", "faster_whisper")
    if provider in (None, "none"):
        return None
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
        self.model = WhisperModel(
            cfg.get("model", "tiny"),
            device=cfg.get("device", "cpu"),
            compute_type=cfg.get("compute_type", "int8"),
        )
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
        if no_speech > self.reject_no_speech:   # duration-weighted: mostly non-speech
            log.info("STT dropped (no_speech=%.2f): %r", no_speech, text[:40])
            return "", None, 0.0
        if avg_lp < self.reject_logprob:        # duration-weighted low confidence → garbage
            log.info("STT dropped (avg_logprob=%.2f): %r", avg_lp, text[:40])
            return "", None, 0.0
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
                return "", None, 0.0
        # a known Whisper artifact ("thank you for watching") — but "thank you"/"bye" are also
        # REAL utterances, so only drop the artifact when the audio also looks like non-speech.
        if _is_hallucination(text) and no_speech > 0.5:
            log.info("STT dropped (hallucination, no_speech=%.2f): %r", no_speech, text[:40])
            return "", None, 0.0
        return text, det_lang, prob
