"""Speaker gating — "is this the SAME person?", not just "is this speech?".

Silero tells us a sound is speech. It cannot tell us WHOSE. So a colleague talking
across the room, a television, or a second person leaning into the mic all open a turn
and get answered as though they were the caller. Loudness is the only thing currently
separating them, and a level check is not speaker recognition.

This module takes a voiceprint of the first person heard on the call and drops later
utterances that don't match it. The comparison is an ECAPA-TDNN embedding (the same
family used for speaker verification) reduced to a cosine similarity — frequency
analysis proper: a mel filterbank the network turns into a 192-dimensional identity,
rather than the single amplitude number the VAD works from.

Capability-gated like every other optional provider here: if the model isn't installed
the gate is simply absent and every utterance is accepted, so nothing that works today
can break by this existing.

Backends (both free, both offline after first download):
  speechbrain  — ECAPA-TDNN, Apache-2.0, ~80MB   [default, most accurate]
  resemblyzer  — d-vector,   MIT,        ~17MB   [lighter, weaker]

License note: the model weights are Apache-2.0 and NOT gated, unlike Indic-Parler.
"""
import logging
import threading

log = logging.getLogger("zensuvidha.speaker")

# Below this, an utterance is too short to take a reliable voiceprint from. ECAPA needs
# roughly a second of voiced audio; scoring a 400ms "haan" produces noise, and rejecting
# the caller because they answered briefly is far worse than letting a stray voice past.
MIN_ENROL_S = 1.0
MIN_VERIFY_S = 0.6


def get_speaker_gate(cfg: dict):
    """Build the gate, or return None if it is disabled or unavailable."""
    cfg = cfg or {}
    if not cfg.get("speaker_gate", False):
        return None
    backend = (cfg.get("speaker_backend") or "speechbrain").lower()
    try:
        gate = SpeakerGate(cfg, backend)
    except Exception as e:  # noqa: BLE001
        log.warning("speaker gate unavailable (%s) — every voice will be accepted: %s",
                    backend, e)
        return None
    log.info("Speaker gate on: %s, threshold %.2f", backend, gate.threshold)
    return gate


def _voiced_only(data, sr: int):
    """Drop frames that are near the clip's own noise floor.

    Deliberately conservative: the floor is a low percentile of frame energy, the gate
    sits well above it, and a little padding is kept around each voiced run so word
    onsets survive. If that leaves almost nothing (a clip that really is all speech, or
    really is all silence) the original is returned — this must never be able to empty
    an utterance on its own.
    """
    import numpy as np
    hop = max(1, sr // 100)                       # 10ms frames
    n = data.size // hop
    if n < 3:
        return data
    frames = data[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype("float64") ** 2).mean(axis=1))
    floor = float(np.percentile(rms, 20))
    peak = float(rms.max())
    if peak <= 0:
        return data
    gate = max(floor * 3.0, peak * 0.08)
    voiced = rms > gate
    if not voiced.any():
        return data
    # Keep the SPAN from the first voiced frame to the last, and nothing clever inside
    # it. Splicing out the internal gaps — which this used to do — concatenates
    # non-adjacent audio, and every join is an artificial transient the encoder hears as
    # part of the voice. Measured on one caller's own clip:
    #
    #     spliced (gaps removed)   same speaker 0.450   <- BELOW the 0.55 threshold:
    #                                                      our own trimming refused them
    #     voiced span              same speaker 0.843
    #     no trimming at all       same speaker 0.830
    #
    # The span keeps everything the trim existed for — a one-word answer padded with
    # pre-roll and endpoint silence still loses its padding — without inventing edges
    # in the middle of a sentence.
    pad = 3                                        # ~30ms either side, so onsets survive
    idx = np.flatnonzero(voiced)
    a = max(0, int(idx[0]) - pad)
    b = min(n, int(idx[-1]) + pad + 1)
    out = frames[a:b].reshape(-1)
    return out if out.size >= hop * 3 else data


def _to_16k_mono(audio):
    """Any WAV bytes / array → float32 mono at 16kHz, which is what the models expect."""
    import numpy as np
    if isinstance(audio, np.ndarray):
        return audio.astype("float32"), 16000
    import io
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = data.astype("float32")
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                         np.arange(len(data)), data).astype("float32")
        sr = 16000
    return data, sr


class SpeakerGate:
    """One shared model; the voiceprint itself lives on each Session."""

    def __init__(self, cfg: dict, backend: str = "speechbrain"):
        self.backend = backend
        self.threshold = float(cfg.get("speaker_threshold", 0.55))
        self._lock = threading.Lock()      # inference is not guaranteed concurrent-safe
        if backend == "resemblyzer":
            from resemblyzer import VoiceEncoder
            self._enc = VoiceEncoder(verbose=False)
        else:
            from speechbrain.inference.speaker import EncoderClassifier
            model = cfg.get("speaker_model") or "speechbrain/spkrec-ecapa-voxceleb"
            self._enc = EncoderClassifier.from_hparams(
                source=model, savedir=f"models/{model.split('/')[-1]}")

    # ---- embedding ---------------------------------------------------------
    def embed(self, audio, min_seconds: float = MIN_VERIFY_S):
        """A unit-length voiceprint, or None when there is too little VOICE to trust.

        The length test measures VOICED audio, not clip length, and only the voiced part
        is encoded. Measuring the whole clip was doubly wrong, both ways:

          * a one-word answer ("haan") arrives as 300ms pre-roll + 350ms of voice +
            800ms of endpoint silence. The clip clears any length test, but ECAPA then
            pools over 76% silence and scores the real caller 0.526 — below threshold,
            so their confirmation was silently ignored. Measured: 2.5s voiced -> 0.985,
            0.8s -> 0.754, 0.5s -> 0.599, 0.35s -> 0.526 REJECT. The same 0.35s with no
            padding is correctly refused and therefore ACCEPTED, so the padding alone
            decided it.
          * a short first turn ("hello") could clear MIN_ENROL_S on total samples and
            enrol a silence-dominated voiceprint for the whole call.
        """
        import numpy as np
        try:
            data, sr = _to_16k_mono(audio)
        except Exception as e:  # noqa: BLE001
            log.debug("speaker: could not decode audio: %s", e)
            return None
        data = _voiced_only(data, sr)
        if data.size < int(min_seconds * sr):
            return None
        try:
            with self._lock:
                if self.backend == "resemblyzer":
                    vec = np.asarray(self._enc.embed_utterance(data), dtype="float32")
                else:
                    import torch
                    t = torch.from_numpy(data).unsqueeze(0)
                    vec = self._enc.encode_batch(t).squeeze().detach().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            log.warning("speaker: embedding failed: %s", e)
            return None
        norm = float(np.linalg.norm(vec))
        return (vec / norm) if norm else None

    @staticmethod
    def similarity(a, b) -> float:
        """Cosine similarity of two unit vectors — 1.0 identical, ~0 unrelated."""
        import numpy as np
        if a is None or b is None:
            return 0.0
        return float(np.dot(np.asarray(a), np.asarray(b)))

    def judge(self, voiceprint, audio):
        """(accept, similarity, embedding) — `matches` plus the vector it computed.

        The embedding is by far the most expensive part of the gate, and the rejection
        path needs the SAME vector again (to decide whether it is the same voice being
        refused over and over). Returning it here means one encoder pass per turn
        instead of two.
        """
        if voiceprint is None:
            return True, None, None
        vec = self.embed(audio)
        if vec is None:
            return True, None, None               # too short to judge → give the benefit
        sim = self.similarity(voiceprint, vec)
        return sim >= self.threshold, sim, vec

    def matches(self, voiceprint, audio):
        """(accept, similarity). Accepts when we cannot judge — a false reject silences
        the real caller, which is far worse than letting one stray utterance through."""
        ok, sim, _vec = self.judge(voiceprint, audio)
        return ok, sim
