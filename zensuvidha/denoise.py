"""DeepFilterNet noise suppression, as a live A/B switch in front of STT.

WHY THIS IS OFF BY DEFAULT — measured on this codebase, not assumed:

  Whisper WER          raw 0.00   ·  DeepFilterNet 0.10
  Speaker similarity   raw 0.675  ·  DeepFilterNet 0.596
  SNR sweep +20..-5dB  raw won 2, tied 8, DeepFilterNet won 0 of 10

Whisper was trained on 680k hours of messy audio and ECAPA on YouTube interviews;
both learned their own noise robustness, so filtering in front of them removes detail
they use. DeepFilterNet is by far the strongest denoiser of the ones tried — and that
is exactly why it costs the most here: the harm scales with the suppression.

It is wired in anyway so the trade-off can be heard rather than argued about. Flip the
toggle in the UI (or `stt.denoise_backend`) and compare live.

Runs as the standalone Rust binary, NEVER the pip wheel: `pip install deepfilternet`
pins numpy<2 and imports `torchaudio.backend.common` (removed in modern torchaudio),
which silently downgraded numpy here and broke the speaker gate while tests stayed green.

  bash scripts/download_deepfilter.sh      # ~28MB, MIT/Apache-2.0, models unrestricted

Cost, measured: ~115 ms per utterance, FLAT — 8 s of audio costs the same as 0.5 s,
because it is entirely process spawn. The filtering itself is effectively free.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import threading

log = logging.getLogger("zensuvidha.denoise")

DEFAULT_BIN = "models/deep-filter"
SR = 48000               # DeepFilterNet is 48 kHz only; we resample around it


def get_denoiser(cfg: dict):
    """Build the denoiser, or None when it isn't available. Never raises."""
    cfg = cfg or {}
    atten = cfg.get("deepfilter_atten_db")
    path = cfg.get("deepfilter_bin") or DEFAULT_BIN
    if not os.path.isabs(path):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, path)
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        found = shutil.which("deep-filter")
        if not found:
            log.info("DeepFilterNet not installed (%s) — the toggle will stay off. "
                     "Run: bash scripts/download_deepfilter.sh", path)
            return None
        path = found
    try:
        return DeepFilterDenoiser(path, atten_db=atten)
    except Exception as e:  # noqa: BLE001
        log.warning("DeepFilterNet unavailable: %s", e)
        return None


class DeepFilterDenoiser:
    """One binary, called per utterance. Thread-safe; failures fall back to raw audio."""

    def __init__(self, binary: str, atten_db=None):
        self.binary = binary
        # How far it is ALLOWED to suppress. 100 dB (the tool's default) = unlimited;
        # 0 dB = no suppression at all. Measured: clean speech survives intact even at
        # the default (98% of energy kept down to a whisper), but speech ENTANGLED with
        # heavy noise does get eaten (WER 0.15 -> 0.38 at -5dB SNR). Lowering this bounds
        # how much of the caller can ever be removed, at the cost of less noise removal.
        self.atten_db = None if atten_db in (None, "", "null") else float(atten_db)
        self._lock = threading.Lock()
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        self.version = (out.stdout or out.stderr).strip().splitlines()[0] if out.stdout or out.stderr else "?"
        log.info("DeepFilterNet ready: %s (%s)%s", self.version, binary,
                 f" atten-limit {self.atten_db}dB" if self.atten_db is not None else "")

    def __call__(self, data, sr: int = 16000):
        """float32 mono in → float32 mono out at the SAME rate. Returns the input
        unchanged on any failure: a denoiser must never be able to drop a turn."""
        import numpy as np
        import soundfile as sf
        try:
            up = _resample(data, sr, SR)
            with tempfile.TemporaryDirectory() as td:
                src = os.path.join(td, "u.wav")
                sf.write(src, up, SR, subtype="PCM_16")
                cmd = [self.binary, "-o", td]
                if self.atten_db is not None:
                    cmd += ["-a", str(self.atten_db)]
                with self._lock:
                    r = subprocess.run(cmd + [src], capture_output=True, timeout=120)
                if r.returncode != 0:
                    log.warning("deep-filter exited %s — using raw audio", r.returncode)
                    return data
                # the binary writes its result into the output dir; take the newest wav
                # that isn't the file we just wrote
                cands = [os.path.join(td, f) for f in os.listdir(td) if f.endswith(".wav")]
                cands = [c for c in cands if os.path.getmtime(c) >= os.path.getmtime(src)]
                if not cands:
                    log.warning("deep-filter produced no output — using raw audio")
                    return data
                got, out_sr = sf.read(max(cands, key=os.path.getmtime))
            if getattr(got, "ndim", 1) > 1:
                got = got.mean(axis=1)
            return _resample(np.asarray(got, dtype="float32"), out_sr, sr)
        except Exception as e:  # noqa: BLE001
            log.warning("denoise failed (%s) — using raw audio", e)
            return data


def _resample(x, src: int, dst: int):
    import numpy as np
    if src == dst:
        return np.asarray(x, dtype="float32")
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype("float32")
