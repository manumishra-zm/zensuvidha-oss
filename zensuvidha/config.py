"""Load config.yaml with a few environment-variable overrides."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None) -> dict:
    path = Path(path or os.environ.get("ZS_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(path.read_text()) or {}

    # convenient env overrides (handy for Docker / quick tests)
    llm = cfg.setdefault("llm", {})
    llm["base_url"] = os.environ.get("OLLAMA_URL", llm.get("base_url", "http://localhost:11434"))
    llm["model"] = os.environ.get("ZS_LLM_MODEL", llm.get("model", "qwen3:4b"))
    if os.environ.get("ZS_TTS"):
        cfg.setdefault("tts", {})["provider"] = os.environ["ZS_TTS"]
    if os.environ.get("ZS_DEFAULT_PACK"):
        cfg["default_pack"] = os.environ["ZS_DEFAULT_PACK"]
    return cfg
