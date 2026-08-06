"""Production logging setup.

Features (all config-driven under `server:` in config.yaml):
  * configurable level                 — server.log_level  (DEBUG/INFO/WARNING/…)
  * rotating file log                  — server.log_file   (path; None = console only)
  * structured JSON logs               — server.log_json   (for Loki/Datadog/CloudWatch)
  * per-CALL correlation id on every   — set via set_call_id(); injected into every record
    line so one caller's turns are traceable under concurrent load.

Correlation ids ride a contextvar, so any async task spawned while handling a call
(turn task, TTS synth, filler) inherits the caller's id automatically.
"""
import contextvars
import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path

# per-call id (e.g. "a3f9c1"); "-" outside of a call
call_id_var: "contextvars.ContextVar[str]" = contextvars.ContextVar("call_id", default="-")


def set_call_id(cid: str) -> None:
    call_id_var.set(cid or "-")


class _CallIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.call_id = call_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                  + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "call": getattr(record, "call_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def setup_logging(cfg: dict) -> int:
    """Configure the root logger from config. Idempotent (safe to call once at startup)."""
    scfg = (cfg or {}).get("server", {})
    level = getattr(logging, str(scfg.get("log_level", "INFO")).upper(), logging.INFO)
    as_json = bool(scfg.get("log_json", False))
    log_file = scfg.get("log_file")   # e.g. "logs/zensuvidha.log"; None → console only

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):     # replace anything basicConfig may have installed
        root.removeHandler(h)

    fmt: logging.Formatter = _JsonFormatter() if as_json else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(call_id)s]: %(message)s")
    cid = _CallIdFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(cid)
    root.addHandler(console)

    if log_file:
        p = Path(log_file)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(p), maxBytes=int(scfg.get("log_file_max_bytes", 5_000_000)),
            backupCount=int(scfg.get("log_file_backups", 5)), encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(cid)
        root.addHandler(fh)

    logging.getLogger("httpx").setLevel(logging.WARNING)   # quiet per-request HTTP noise
    return level
