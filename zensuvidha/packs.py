"""Industry Pack loader — the plug-in / plug-out layer.

A pack is pure YAML data that overrides `_base.yaml`. Adding a new industry
means dropping a new packs/<name>.yaml file. No code changes.
"""
from pathlib import Path
import os
import re

import yaml

ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = ROOT / "packs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def list_packs() -> list[str]:
    return sorted(p.stem for p in PACK_DIR.glob("*.yaml") if not p.stem.startswith("_"))


# A pack name arrives straight from a URL query param and a WS `switch` message, so it
# is caller-controlled. Unvalidated it is a path-traversal primitive: `../../etc/foo`
# loads any .yaml on disk, and whatever sits under its `greeting:` key is then SPOKEN
# back down the phone. Restrict it to the shape a real pack id actually has.
_SAFE_PACK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


_PACK_CACHE: dict = {}     # (path, mtime) -> merged pack, with its
                           #   derived index and vocabulary attached


def load_pack(name: str) -> dict:
    name = str(name or "")
    if not _SAFE_PACK.match(name) or name.startswith("_"):
        raise FileNotFoundError(
            f"Industry pack '{name[:40]}' not found. Available: {', '.join(list_packs())}")
    base_path = PACK_DIR / "_base.yaml"
    path = (PACK_DIR / f"{name}.yaml").resolve()
    # Belt and braces: even a name that satisfies the pattern must resolve INSIDE the
    # pack directory (symlinks, case-folding filesystems).
    if not str(path).startswith(str(PACK_DIR.resolve()) + os.sep):
        raise FileNotFoundError(f"Industry pack '{name[:40]}' not found.")
    if not path.exists():
        raise FileNotFoundError(
            f"Industry pack '{name}' not found. Available: {', '.join(list_packs())}"
        )
    # Cached on (path, mtime). Measured at 51ms per call — two YAML reads and a deep
    # merge — paid on EVERY session, at exactly the moment the caller is waiting for a
    # greeting. It also threw away the derived structures that live on the pack (the
    # semantic index, the expectation vocabulary), so those rebuilt per call too.
    #
    # Keyed by modification time rather than name alone, so editing a pack during
    # development still takes effect on the next call without a restart.
    key = (str(path), path.stat().st_mtime_ns,
           base_path.stat().st_mtime_ns if base_path.exists() else 0)
    hit = _PACK_CACHE.get(key)
    if hit is not None:
        return hit
    # Read AFTER the cache check. The first version parsed _base.yaml before looking,
    # so a cache hit still cost a file read and a YAML parse — 6.7ms of the 51ms it was
    # meant to remove.
    base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}
    merged = _deep_merge(base or {}, yaml.safe_load(path.read_text()) or {})
    merged["id"] = name
    # Bounded: one entry per pack per edit. Without a cap, a pack edited in a loop
    # during development would grow this without limit.
    if len(_PACK_CACHE) > 32:
        _PACK_CACHE.clear()
    _PACK_CACHE[key] = merged
    return merged
