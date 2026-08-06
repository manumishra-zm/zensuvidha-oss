"""Text CLI — the zero-friction way to try the engine. Needs only Ollama.

  python -m zensuvidha.cli --pack clinic
  python -m zensuvidha.cli --pack restaurant --speak     # also synthesize audio
"""
import argparse

from .config import load_config
from .llm import OllamaLLM
from .orchestrator import Session
from .packs import list_packs, load_pack


def _tag(res: dict) -> str:
    a = res.get("action", {}) or {}
    t = a.get("type", "none")
    if t == "book_appointment" or a.get("booking_id"):
        return f"  ✅ booked #{a.get('booking_id')}"
    if t == "collect" and a.get("missing"):
        return f"  … still needs: {', '.join(a['missing'])}"
    if res.get("escalated"):
        return "  ⚠ escalating to human"
    return ""


def _play(tts, text):
    """Best-effort playback of a reply through the OS audio player."""
    if not tts:
        return
    import os
    import subprocess
    import sys
    import tempfile
    try:
        data = tts.synth(text)
        if not data:
            return
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(data)
        f.close()
        if sys.platform == "darwin":
            player = ["afplay", f.name]
        elif sys.platform.startswith("linux"):
            player = ["aplay", "-q", f.name]
        else:
            player = None  # Windows: skip playback in CLI (use the web UI)
        if player:
            subprocess.run(player, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(f.name)
    except Exception as e:  # noqa: BLE001
        print(f"[audio] {e}")


def main():
    ap = argparse.ArgumentParser(description="ZenSuvidha OSS — text CLI")
    ap.add_argument("--pack", default=None, help=f"one of: {', '.join(list_packs())}")
    ap.add_argument("--speak", action="store_true", help="synthesize + play replies (TTS)")
    args = ap.parse_args()

    cfg = load_config()
    pack_name = args.pack or cfg.get("default_pack", "clinic")
    pack = load_pack(pack_name)
    llm = OllamaLLM(cfg["llm"])

    ok, info = llm.health()
    if not ok:
        print(f"⚠  Ollama not reachable ({info}).")
        print("   Start Ollama and run:  ollama pull", cfg["llm"]["model"])
        return
    if cfg["llm"]["model"] not in info:
        print(f"⚠  Model '{cfg['llm']['model']}' not pulled. Run:  ollama pull {cfg['llm']['model']}")

    tts = None
    if args.speak:
        from .tts import get_tts
        tts = get_tts(cfg["tts"])

    session = Session(pack, llm, tts=tts, guard_cfg=cfg.get("guard"))
    print(f"\n─ ZenSuvidha · pack: {pack_name} ─ (available: {', '.join(list_packs())})")
    print("  Type 'quit' to end. Try: \"I'd like to book an appointment\"\n")
    g = session.greeting()
    print("Agent:", g)
    _play(tts, g)

    while True:
        try:
            u = input("You:   ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if u.lower() in {"quit", "exit", "/q"}:
            break
        if not u:
            continue
        res = session.handle_text(u)
        print("Agent:", res["say"], _tag(res))
        _play(tts, res["say"])
        if res.get("escalated"):
            print("       [call would now transfer to a human agent]")


if __name__ == "__main__":
    main()
