"""Tiny SQLite booking store — stands in for the per-vertical calendar/POS/EMR
integration an Industry Pack would wire up in production.
"""
import json
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "zensuvidha.db"


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB, timeout=5.0)
    c.execute("PRAGMA journal_mode=WAL")     # concurrent readers while writing
    c.execute("PRAGMA busy_timeout=5000")    # wait instead of erroring on lock
    c.execute(
        """CREATE TABLE IF NOT EXISTS bookings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pack TEXT, name TEXT, phone TEXT, service TEXT,
            when_ TEXT, notes TEXT, slots TEXT, created REAL)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS turns(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT, pack TEXT, role TEXT, text TEXT,
            lang TEXT, latency_ms INTEGER, created REAL)"""
    )
    return c


def create_booking(pack: str, slots: dict) -> int:
    c = _conn()
    try:
        cur = c.execute(
            "INSERT INTO bookings(pack,name,phone,service,when_,notes,slots,created) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                pack,
                slots.get("name"),
                slots.get("phone"),
                slots.get("service") or slots.get("doctor") or slots.get("party_size"),
                slots.get("datetime") or slots.get("when") or slots.get("time"),
                slots.get("notes"),
                json.dumps(slots, ensure_ascii=False),
                time.time(),
            ),
        )
        c.commit()
        return cur.lastrowid
    finally:
        c.close()


def list_bookings(pack: str | None = None) -> list[dict]:
    c = _conn()
    q = "SELECT id,pack,name,phone,service,when_,created FROM bookings"
    q += " WHERE pack=?" if pack else ""
    q += " ORDER BY id DESC"
    rows = c.execute(q, ((pack,) if pack else ())).fetchall()
    c.close()
    cols = ["id", "pack", "name", "phone", "service", "when", "created"]
    return [dict(zip(cols, r)) for r in rows]


# ---- call-transcript persistence (QA / audit / call history) --------------- #
def log_turn(call_id: str, pack: str, role: str, text: str,
             lang: str | None = None, latency_ms: int | None = None) -> None:
    """Persist one conversation turn (caller text OR assistant reply)."""
    if not (text or "").strip():
        return
    c = _conn()
    c.execute(
        "INSERT INTO turns(call_id,pack,role,text,lang,latency_ms,created) VALUES(?,?,?,?,?,?,?)",
        (call_id, pack, role, text, lang, latency_ms, time.time()),
    )
    c.commit()
    c.close()


def list_turns(call_id: str | None = None, pack: str | None = None, limit: int = 200) -> list[dict]:
    c = _conn()
    q = "SELECT id,call_id,pack,role,text,lang,latency_ms,created FROM turns"
    where, args = [], []
    if call_id:
        where.append("call_id=?"); args.append(call_id)
    if pack:
        where.append("pack=?"); args.append(pack)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    rows = c.execute(q, tuple(args)).fetchall()
    c.close()
    cols = ["id", "call_id", "pack", "role", "text", "lang", "latency_ms", "created"]
    return [dict(zip(cols, r)) for r in rows]
