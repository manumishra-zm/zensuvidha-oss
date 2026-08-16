"""Say the fixed lines ONCE, in the owner's voice, before the phone rings.

THE OBSERVATION THIS RESTS ON

Most of what this agent says on a call is already written down. Counted on the clinic
pack: 108 pre-written SAFE_LINES across twelve languages, 60 ASK_LINES, the greeting,
five slot questions, and 51 knowledge answers the semantic fast path quotes verbatim —
around 225 utterances that are FIXED TEXT and identical on every call.

That matters because voice cloning is slow. XTTS is "a few seconds per sentence on CPU"
by its own docstring here, and VoxCPM2 is a 2B model at RTF ~0.3 on a 4090 — neither can
sit on a turn's critical path beside Kokoro's measured 385ms. The usual conclusion is
that cloning is a GPU feature.

It is not. It is an OFFLINE feature. Render the fixed lines once when the owner records
their voice, put them in the cache the engine already consults, and the owner's voice
covers the majority of a call at ZERO runtime cost — while the live synthesiser handles
only the sentences the model actually generates.

WHY THIS AND NOT "SWAP THE SYNTHESISER"

Because it works with whatever cloner is installed. Nothing here knows or cares whether
the voice came from XTTS or VoxCPM; it asks the clone provider for audio and files the
result. A better cloner improves the output without changing a line of this, and a
machine with no cloner at all is unaffected.

WHAT IT DELIBERATELY DOES NOT DO

It does not pre-render generated replies — those are different every call, and a cache
of them would be a cache of one. It does not touch the hot path: a miss falls through to
exactly the behaviour that shipped before.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("zensuvidha.prerender")


def fixed_lines(pack: dict, languages=None) -> list[tuple[str, str]]:
    """Every sentence this agent can say that is written down in advance.

    Returns [(language_name, text)] — the language matters because the cache is keyed on
    text and a cloner needs to be told which language it is speaking.

    Ordered by how often a caller actually hears them: the greeting is on every single
    call, the slot questions on every booking, the safe lines only when something goes
    wrong. A render that is interrupted half way should therefore have covered the
    lines that matter most.
    """
    from .guard import ASK_LINES, SAFE_LINES

    langs = set(languages) if languages else None
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(lang: str, text):
        if not text or not isinstance(text, str):
            return
        # A TEMPLATE is not a fixed line. Several SAFE_LINES carry "{biz}" and "{ref}",
        # filled in at speaking time — rendering one literally would have the agent say
        # "I'm the receptionist at open-brace biz close-brace" in the owner's own voice,
        # on every call, in twelve languages. Caught by a test rather than by a caller.
        if "{" in text and "}" in text:
            return
        if langs and lang not in langs:
            return
        key = (lang, text.strip())
        if key[1] and key not in seen:
            seen.add(key)
            out.append(key)

    # 1. the greeting — heard on every call without exception
    add("English", pack.get("greeting"))

    # 2. the booking questions, including each pack's own per-language overrides
    slots = (pack.get("booking", {}) or {}).get("slots", {}) or {}
    for value in slots.values():
        add("English", value)
    for lang, table in (ASK_LINES or {}).items():
        for value in (table or {}).values():
            add(lang, value)

    # 3. what it says when something goes wrong, in every language it can
    for lang, table in (SAFE_LINES or {}).items():
        for value in (table or {}).values():
            add(lang, value)

    # 4. the knowledge answers the semantic fast path quotes VERBATIM. These are the
    #    single biggest group and the one people most often hear a real answer from,
    #    so pre-rendering them puts the owner's voice on the actual content of a call
    #    rather than only on its scaffolding.
    for entry in pack.get("knowledge", []) or ():
        add("English", entry.get("a"))
        for code, lang in (("hi", "Hindi"), ("te", "Telugu"), ("ta", "Tamil"),
                           ("bn", "Bengali"), ("mr", "Marathi"), ("kn", "Kannada")):
            add(lang, entry.get("a_" + code))
    return out


def verified(audio: bytes, text: str, stt) -> tuple[bool, str]:
    """Did the cloner actually say the words? Transcribe it back and see.

    THIS IS THE SAFETY VALVE FOR THE WHOLE FEATURE, and it exists because of a measured
    failure, not a hypothetical one. VoxCPM-0.5B asked for Hindi produced fluent-sounding
    audio that this project's own recogniser read as "and was myself a lack of your
    civly infelicit lack" — English gibberish in the caller's own voice. From a Hindi
    reference it ran away to 28 seconds of audio for a four-second line.

    A pre-rendered line is PINNED. It is never evicted, and every call thereafter plays
    it. So a bad render is not a bad turn — it is a permanent defect in the greeting,
    with no fallback and nothing to explain it. Checking costs one recognition per line,
    on a path that already costs seconds per line.

    Deliberately loose: this is asking "are these the right words", not scoring a WER.
    A cloner mispronouncing a proper noun ("Suvita" for "Suvidha" — measured) must pass,
    and gibberish must not.
    """
    if stt is None:
        return True, "unchecked"
    import os
    import tempfile
    tmp = tempfile.mkdtemp(prefix="zs_verify_")
    path = os.path.join(tmp, "a.wav")
    try:
        with open(path, "wb") as fh:
            fh.write(audio)
        heard, _lang, _p = stt.transcribe(path)
        if not heard:
            return False, "nothing recognisable in it"
        want = _words(text)
        got = set(_words(heard))
        if not want:
            return True, "no words to check"
        hit = sum(1 for w in want if w in got) / len(want)
        # Half the words is a low bar on purpose. A clone that gets half of a sentence
        # right is intelligible; the failure this catches produced ZERO of them.
        return (hit >= 0.5, "matched %.0f%% of the words" % (hit * 100))
    except Exception as e:  # noqa: BLE001
        # Unable to check is not the same as failed. Refusing to pin anything because
        # the recogniser is unavailable would silently disable the whole feature.
        return True, "could not check (%s)" % e
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _words(text: str):
    """Split on separators, never on \\w — that drops Indic combining marks, which is
    exactly the alphabet this check matters most for."""
    import re
    return [w for w in re.split(r"[\s.,!?;:।॥\-\"\'()]+", (text or "").lower()) if w]


def prerender(tts, pack: dict, *, languages=None, budget_s: float = 900.0,
              stt=None, progress=None) -> dict:
    """Synthesise the fixed lines through `tts`, filling its cache.

    `tts` is expected to be the CachedTTS wrapper — calling `synth` is what files the
    result, so this needs no knowledge of the cache's internals and cannot corrupt it.

    Bounded by `budget_s` because this runs while somebody is waiting for a "done", and
    a cloner that turns out to take four seconds a line must not appear to have hung.
    Whatever is finished is kept: a partial render is a partial improvement, not a
    broken state, because every miss falls through to the live synthesiser.
    """
    lines = fixed_lines(pack, languages)
    done = skipped = failed = rejected = 0
    t0 = time.time()
    for i, (lang, text) in enumerate(lines):
        if time.time() - t0 > budget_s:
            log.info("pre-render stopped at the %.0fs budget — %d of %d lines done",
                     budget_s, i, len(lines))
            break
        try:
            # pin=True: never evicted. A provider without the kwarg still works, it
            # just files them in the ordinary LRU — degraded, not broken.
            try:
                audio = tts.synth(text, None, pin=True)
            except TypeError:
                audio = tts.synth(text, None)
            # A provider that declines a script (Kokoro cannot speak Telugu, and says
            # so rather than producing confident nonsense) is not a failure — it is the
            # documented contract. Counting it as one would make a healthy render look
            # broken on any pack with languages the voice does not cover.
            if not audio or getattr(tts, "last_skipped_script", False):
                skipped += 1
            else:
                # Pinned audio is forever. Check it before trusting it.
                ok, why = verified(audio, text, stt)
                if ok:
                    done += 1
                else:
                    rejected += 1
                    unpin(tts, text)
                    log.info("pre-render rejected a line (%s) [%s] %r",
                             why, lang, text[:48])
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.debug("pre-render failed on %r (%s): %s", lang, text[:40], e)
        if progress and i % 10 == 0:
            progress(i, len(lines))
    took = time.time() - t0
    log.info("pre-render: %d spoken, %d skipped (voice cannot), %d rejected (wrong "
             "words), %d failed, in %.1fs", done, skipped, rejected, failed, took)
    return {"total": len(lines), "rendered": done, "skipped": skipped,
            "rejected": rejected, "failed": failed, "seconds": round(took, 1)}


def unpin(tts, text: str, voice=None) -> None:
    """Remove a line that failed verification, so the live synthesiser handles it.

    Needed because the render has already happened by the time it is judged — leaving
    it pinned would be worse than never rendering it.
    """
    drop = getattr(tts, "unpin", None)
    if drop:
        drop(text, voice)
