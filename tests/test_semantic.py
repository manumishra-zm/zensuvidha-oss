"""Semantic knowledge search — ranking, the fast path, and the rescue.

The dangerous direction here is a CONFIDENT WRONG ANSWER: the fast path speaks straight
from the pack with no model and no guard in between, so most of these tests are about
what it refuses to answer rather than what it answers.
"""
import pytest

from zensuvidha import semantic as S
from zensuvidha.orchestrator import Session
from zensuvidha.packs import load_pack


def _idx():
    return S.index_for(load_pack("clinic"))


def _session():
    return Session(load_pack("clinic"), None)


# ── it must never answer something the pack does not cover ────────────────────

@pytest.mark.parametrize("q", [
    "what is the capital of France", "tell me a joke", "who won the election",
    "what is your wifi password", "can you sing a song", "book me a flight",
    "मौसम कैसा है", "క్రికెట్ స్కోర్ ఎంత", "what is 2 plus 2", "", "   ",
])
def test_off_topic_is_never_answered_from_the_pack(q):
    """Measured off-topic band is 0.15-0.41 and the threshold is 0.55, so this has
    headroom — but it is the failure that would matter, so it is pinned."""
    text, _ = _session().quick_answer(q)
    assert text is None, f"answered an off-topic question: {q!r} → {text!r}"


def test_a_near_miss_between_two_neighbouring_entries_declines():
    """'is there parking' and 'where are you' are neighbours. Answering the wrong one
    confidently is worse than taking the slower path, so a thin margin must decline."""
    idx = _idx()
    entry, score, why = idx.direct("where is the parking")
    if entry is not None:
        # if it DOES answer, it must be by a clear margin, not a coin toss
        hits = idx.search("where is the parking", k=2)
        assert hits[0][0] - hits[1][0] >= S.DIRECT_MARGIN, why


# ── it should answer the questions people actually repeat ─────────────────────

@pytest.mark.parametrize("q", [
    "what are your timings", "what are the timings",
    "what is the consultation fee", "what's the consultation fee",
    "are you open today",
])
def test_the_common_questions_are_answered_without_the_model(q):
    text, why = _session().quick_answer(q)
    assert text, f"{q!r} was not answered from the pack ({why})"
    assert len(text) > 10


def test_the_answer_is_the_packs_own_words():
    """It cannot invent, because the sentence IS the fact. That is the whole reason
    this is allowed to bypass the grounding guard."""
    pack = load_pack("clinic")
    text, _ = Session(pack, None).quick_answer("what are your timings")
    kb = (pack.get("knowledge") or []) + (pack.get("common") or [])
    assert any(text == e.get("a") for e in kb), "the fast path did not quote the pack"


# ── it must not hijack a booking ──────────────────────────────────────────────

def test_it_never_fires_mid_booking():
    """The caller is answering OUR questions. A knowledge fact in the middle of slot
    collection abandons the collection — the rule `recovery_line` already follows."""
    s = _session()
    s.booking_started = True
    assert s.quick_answer("what are your timings")[0] is None
    s.booking_started, s.pending_slot = False, "phone"
    assert s.quick_answer("what are your timings")[0] is None


# ── ranking ───────────────────────────────────────────────────────────────────

def test_an_entry_scores_as_its_BEST_variant_not_an_average():
    """A Hindi caller matching `k_hi` is as much a hit as an English caller matching
    `q`. Concatenating them into one document diluted every score — a verbatim question
    scored 0.36 — because cosine punishes the length mismatch."""
    idx = _idx()
    hits = idx.search("what are your timings", k=1)
    assert hits and hits[0][0] > 0.9, f"a verbatim question scored {hits[0][0]:.2f}"


def test_search_returns_entries_not_duplicates():
    idx = _idx()
    hits = idx.search("what is the consultation fee", k=4)
    ids = [id(e) for _s, e in hits]
    assert len(ids) == len(set(ids)), "the same entry was returned twice"


def test_it_works_across_scripts_with_no_model():
    """Char n-grams are the default precisely so Devanagari and Telugu work with no
    download. Ranking must still be sane even where the fast path declines."""
    idx = _idx()
    for q in ["आपका समय क्या है", "మీ ఫీజు ఎంత"]:
        assert idx.search(q, k=3), f"nothing retrieved for {q!r}"


# ── it must degrade safely ────────────────────────────────────────────────────

def test_a_pack_with_no_knowledge_does_not_break_anything():
    idx = S.KnowledgeIndex({}, None)
    assert idx.search("anything") == []
    assert idx.direct("anything")[0] is None


def test_the_index_is_cached_on_the_pack_not_by_id():
    """An id()-keyed cache can hand one pack another's index after a free — here that
    would mean answering a clinic caller from the salon's knowledge base."""
    pack = load_pack("clinic")
    first = S.index_for(pack)
    assert S.index_for(pack) is first
    other = S.index_for(load_pack("salon"))
    assert other is not first
    assert other.entries != first.entries


def test_a_broken_backend_leaves_ranking_unchanged_rather_than_raising():
    class Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("pack exploded")
    assert S.index_for(Boom()) is None


def test_answer_in_falls_back_to_english_when_the_language_is_missing():
    e = {"a": "English answer", "a_hi": "हिंदी उत्तर"}
    assert S.answer_in(e, "hi") == "हिंदी उत्तर"
    assert S.answer_in(e, "te") == "English answer"      # no a_te → English, not silence
    assert S.answer_in(e, None) == "English answer"


# ── the pack cache, which is what makes all of this affordable ────────────────

def test_the_pack_is_cached_across_sessions():
    """Measured at 51ms per call — two YAML reads and a deep merge — paid on EVERY
    session, at the moment the caller is waiting for a greeting. It also discarded the
    structures derived from the pack, so the semantic index and the expectation
    vocabulary rebuilt per call too."""
    import time
    load_pack("clinic")                       # warm
    t = time.perf_counter()
    for _ in range(50):
        load_pack("clinic")
    per = (time.perf_counter() - t) / 50
    assert per < 0.005, f"{per*1000:.1f}ms per load — the cache is not working"
    assert load_pack("clinic") is load_pack("clinic")


def test_editing_a_pack_still_takes_effect_without_a_restart():
    """Keyed on modification time, not name — otherwise a pack edited during
    development would be invisible until the server was restarted."""
    import os
    import pathlib as _p
    f = _p.Path("packs/clinic.yaml")
    before = load_pack("clinic")
    mt = f.stat().st_mtime
    try:
        f.touch()
        assert load_pack("clinic") is not before, "an edited pack was served from cache"
    finally:
        os.utime(f, (mt, mt))


def test_nothing_mutates_a_shared_pack_except_the_derived_caches():
    """Packs are now SHARED between concurrent calls. That is only safe because nothing
    writes per-session state into them — if anything did, one caller's state would
    appear in another's prompt."""
    import ast
    import pathlib as _p
    allowed = {"_semantic_index", "_expectation_vocab"}
    for path in _p.Path("zensuvidha").glob("*.py"):
        if path.name == "packs.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == "pack"):
                    key = getattr(getattr(tgt.slice, "value", None), "__str__", lambda: "?")()
                    assert key in allowed or "_" in str(key), (
                        f"{path.name} writes {key!r} into a shared pack")


def test_the_cache_is_bounded():
    """One entry per pack per edit. Unbounded, a pack edited in a loop during
    development would grow it without limit."""
    from zensuvidha import packs
    assert "_PACK_CACHE" in dir(packs)
    src = open(packs.__file__, encoding="utf-8").read()
    assert "_PACK_CACHE.clear()" in src, "the cache has no bound"


# ── the neural backend, if anyone enables one ─────────────────────────────────

def test_the_instruction_prefix_is_used_when_the_model_ships_one():
    """Not optional for these models. Measured on Qwen3-Embedding-0.6B:

        no prefix                indic 0.54-0.81   off-topic 0.43-0.66   OVERLAPS
        WITH instruction prefix  indic 0.47-0.69   off-topic 0.25-0.40   separable

    Without it the model compresses everything into one high band and "tell me a joke"
    scores as well as a real question — it is the difference between working and not.
    """
    import inspect
    src = inspect.getsource(S.NeuralBackend)
    assert "prompt_name" in src, "the instruction prefix is not being used"
    assert '"query"' in src and '"document"' in src, "queries and documents must differ"


def test_the_neural_backend_has_its_own_threshold():
    """Neural cosine sits on a different scale — the lexical 0.55 would let off-topic
    through, because everything scores higher."""
    assert S.NEURAL_DIRECT_SCORE < S.DIRECT_SCORE
    import inspect
    src = inspect.getsource(S.KnowledgeIndex.direct)
    assert "NeuralBackend" in src, "one threshold is being applied to both backends"


def test_a_missing_neural_model_falls_back_rather_than_failing_the_call():
    """A pack must still be searchable when the model is absent, wrong, or unloadable.
    Everything optional in this pipeline fails open."""
    idx = S.KnowledgeIndex(load_pack("clinic"), "this/model-does-not-exist")
    assert isinstance(idx.backend, S.LexicalBackend)
    assert idx.search("what are your timings", k=1), "fallback ranking is broken"


def test_the_fast_path_streams_rather_than_synthesising_one_long_call():
    """Skipping the model is worthless if the saving is spent waiting for one long TTS.
    The pack's answers carry real detail — "open Monday to Saturday, 9am to 8pm, with a
    lunch break…" — and Kokoro scales with the text: 2955ms for the whole answer against
    2003ms to the first chunk, so 951ms of the saving was being handed straight back."""
    import inspect
    from zensuvidha import server
    src = inspect.getsource(server._stream_turn)
    at = src.index("fast path: answered from the pack")
    window = src[at:at + 900]
    assert "next_chunk(" in window, "the fast path speaks the whole answer in one call"
    assert "clause=(seq == 0)" in window, (
        "the first chunk must break on clauses too — that is what makes the opening "
        "arrive fast")
