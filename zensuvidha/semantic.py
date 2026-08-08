"""Semantic search over a pack's knowledge — ranking, a fast path, and a rescue.

Three jobs, one index
---------------------
1. **RANK** the facts injected per turn. Measured on the shipping retriever: recall is
   fine but precision is poor — rank 1 is always the business NAME and rank 2 is often
   the ADDRESS, with roughly half the injected facts irrelevant to the question. That is
   precisely the documented failure mode of a 4B model here: *it answers the easy
   question (the address) and drops the one that was asked.* We were feeding it the
   distraction.

2. **ANSWER DIRECTLY** when the match is unambiguous. The pack already carries every
   answer written out in the caller's own language (`a_hi`, `a_te`, …). Speaking one is
   faster than generating it, cannot invent anything, and skips tokenizer inflation
   entirely — a 35-word Telugu reply costs 7.3s to generate at the measured 38.7 tok/s
   and 0s to quote.

3. **RESCUE** a turn the LLM could not answer — down, timed out, or blocked by the
   guard. Today that becomes "I don't have that detail". If the pack plainly contains
   the answer, saying it is better than refusing it.

Why char n-grams by default
---------------------------
The default backend is dependency-free: TF-IDF over character n-grams, cosine similarity.
That is a deliberate choice, not a placeholder.

  * It works ACROSS SCRIPTS with no model. Devanagari, Telugu, Tamil and Latin are all
    just character sequences.
  * It handles Indic MORPHOLOGY, which is the exact thing word matching fails at here —
    "సమయం" and "సమయంలో" share every n-gram but are different words, and the shipping
    retriever needed a hand-written prefix rule to cope.
  * It costs nothing: no torch, no 1.2GB download, no GPU, and it runs on the laptop
    this project exists to run on.

A neural backend (Qwen3-Embedding and friends) is supported and better at true
paraphrase — "how much to see a doctor" against "what is your consultation fee" shares
few characters. It is OPT-IN, because requiring a gigabyte of weights to answer "what
are your timings" is the wrong default for this project.
"""
from __future__ import annotations

import logging
import math
import re

log = logging.getLogger("zensuvidha.semantic")

# Character n-gram width. 3 is the usual choice for cross-lingual matching: long enough
# to carry a morpheme, short enough that an inflected form still overlaps its stem.
NGRAM = 3

# Answer directly, with no LLM, only when the match is unambiguous. Two conditions, and
# both matter: a high absolute score means "this really is that question", and a clear
# MARGIN over the runner-up means "and not the one next to it". A confident wrong answer
# on a phone call is far worse than a slow right one, so this is deliberately strict and
# it is better for it to decline than to guess.
# CALIBRATED, not guessed — the 0.55 speaker threshold was picked on synthetic audio
# and proved wrong by more than 2×, so these come from the measured distribution:
#
#   English, near-verbatim   0.63 – 1.00     "what are the timings", "what's the fee"
#   off-topic                0.15 – 0.41     "tell me a joke", "who won the election"
#   INDIC, near-verbatim     0.21 – 0.49     ← overlaps off-topic
#
# So 0.55 separates English cleanly and DECLINES for Indic, which is the correct answer
# rather than a shortcoming to paper over: an Indic question matches against the pack's
# `k_hi`/`k_te` KEYWORD lists, and a natural sentence does not look like a keyword list.
# Those callers take the normal path — exactly what they get today — and the honest
# irony is that latency hurts them most (Telugu inflates 6.2×). Closing that gap is what
# the neural backend is FOR, and the measurement above is the argument for enabling it.
DIRECT_SCORE = 0.55
DIRECT_MARGIN = 0.15

# Ranking is a softer job — a slightly-off fact merely wastes a prompt slot.
RANK_FLOOR = 0.08

_INDEX_KEY = "_semantic_index"      # cached on the pack, so it dies with the pack


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation that carries no meaning."""
    t = (text or "").lower()
    t = re.sub(r"[^\wऀ-෿؀-ۿ\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _grams(text: str) -> dict:
    """Character n-grams with counts. Word boundaries are kept as spaces so a gram can
    straddle two words, which is what makes short phrases match at all."""
    t = _norm(text)
    if not t:
        return {}
    if len(t) <= NGRAM:
        return {t: 1}
    out: dict = {}
    for i in range(len(t) - NGRAM + 1):
        g = t[i:i + NGRAM]
        out[g] = out.get(g, 0) + 1
    return out


class LexicalBackend:
    """TF-IDF over character n-grams. No dependencies, works in every script."""

    name = "char-ngram"

    def __init__(self, docs: list[str]):
        self._vecs = [_grams(d) for d in docs]
        # document frequency, for IDF — without it, grams that appear in every entry
        # ("the", " an") dominate the similarity and everything matches everything.
        df: dict = {}
        for v in self._vecs:
            for g in v:
                df[g] = df.get(g, 0) + 1
        n = max(1, len(self._vecs))
        self._idf = {g: math.log(1.0 + n / c) for g, c in df.items()}
        self._norms = [self._norm_of(v) for v in self._vecs]

    def _norm_of(self, v: dict) -> float:
        return math.sqrt(sum((c * self._idf.get(g, 0.0)) ** 2 for g, c in v.items())) or 1.0

    def scores(self, query: str) -> list[float]:
        q = _grams(query)
        if not q:
            return [0.0] * len(self._vecs)
        qn = self._norm_of(q)
        out = []
        for v, vn in zip(self._vecs, self._norms):
            # iterate the SHORTER side; a query is usually far shorter than an entry
            small, big = (q, v) if len(q) <= len(v) else (v, q)
            dot = sum(c * big.get(g, 0) * (self._idf.get(g, 0.0) ** 2)
                      for g, c in small.items() if g in big)
            out.append(dot / (qn * vn))
        return out


class NeuralBackend:
    """Sentence embeddings — better at true paraphrase, at the cost of a download.

    Opt-in. "How much to see a doctor" and "what is your consultation fee" share almost
    no characters, and that is the case n-grams genuinely cannot reach.
    """

    def __init__(self, docs: list[str], model_name: str):
        from sentence_transformers import SentenceTransformer   # optional dependency
        self.name = f"neural:{model_name}"
        self._m = SentenceTransformer(model_name)
        # Encoded ONCE per pack. Only the query is encoded per turn, which is what keeps
        # this affordable — the knowledge base does not change mid-call.
        self._doc = self._m.encode(docs, normalize_embeddings=True,
                                   convert_to_numpy=True)

    def scores(self, query: str) -> list[float]:
        import numpy as np
        q = self._m.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return list(np.asarray(self._doc @ q[0], dtype="float64"))


class KnowledgeIndex:
    """Searchable view of one pack's knowledge, in every language it carries."""

    def __init__(self, pack: dict, model_name: str | None = None):
        self.entries: list[dict] = ((pack.get("knowledge") or [])
                                    + (pack.get("common") or []))
        # What a caller might SAY for each entry: the English question, its tags, and the
        # native keyword lists the pack already maintains (`k_hi`, `k_te`, …). Indexing
        # the ANSWER too would match on shared boilerplate rather than on the question.
        # Each entry becomes SEVERAL short documents — the English question, the tags,
        # and each native keyword list — scored separately, best one wins.
        #
        # Concatenating them into one long document diluted every score: a five-word
        # query against a document carrying the question in three languages plus its
        # tags scored 0.36 even when it WAS that question verbatim, because cosine
        # punishes the length mismatch. A Hindi caller was also competing against the
        # English text sitting in the same vector. Split, "What are your timings?"
        # matches the question itself and nothing else has to be explained away.
        docs, self._owner = [], []
        for i, e in enumerate(self.entries):
            variants = [str(e.get("q", "")), " ".join(e.get("tags") or [])]
            variants += [str(v) for k, v in e.items() if k.startswith("k_")]
            for v in variants:
                if v.strip():
                    docs.append(v)
                    self._owner.append(i)
        self.backend = None
        if docs:
            if model_name:
                try:
                    self.backend = NeuralBackend(docs, model_name)
                except Exception as e:  # noqa: BLE001
                    log.warning("semantic: neural backend unavailable (%s) — "
                                "falling back to char n-grams", e)
            if self.backend is None:
                self.backend = LexicalBackend(docs)
        log.info("semantic: indexed %d knowledge entries (%s)",
                 len(self.entries), getattr(self.backend, "name", "none"))

    def search(self, query: str, k: int = 4) -> list[tuple[float, dict]]:
        """The k best ENTRIES, best first, above RANK_FLOOR.

        An entry scores as its best-matching variant: a Hindi caller matching `k_hi` is
        just as much a hit as an English caller matching `q`, and neither should be
        averaged down by the other.
        """
        if not self.entries or self.backend is None or not (query or "").strip():
            return []
        per_doc = self.backend.scores(query)
        best: dict = {}
        for score, owner in zip(per_doc, self._owner):
            if score > best.get(owner, -1.0):
                best[owner] = score
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return [(sc, self.entries[i]) for i, sc in ranked[:k] if sc >= RANK_FLOOR]

    def direct(self, query: str) -> tuple[dict | None, float, str]:
        """The single entry that answers this outright, if one plainly does.

        Returns (entry, score, why). Declines unless the best match is both strong in
        absolute terms and clearly ahead of the next one — "is there parking" and "where
        are you" are neighbours, and answering the wrong one confidently is worse than
        taking the slower path.
        """
        hits = self.search(query, k=2)
        if not hits:
            return None, 0.0, ""
        best, entry = hits[0]
        runner = hits[1][0] if len(hits) > 1 else 0.0
        if best < DIRECT_SCORE:
            return None, best, f"best match {best:.2f} < {DIRECT_SCORE}"
        if best - runner < DIRECT_MARGIN:
            return None, best, f"only {best - runner:.2f} ahead of the next entry"
        return entry, best, f"matched {entry.get('q', '?')!r} at {best:.2f}"


def index_for(pack: dict, model_name: str | None = None) -> KnowledgeIndex | None:
    """The index for this pack, built once and cached on it.

    Cached on the dict rather than by id(): CPython reuses addresses after a free, so an
    id()-keyed cache can hand one pack another pack's index — and here that would mean
    answering a clinic caller from the salon's knowledge base.
    """
    try:
        got = pack.get(_INDEX_KEY)
        if got is not None:
            return got
        idx = KnowledgeIndex(pack, model_name)
    except Exception as e:  # noqa: BLE001
        # The cache READ was outside the guard, so a pack that raises on .get()
        # propagated straight out of a function whose whole contract is "returns None
        # if it cannot help". Everything optional in this pipeline fails open; this is
        # not the file to make an exception in.
        log.warning("semantic: could not index this pack (%s) — ranking unchanged", e)
        return None
    try:
        pack[_INDEX_KEY] = idx
    except Exception:  # noqa: BLE001
        pass                      # a read-only mapping: rebuild each time, still correct
    return idx


def answer_in(entry: dict, lang_code: str | None) -> str:
    """The entry's answer in the caller's language, or English.

    The pack carries these written out, which is the whole point: quoting one costs
    nothing and cannot invent, where GENERATING the same sentence in Telugu costs 6.2×
    the tokens of the English it was translated from.
    """
    if lang_code and lang_code != "en":
        native = entry.get(f"a_{lang_code}")
        if native:
            return str(native)
    return str(entry.get("a", "") or "")
