"""Relevance scoring — how much is a chunk worth keeping, given the query?

Three scorers, no heavy dependencies:

- ``LexicalScorer``   — BM25-style term overlap. Zero cost, strong on rare
  strings (error codes, part numbers, proper nouns) that embeddings blur.
- ``PositionScorer``  — encodes the lost-in-the-middle effect: models attend
  most to the head and tail of a long context, so edge chunks carry more
  effective weight than their content alone suggests.
- ``HybridScorer``    — weighted blend, which is what you actually want.

There is deliberately no embedding scorer in the core package: it would drag in
a model download and make the whole thing un-runnable offline. Plug one in via
the ``Scorer`` protocol if you have embeddings available.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from context_compress.types import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words too common to carry retrieval signal. Kept short on purpose — an
#: aggressive stoplist hurts on technical text where "not", "all", "between"
#: change meaning materially.
_STOP = frozenset(
    "a an the is are was were be been being of to in on at for with by from as "
    "and or but if then than that this these those it its i you he she they we".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@runtime_checkable
class Scorer(Protocol):
    """Assign a [0, 1] relevance score to each chunk given a query."""

    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class LexicalScorer:
    """BM25 over the chunk set.

    BM25 rather than raw TF-IDF because term saturation matters here: a chunk
    that repeats the query term twenty times is not twenty times more relevant,
    and without saturation the compressor keeps keyword-stuffed boilerplate over
    the passage that actually answers the question.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return [0.0] * len(chunks)

        docs = [tokenize(c.text) for c in chunks]
        lengths = [len(d) for d in docs]
        avg_len = sum(lengths) / len(lengths) if lengths else 1.0
        n_docs = len(docs)

        # Document frequency per query term.
        df = Counter()
        for d in docs:
            for term in set(q_terms):
                if term in d:
                    df[term] += 1

        raw: list[float] = []
        for doc, dl in zip(docs, lengths):
            tf = Counter(doc)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                # +1 smoothing keeps idf positive even for terms in every doc.
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(avg_len, 1e-9))
                s += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
            raw.append(s)
        return _normalise(raw)


class PositionScorer:
    """Score by position, modelling the lost-in-the-middle effect.

    Liu et al. (2023) showed retrieval accuracy is U-shaped in position: content
    at the start and end of a long context is recalled far more reliably than
    content buried in the middle. When we must drop chunks, dropping from the
    weak middle costs less real-world accuracy than dropping from the edges.
    """

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        n = len(chunks)
        if n == 0:
            return []
        if n == 1:
            return [1.0]
        out = []
        for i in range(n):
            rel = i / (n - 1)              # 0 at head, 1 at tail
            edge = abs(rel - 0.5) * 2      # 1 at edges, 0 dead centre
            out.append(edge)
        return out


class HybridScorer:
    """Weighted blend of lexical relevance and positional weight.

    Lexical dominates by default (0.8): position is a tie-breaker, not a
    primary signal. Weighting position too highly keeps irrelevant material
    purely for sitting near an edge.
    """

    def __init__(self, lexical_weight: float = 0.8) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be in [0, 1]")
        self.lexical_weight = lexical_weight
        self._lex = LexicalScorer()
        self._pos = PositionScorer()

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        lex = self._lex.score(query, chunks)
        pos = self._pos.score(query, chunks)
        w = self.lexical_weight
        return [w * l + (1 - w) * p for l, p in zip(lex, pos)]


def _normalise(values: list[float]) -> list[float]:
    """Min-max to [0, 1]. All-equal input maps to all-1.0, not all-0.0 —
    otherwise a set of uniformly-relevant chunks would all score zero and be
    dropped, which is the opposite of correct."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 if hi > 0 else 0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
