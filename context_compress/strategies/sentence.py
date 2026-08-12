"""Sentence-level pruning — compress *within* chunks, not just between them.

Chunk-level selection is all-or-nothing: a 400-token passage containing one
critical sentence costs 400 tokens or nothing. This strategy explodes each chunk
into sentences, scores them individually, and rebuilds the chunk from the
sentences that earn their place.

That granularity is where the real savings live on RAG workloads — retrieved
passages are typically 10-20% signal and 80-90% surrounding prose.

The trade-off is coherence: prune too hard and you get disjointed fragments the
model has to stitch together. ``min_sentences`` guards against shredding a
passage into single clauses.
"""

from __future__ import annotations

from context_compress.scoring import LexicalScorer, Scorer
from context_compress.strategies.base import Strategy
from context_compress.tokenizer import count_tokens, split_sentences
from context_compress.types import Chunk, CompressionResult


class SentencePruneStrategy(Strategy):
    """Keep the highest-value sentences across all compressible chunks.

    Parameters
    ----------
    min_sentences
        Always retain at least this many sentences from any surviving chunk,
        taken in document order. Preserves enough local context that the kept
        material still reads as prose.
    """

    name = "sentence_prune"

    def __init__(self, scorer: Scorer | None = None, min_sentences: int = 1) -> None:
        self.scorer = scorer or LexicalScorer()
        self.min_sentences = max(0, min_sentences)

    def compress(self, query: str, chunks: list[Chunk], budget: int) -> CompressionResult:
        remaining = self._check_budget(chunks, budget)
        compressible = [c for c in chunks if not c.protected]

        # Explode into (owning chunk index, sentence index, text).
        units: list[tuple[int, int, str]] = []
        for ci, c in enumerate(compressible):
            for si, sent in enumerate(split_sentences(c.text)):
                units.append((ci, si, sent))

        if not units:
            return self._result(chunks)

        # Score sentences as if each were its own chunk.
        as_chunks = [Chunk(text=t, position=i) for i, (_, _, t) in enumerate(units)]
        scores = self.scorer.score(query, as_chunks)

        # Guarantee the first `min_sentences` of each chunk survive by pinning
        # their score above anything the scorer can produce.
        for idx, (ci, si, _) in enumerate(units):
            if si < self.min_sentences:
                scores[idx] = float("inf")

        order = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)

        used = 0
        keep: set[int] = set()
        for idx in order:
            cost = count_tokens(units[idx][2])
            if used + cost <= remaining:
                keep.add(idx)
                used += cost

        # Rebuild each chunk from its surviving sentences, in original order.
        rebuilt: dict[int, list[str]] = {}
        for idx in sorted(keep, key=lambda i: (units[i][0], units[i][1])):
            ci, _, text = units[idx]
            rebuilt.setdefault(ci, []).append(text)

        for ci, c in enumerate(compressible):
            kept_sents = rebuilt.get(ci)
            if kept_sents:
                c.text = " ".join(kept_sents)
                c.kept = True
            else:
                c.kept = False

        return self._result(chunks)
