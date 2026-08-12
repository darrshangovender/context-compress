"""Top-K selection — keep the highest-scoring chunks that fit the budget.

This is the workhorse. Score every compressible chunk against the query, sort
descending, and greedily fill the budget. Selection order is by score, but the
surviving chunks are re-emitted in original document order — reordering context
to match relevance rank measurably hurts models on multi-hop questions where
the narrative sequence carries meaning.
"""

from __future__ import annotations

from context_compress.scoring import HybridScorer, Scorer
from context_compress.strategies.base import Strategy
from context_compress.tokenizer import count_tokens
from context_compress.types import Chunk, CompressionResult


class TopKStrategy(Strategy):
    """Greedy knapsack by relevance score.

    Parameters
    ----------
    scorer
        Anything implementing the ``Scorer`` protocol. Defaults to hybrid.
    min_score
        Chunks scoring below this are never kept, even if budget remains.
        Prevents padding the context with noise just because there's room —
        irrelevant context measurably degrades answer quality, so unused budget
        is better than filled budget.
    """

    name = "topk"

    def __init__(self, scorer: Scorer | None = None, min_score: float = 0.0) -> None:
        self.scorer = scorer or HybridScorer()
        self.min_score = min_score

    def compress(self, query: str, chunks: list[Chunk], budget: int) -> CompressionResult:
        remaining = self._check_budget(chunks, budget)
        compressible = [c for c in chunks if not c.protected]
        for c in compressible:
            c.kept = False

        if compressible:
            scores = self.scorer.score(query, compressible)
            for c, s in zip(compressible, scores):
                c.score = s

            ranked = sorted(compressible, key=lambda c: c.score or 0.0, reverse=True)
            used = 0
            for c in ranked:
                if (c.score or 0.0) < self.min_score:
                    continue
                cost = count_tokens(c.text)
                if used + cost <= remaining:
                    c.kept = True
                    used += cost

        return self._result(chunks)
