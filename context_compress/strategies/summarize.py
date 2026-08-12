"""Extractive summarisation — condense low-value chunks rather than dropping them.

The other strategies are binary: a chunk survives whole or vanishes. That's a
blunt instrument for material which is *somewhat* relevant — background that
provides framing but doesn't answer the question directly.

This strategy keeps high-scoring chunks verbatim and replaces marginal ones with
an extractive summary (their own highest-signal sentences). Extractive, not
abstractive, and deliberately so: an LLM-generated abstract would cost a model
call per chunk and introduce a hallucination surface inside your context — the
one place you most need ground truth intact.
"""

from __future__ import annotations

from context_compress.scoring import HybridScorer, LexicalScorer, Scorer
from context_compress.strategies.base import Strategy
from context_compress.tokenizer import count_tokens, split_sentences
from context_compress.types import Chunk, CompressionResult


class SummarizeStrategy(Strategy):
    """Tiered compression: keep, condense, or drop based on relevance.

    Parameters
    ----------
    keep_threshold
        Score at or above which a chunk is kept verbatim.
    drop_threshold
        Score below which a chunk is dropped entirely. Between the two, chunks
        are condensed to their top sentences.
    condense_ratio
        Fraction of a condensed chunk's sentences to retain.
    """

    name = "summarize"

    def __init__(
        self,
        scorer: Scorer | None = None,
        keep_threshold: float = 0.7,
        drop_threshold: float = 0.2,
        condense_ratio: float = 0.4,
    ) -> None:
        if not 0.0 <= drop_threshold <= keep_threshold <= 1.0:
            raise ValueError("need 0 <= drop_threshold <= keep_threshold <= 1")
        if not 0.0 < condense_ratio <= 1.0:
            raise ValueError("condense_ratio must be in (0, 1]")
        self.scorer = scorer or HybridScorer()
        self.keep_threshold = keep_threshold
        self.drop_threshold = drop_threshold
        self.condense_ratio = condense_ratio
        self._sent_scorer = LexicalScorer()

    def compress(self, query: str, chunks: list[Chunk], budget: int) -> CompressionResult:
        remaining = self._check_budget(chunks, budget)
        compressible = [c for c in chunks if not c.protected]
        if not compressible:
            return self._result(chunks)

        scores = self.scorer.score(query, compressible)
        for c, s in zip(compressible, scores):
            c.score = s

        # Tier each chunk, condensing the middle band.
        staged: list[tuple[Chunk, str]] = []
        for c in compressible:
            s = c.score or 0.0
            if s >= self.keep_threshold:
                staged.append((c, c.text))
                c.meta["tier"] = "verbatim"
            elif s < self.drop_threshold:
                c.kept = False
                c.meta["tier"] = "dropped"
            else:
                staged.append((c, self._condense(query, c.text)))
                c.meta["tier"] = "condensed"

        # Apply in score order until the budget is exhausted.
        staged.sort(key=lambda pair: pair[0].score or 0.0, reverse=True)
        used = 0
        for c, text in staged:
            cost = count_tokens(text)
            if used + cost <= remaining:
                c.text = text
                c.kept = True
                used += cost
            else:
                c.kept = False
                c.meta["tier"] = "dropped_budget"

        return self._result(chunks)

    def _condense(self, query: str, text: str) -> str:
        """Keep the top-scoring fraction of sentences, in original order."""
        sents = split_sentences(text)
        if len(sents) <= 1:
            return text
        n_keep = max(1, int(round(len(sents) * self.condense_ratio)))
        as_chunks = [Chunk(text=s, position=i) for i, s in enumerate(sents)]
        s_scores = self._sent_scorer.score(query, as_chunks)
        top = sorted(range(len(sents)), key=lambda i: s_scores[i], reverse=True)[:n_keep]
        return " ".join(sents[i] for i in sorted(top))
