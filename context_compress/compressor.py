"""The Compressor — compose strategies into a pipeline.

Strategies compose, and order is load-bearing:

    Compressor([DedupeStrategy(), TopKStrategy()], budget=2000)

Dedupe first so the selector never spends budget ranking three copies of the
same passage; select second. Reversing that wastes budget on redundancy.

The pipeline threads chunks through each stage, letting each drop or rewrite
content, and reports the end-to-end result.
"""

from __future__ import annotations

from context_compress.strategies.base import BudgetTooSmall, Strategy
from context_compress.tokenizer import count_tokens
from context_compress.types import Chunk, CompressionResult, Role


class Compressor:
    """Run a chunk bundle through an ordered list of strategies."""

    def __init__(self, strategies: list[Strategy], budget: int) -> None:
        if not strategies:
            raise ValueError("need at least one strategy")
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.strategies = strategies
        self.budget = budget

    def compress(self, query: str, chunks: list[Chunk]) -> CompressionResult:
        original = sum(count_tokens(c.text) for c in chunks)
        working = chunks

        for strat in self.strategies:
            result = strat.compress(query, working, self.budget)
            # Survivors feed the next stage; dropped chunks stay dropped.
            working = [c for c in result.chunks if c.kept or c.protected]

        compressed = sum(count_tokens(c.text) for c in working if c.kept)
        return CompressionResult(
            chunks=working,
            original_tokens=original,
            compressed_tokens=compressed,
            strategy=" → ".join(s.name for s in self.strategies),
            dropped=len(chunks) - len([c for c in working if c.kept]),
        )

    # -- convenience ----------------------------------------------------

    @staticmethod
    def chunks_from_passages(
        passages: list[str],
        role: Role = Role.CONTEXT,
        source: str | None = None,
    ) -> list[Chunk]:
        """Build a chunk list from plain strings, preserving order."""
        return [
            Chunk(text=p, role=role, source=source, position=i)
            for i, p in enumerate(passages)
        ]


__all__ = ["Compressor", "BudgetTooSmall"]
