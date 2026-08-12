"""Strategy interface.

A strategy takes a list of chunks plus a token budget and decides what survives.
Every strategy honours two invariants:

1. **Protected roles are never dropped or altered.** System prompts,
   instructions and the query itself pass through untouched.
2. **The result never exceeds the budget** (unless the protected content alone
   already exceeds it, which is a caller error the strategy surfaces rather
   than silently mangling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_compress.tokenizer import count_tokens
from context_compress.types import Chunk, CompressionResult


class BudgetTooSmall(Exception):
    """Protected content alone exceeds the requested budget."""


class Strategy(ABC):
    """Compress a chunk list to fit a token budget."""

    name: str = "strategy"

    @abstractmethod
    def compress(self, query: str, chunks: list[Chunk], budget: int) -> CompressionResult:
        """Return a CompressionResult whose kept chunks fit within ``budget``."""

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def _protected_cost(chunks: list[Chunk], model: str = "gpt-4o") -> int:
        return sum(count_tokens(c.text, model) for c in chunks if c.protected)

    def _check_budget(self, chunks: list[Chunk], budget: int) -> int:
        """Return the budget left for compressible content, after protected."""
        floor = self._protected_cost(chunks)
        if floor > budget:
            raise BudgetTooSmall(
                f"protected content needs {floor} tokens but budget is {budget}"
            )
        return budget - floor

    def _result(self, chunks: list[Chunk], budget_used_on: str = "") -> CompressionResult:
        original = sum(count_tokens(c.text) for c in chunks)
        compressed = sum(count_tokens(c.text) for c in chunks if c.kept)
        return CompressionResult(
            chunks=chunks,
            original_tokens=original,
            compressed_tokens=compressed,
            strategy=budget_used_on or self.name,
            dropped=sum(1 for c in chunks if not c.kept),
        )
