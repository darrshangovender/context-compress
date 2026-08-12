"""Core data types for the compression pipeline.

Everything crossing a module boundary is one of these dataclasses, so a
compression run can be serialised, logged, diffed, and replayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Where a chunk sits in the prompt. Drives protection policy."""

    SYSTEM = "system"          # never compressed
    INSTRUCTION = "instruction"  # never compressed
    CONTEXT = "context"        # the compressible bulk (RAG passages, docs)
    HISTORY = "history"        # conversation turns, compressible
    QUERY = "query"            # never compressed


#: Roles that must survive compression untouched. Dropping an instruction to
#: save tokens is the single worst failure mode of a compressor — it silently
#: changes the task rather than the evidence.
PROTECTED: frozenset[Role] = frozenset({Role.SYSTEM, Role.INSTRUCTION, Role.QUERY})


@dataclass
class Chunk:
    """One addressable unit of context.

    ``score`` is populated by a scorer; ``kept`` by a strategy. Keeping both on
    the chunk means a compression decision is fully explainable after the fact.
    """

    text: str
    role: Role = Role.CONTEXT
    source: str | None = None
    position: int = 0
    score: float | None = None
    kept: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def protected(self) -> bool:
        return self.role in PROTECTED


@dataclass
class CompressionResult:
    """Outcome of compressing a context bundle."""

    chunks: list[Chunk]
    original_tokens: int
    compressed_tokens: int
    strategy: str
    dropped: int = 0

    @property
    def text(self) -> str:
        """The surviving context, reassembled in original order."""
        kept = [c for c in self.chunks if c.kept]
        kept.sort(key=lambda c: c.position)
        return "\n\n".join(c.text for c in kept)

    @property
    def ratio(self) -> float:
        """Compressed / original. 0.25 means we kept a quarter of the tokens."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def saved_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return self.saved_tokens / self.original_tokens


@dataclass
class FidelityReport:
    """How much of the answer-bearing content survived compression.

    ``recall`` is the headline: of the facts needed to answer the question, what
    fraction is still present. A compressor that halves tokens but drops recall
    to 0.4 is worse than useless — it saves money by breaking the product.
    """

    recall: float
    precision: float
    kept_ratio: float
    n_required: int
    n_found: int
    missing: list[str] = field(default_factory=list)

    @property
    def f1(self) -> float:
        if self.recall + self.precision == 0:
            return 0.0
        return 2 * self.recall * self.precision / (self.recall + self.precision)
