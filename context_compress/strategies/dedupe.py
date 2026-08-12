"""Near-duplicate removal.

RAG retrieval routinely returns the same fact three times: overlapping chunks
from a sliding window, the same paragraph in a doc and its changelog, a FAQ
answer duplicated across pages. Every copy after the first is pure token waste
that also crowds out genuinely distinct evidence.

Detection is MinHash-style Jaccard over token shingles — no embedding model, no
network, deterministic. Exact-duplicate detection is a hash lookup; near-duplicate
uses a similarity threshold.
"""

from __future__ import annotations

import hashlib

from context_compress.scoring import tokenize
from context_compress.strategies.base import Strategy
from context_compress.types import Chunk, CompressionResult


def shingles(text: str, n: int = 3) -> set[int]:
    """Hashed n-gram shingles of the token stream.

    Hashing to ints keeps the sets small and comparison fast; collisions at
    64-bit truncation are irrelevant at document scale.
    """
    toks = tokenize(text)
    if len(toks) < n:
        return {_h(" ".join(toks))} if toks else set()
    return {_h(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)}


def _h(s: str) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class DedupeStrategy(Strategy):
    """Drop chunks that substantially repeat an earlier kept chunk.

    Note this is budget-aware but not budget-driven: it removes redundancy and
    stops. If the result still exceeds budget, compose it with another strategy
    (the Compressor pipeline does exactly this) — dedupe first, then select from
    what remains. Running them the other way round wastes budget on duplicates.
    """

    name = "dedupe"

    def __init__(self, threshold: float = 0.8, shingle_size: int = 3) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.shingle_size = shingle_size

    def compress(self, query: str, chunks: list[Chunk], budget: int) -> CompressionResult:
        self._check_budget(chunks, budget)
        compressible = [c for c in chunks if not c.protected]

        kept_sigs: list[set[int]] = []
        for c in compressible:
            sig = shingles(c.text, self.shingle_size)
            if any(jaccard(sig, prev) >= self.threshold for prev in kept_sigs):
                c.kept = False
                c.meta["dropped_reason"] = "near_duplicate"
            else:
                c.kept = True
                kept_sigs.append(sig)

        return self._result(chunks)
