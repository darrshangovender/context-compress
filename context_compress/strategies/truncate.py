"""Head/tail truncation — the baseline everyone actually ships.

Keep chunks from the front (and optionally the back) until the budget runs out.
No scoring, no query awareness. It exists to be beaten: any smarter strategy has
to justify its complexity against this, and on some workloads it genuinely wins
because it is free and preserves document order perfectly.
"""

from __future__ import annotations

from context_compress.strategies.base import Strategy
from context_compress.tokenizer import count_tokens
from context_compress.types import Chunk, CompressionResult


class TruncateStrategy(Strategy):
    """Fill the budget from the head, and optionally reserve room for the tail.

    ``tail_ratio`` splits the budget between head and tail. 0.0 is pure
    head-truncation; 0.3 reserves 30% for the most recent/last chunks, which is
    usually right for conversation history where the latest turns matter most.
    """

    name = "truncate"

    def __init__(self, tail_ratio: float = 0.0) -> None:
        if not 0.0 <= tail_ratio < 1.0:
            raise ValueError("tail_ratio must be in [0, 1)")
        self.tail_ratio = tail_ratio

    def compress(self, query: str, chunks: list[Chunk], budget: int) -> CompressionResult:
        remaining = self._check_budget(chunks, budget)
        compressible = [c for c in chunks if not c.protected]
        for c in compressible:
            c.kept = False

        tail_budget = int(remaining * self.tail_ratio)
        head_budget = remaining - tail_budget

        used_head = 0
        taken: set[int] = set()
        for i, c in enumerate(compressible):
            cost = count_tokens(c.text)
            if used_head + cost <= head_budget:
                c.kept = True
                taken.add(i)
                used_head += cost
            else:
                break

        if tail_budget > 0:
            used_tail = 0
            for i in range(len(compressible) - 1, -1, -1):
                if i in taken:
                    break
                cost = count_tokens(compressible[i].text)
                if used_tail + cost <= tail_budget:
                    compressible[i].kept = True
                    used_tail += cost
                else:
                    break

        return self._result(chunks)
