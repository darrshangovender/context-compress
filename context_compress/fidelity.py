"""Fidelity evaluation — did compression destroy the answer?

Compression ratio alone is a vanity metric. Any compressor can hit 90% savings
by deleting everything. The number that matters is whether the facts required to
answer the question survived.

``FidelityEvaluator`` takes a set of required facts (short strings that must
remain findable) and measures recall against the compressed context. This is what
turns the benchmark from "how small did it get" into "how small did it get
*without breaking*", which is the only comparison worth publishing.
"""

from __future__ import annotations

import re

from context_compress.types import CompressionResult, FidelityReport


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for tolerant matching."""
    return re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\s+", " ", text.lower())).strip()


class FidelityEvaluator:
    """Measure survival of required facts through compression.

    Parameters
    ----------
    fuzzy
        When True, a fact counts as present if all of its content words appear
        in the compressed text (order-independent). Handles the sentence-prune
        case where a fact's words survive but the exact phrasing is broken up.
        When False, requires exact substring containment.
    """

    def __init__(self, fuzzy: bool = True) -> None:
        self.fuzzy = fuzzy

    def evaluate(
        self,
        result: CompressionResult,
        required_facts: list[str],
        original_text: str | None = None,
    ) -> FidelityReport:
        compressed = _normalise(result.text)
        found, missing = [], []

        for fact in required_facts:
            if self._present(fact, compressed):
                found.append(fact)
            else:
                missing.append(fact)

        n_req = len(required_facts)
        recall = len(found) / n_req if n_req else 1.0

        # Precision here is "signal density": what fraction of the surviving
        # tokens is doing useful work. Approximated as required-fact tokens over
        # total kept tokens — a compressor that keeps everything scores perfect
        # recall but poor precision, which is exactly the trade we want visible.
        fact_tokens = sum(len(f.split()) for f in found)
        kept_tokens = max(1, result.compressed_tokens)
        precision = min(1.0, fact_tokens / kept_tokens)

        return FidelityReport(
            recall=recall,
            precision=precision,
            kept_ratio=result.ratio,
            n_required=n_req,
            n_found=len(found),
            missing=missing,
        )

    def _present(self, fact: str, compressed: str) -> bool:
        norm_fact = _normalise(fact)
        if not norm_fact:
            return True
        if norm_fact in compressed:
            return True
        if not self.fuzzy:
            return False
        words = [w for w in norm_fact.split() if len(w) > 2]
        if not words:
            return norm_fact in compressed
        return all(w in compressed for w in words)
