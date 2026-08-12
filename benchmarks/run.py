"""Reproducible benchmark: compression ratio vs answer fidelity.

The headline question this answers: **how few tokens can you get away with
before the answer stops surviving?**

Compares every strategy (and the recommended pipeline) at a fixed token budget,
reporting tokens saved alongside fact recall. A strategy that saves more but
recalls less is not better — the table makes that trade explicit.

Run:
    python benchmarks/run.py

Fully offline. Writes results.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from context_compress import Compressor
from context_compress.fidelity import FidelityEvaluator
from context_compress.strategies import (
    DedupeStrategy,
    SentencePruneStrategy,
    SummarizeStrategy,
    TopKStrategy,
    TruncateStrategy,
)

from benchmarks.dataset import make_dataset

BUDGET = 120  # tokens of context — deliberately tight to force real choices


def _pipelines():
    return {
        "no-compression": None,
        "truncate": [TruncateStrategy()],
        "truncate+tail": [TruncateStrategy(tail_ratio=0.3)],
        "topk": [TopKStrategy()],
        # min_sentences=1 pins every chunk's lead sentence, which on RAG
        # workloads burns budget on framing prose. Both variants are shown so
        # the cost of that default is visible rather than hidden.
        "sentence-prune(min=1)": [SentencePruneStrategy(min_sentences=1)],
        "sentence-prune(min=0)": [SentencePruneStrategy(min_sentences=0)],
        "summarize": [SummarizeStrategy()],
        "dedupe→topk": [DedupeStrategy(), TopKStrategy()],
        "dedupe→sentence": [DedupeStrategy(), SentencePruneStrategy(min_sentences=0)],
    }


def main() -> int:
    items = make_dataset(n=60, seed=42)
    evaluator = FidelityEvaluator()
    rows: dict[str, dict] = {}

    for name, strategies in _pipelines().items():
        ratios: list[float] = []
        recalls: list[float] = []
        orig_tokens: list[int] = []
        comp_tokens: list[int] = []

        for item in items:
            chunks = Compressor.chunks_from_passages(item.passages)
            if strategies is None:
                # Baseline: send everything, no compression.
                total = sum(__import__("context_compress").count_tokens(p) for p in item.passages)
                ratios.append(1.0)
                recalls.append(1.0)
                orig_tokens.append(total)
                comp_tokens.append(total)
                continue

            result = Compressor(strategies, budget=BUDGET).compress(item.query, chunks)
            report = evaluator.evaluate(result, item.required_facts)
            ratios.append(result.ratio)
            recalls.append(report.recall)
            orig_tokens.append(result.original_tokens)
            comp_tokens.append(result.compressed_tokens)

        rows[name] = {
            "mean_tokens": round(statistics.mean(comp_tokens), 1),
            "kept_ratio": round(statistics.mean(ratios), 4),
            "saved_pct": round(1 - statistics.mean(ratios), 4),
            "recall": round(statistics.mean(recalls), 4),
        }

    baseline_tokens = rows["no-compression"]["mean_tokens"]

    print(f"\n=== context-compress benchmark ===")
    print(f"{len(items)} RAG scenarios · budget {BUDGET} tokens · offline, seed 42\n")
    print(f"{'strategy':20} {'tokens':>8} {'saved':>8} {'recall':>8}")
    print("-" * 46)
    for name, r in rows.items():
        print(f"{name:20} {r['mean_tokens']:>8.1f} {r['saved_pct']:>7.0%} {r['recall']:>8.1%}")

    # Best = highest savings among strategies that hold recall >= 0.95.
    faithful = {
        k: v for k, v in rows.items() if k != "no-compression" and v["recall"] >= 0.95
    }
    if faithful:
        best = max(faithful.items(), key=lambda kv: kv[1]["saved_pct"])
        print(
            f"\nBest faithful compressor: {best[0]} — "
            f"{best[1]['saved_pct']:.0%} fewer tokens at {best[1]['recall']:.1%} recall"
        )
        print(
            f"({baseline_tokens:.0f} → {best[1]['mean_tokens']:.0f} tokens per request)"
        )
    else:
        print("\nNo strategy held recall >= 0.95 at this budget.")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({"budget": BUDGET, "n_items": len(items), "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
