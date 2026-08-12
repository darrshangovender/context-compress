"""End-to-end example: compress a RAG context and verify nothing broke.

Runs fully offline.

    python examples/rag_pipeline.py
"""

from context_compress import Chunk, Compressor, FidelityEvaluator, Role, count_tokens
from context_compress.strategies import DedupeStrategy, SummarizeStrategy, TopKStrategy

QUERY = "What happened to the postgres query latency?"

# A realistic retrieval result: duplicates, distractors, one answer-bearing passage.
PASSAGES = [
    "The quarterly planning meeting moved to the first Tuesday of each month.",
    "The postgres deployment pipeline runs nightly and publishes to the registry.",
    "During the postgres performance review the team examined the index path. "
    "The p99 query latency fell to 45ms after the index rebuild. "
    "Follow-up monitoring confirmed the improvement held for a week.",
    "The postgres deployment pipeline runs nightly and publishes to the registry.",  # dup
    "Historically the index configuration was managed by hand before automation.",
    "Parking permits must be renewed annually through the facilities portal.",
]

REQUIRED_FACTS = ["p99 query latency fell to 45ms"]


def main() -> None:
    original_tokens = sum(count_tokens(p) for p in PASSAGES)
    print(f"Original context: {original_tokens} tokens across {len(PASSAGES)} passages\n")

    pipelines = {
        "topk only": [TopKStrategy()],
        "dedupe → topk": [DedupeStrategy(), TopKStrategy()],
        "dedupe → summarize": [DedupeStrategy(), SummarizeStrategy()],
    }

    evaluator = FidelityEvaluator()

    for name, strategies in pipelines.items():
        # Protected chunks demonstrate that instructions always survive.
        chunks = [
            Chunk(text="Answer only from the provided context.", role=Role.SYSTEM, position=0),
            *[Chunk(text=p, role=Role.CONTEXT, position=i + 1) for i, p in enumerate(PASSAGES)],
            Chunk(text=QUERY, role=Role.QUERY, position=99),
        ]
        result = Compressor(strategies, budget=90).compress(QUERY, chunks)
        report = evaluator.evaluate(result, REQUIRED_FACTS)

        print(f"[{name}]")
        print(f"  tokens : {result.original_tokens} → {result.compressed_tokens} "
              f"({result.saved_pct:.0%} saved)")
        print(f"  recall : {report.recall:.0%}"
              + (f"  MISSING: {report.missing}" if report.missing else ""))
        print(f"  system prompt survived: {'Answer only' in result.text}")
        print()


if __name__ == "__main__":
    main()
