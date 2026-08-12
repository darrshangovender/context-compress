"""context-compress — fit more signal into fewer tokens.

    from context_compress import Compressor, Chunk, Role
    from context_compress.strategies import DedupeStrategy, TopKStrategy

    chunks = Compressor.chunks_from_passages(retrieved_passages)
    result = Compressor([DedupeStrategy(), TopKStrategy()], budget=2000).compress(query, chunks)
    print(result.text, result.saved_pct)
"""

from context_compress.compressor import Compressor
from context_compress.fidelity import FidelityEvaluator
from context_compress.scoring import HybridScorer, LexicalScorer, PositionScorer
from context_compress.strategies import (
    BudgetTooSmall,
    DedupeStrategy,
    SentencePruneStrategy,
    Strategy,
    SummarizeStrategy,
    TopKStrategy,
    TruncateStrategy,
)
from context_compress.tokenizer import count_tokens
from context_compress.types import Chunk, CompressionResult, FidelityReport, Role

__version__ = "0.1.0"

__all__ = [
    "Compressor",
    "Chunk",
    "Role",
    "CompressionResult",
    "FidelityReport",
    "FidelityEvaluator",
    "Strategy",
    "BudgetTooSmall",
    "TruncateStrategy",
    "TopKStrategy",
    "SentencePruneStrategy",
    "DedupeStrategy",
    "SummarizeStrategy",
    "LexicalScorer",
    "PositionScorer",
    "HybridScorer",
    "count_tokens",
]
