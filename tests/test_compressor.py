import pytest

from context_compress import Chunk, Compressor, Role
from context_compress.fidelity import FidelityEvaluator
from context_compress.strategies import DedupeStrategy, TopKStrategy, TruncateStrategy
from context_compress.types import CompressionResult


def test_requires_strategies():
    with pytest.raises(ValueError):
        Compressor([], budget=100)


def test_requires_positive_budget():
    with pytest.raises(ValueError):
        Compressor([TopKStrategy()], budget=0)


def test_chunks_from_passages_preserves_order():
    chunks = Compressor.chunks_from_passages(["a", "b", "c"])
    assert [c.position for c in chunks] == [0, 1, 2]
    assert all(c.role is Role.CONTEXT for c in chunks)


def test_pipeline_dedupe_then_topk():
    dup = "The retry logic was fixed in release 4.2 which resolved the timeout."
    passages = [
        dup,
        "Postgres index rebuild reduced p99 latency substantially.",
        dup,  # exact duplicate — dedupe should remove
        "Unrelated marketing copy about our company values and mission.",
    ]
    chunks = Compressor.chunks_from_passages(passages)
    result = Compressor([DedupeStrategy(), TopKStrategy()], budget=40).compress(
        "postgres latency", chunks
    )
    assert isinstance(result, CompressionResult)
    assert result.compressed_tokens <= 40
    assert "→" in result.strategy  # composed pipeline is named


def test_result_text_reassembles_in_document_order():
    chunks = Compressor.chunks_from_passages(["alpha", "beta", "gamma"])
    result = Compressor([TruncateStrategy()], budget=10_000).compress("q", chunks)
    assert result.text.index("alpha") < result.text.index("beta") < result.text.index("gamma")


def test_saved_metrics_consistent():
    chunks = Compressor.chunks_from_passages([f"passage {i} " * 20 for i in range(8)])
    result = Compressor([TopKStrategy()], budget=60).compress("passage 3", chunks)
    assert result.saved_tokens == result.original_tokens - result.compressed_tokens
    assert 0.0 <= result.saved_pct <= 1.0
    assert result.ratio == pytest.approx(result.compressed_tokens / result.original_tokens)


def test_empty_result_ratio_is_one():
    r = CompressionResult(chunks=[], original_tokens=0, compressed_tokens=0, strategy="noop")
    assert r.ratio == 1.0
    assert r.saved_pct == 0.0


def test_protected_survives_full_pipeline():
    chunks = [
        Chunk(text="You are a careful assistant.", role=Role.SYSTEM, position=0),
        *[Chunk(text=f"filler {i} " * 20, role=Role.CONTEXT, position=i + 1) for i in range(6)],
        Chunk(text="What is the latency?", role=Role.QUERY, position=99),
    ]
    result = Compressor([DedupeStrategy(), TopKStrategy()], budget=80).compress("latency", chunks)
    assert "careful assistant" in result.text
    assert "What is the latency?" in result.text


# --- fidelity ---------------------------------------------------------------

def test_fidelity_perfect_when_nothing_dropped():
    chunks = Compressor.chunks_from_passages([
        "The p99 latency dropped to 45ms after the index rebuild.",
        "Secondary note about deployment timing.",
    ])
    result = Compressor([TruncateStrategy()], budget=10_000).compress("latency", chunks)
    report = FidelityEvaluator().evaluate(result, ["p99 latency dropped to 45ms"])
    assert report.recall == 1.0
    assert report.n_found == 1


def test_fidelity_detects_lost_fact():
    chunks = Compressor.chunks_from_passages([
        "Irrelevant preamble text " * 10,
        "The critical value is XZ9142.",
    ])
    # Tiny budget + head truncation drops the tail chunk holding the fact.
    result = Compressor([TruncateStrategy()], budget=12).compress("value", chunks)
    report = FidelityEvaluator().evaluate(result, ["The critical value is XZ9142"])
    assert report.recall == 0.0
    assert report.missing


def test_fidelity_fuzzy_matches_reordered_words():
    chunks = Compressor.chunks_from_passages(["Latency was 45ms at p99 after tuning."])
    result = Compressor([TruncateStrategy()], budget=10_000).compress("q", chunks)
    report = FidelityEvaluator(fuzzy=True).evaluate(result, ["p99 45ms latency"])
    assert report.recall == 1.0


def test_fidelity_strict_mode_requires_substring():
    chunks = Compressor.chunks_from_passages(["Latency was 45ms at p99 after tuning."])
    result = Compressor([TruncateStrategy()], budget=10_000).compress("q", chunks)
    report = FidelityEvaluator(fuzzy=False).evaluate(result, ["p99 45ms latency"])
    assert report.recall == 0.0


def test_fidelity_f1_computed():
    chunks = Compressor.chunks_from_passages(["Alpha fact here.", "Beta fact here."])
    result = Compressor([TruncateStrategy()], budget=10_000).compress("q", chunks)
    report = FidelityEvaluator().evaluate(result, ["Alpha fact"])
    assert 0.0 <= report.f1 <= 1.0
