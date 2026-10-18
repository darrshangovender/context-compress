"""Regression tests for 0.1.1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from context_compress import Compressor, FidelityEvaluator
from context_compress.strategies import SentencePruneStrategy, TopKStrategy, TruncateStrategy
from context_compress.types import Chunk, Role

PASSAGES = [
    "Alpha service handles billing. It retries failed charges twice. Nothing else matters here.",
    "The p99 latency fell to 45ms after the cache change. Other numbers stayed flat.",
    "Gamma is unrelated filler text about office plants and watering schedules for spring.",
    "Delta documents the on-call rota. Escalation goes to the platform team after fifteen minutes.",
]


def _chunks() -> list[Chunk]:
    return Compressor.chunks_from_passages(PASSAGES)


def test_compress_does_not_modify_the_callers_chunks():
    """Strategies write kept/score/text onto the chunks they are given, so a chunk list
    used to be single-use: a second compression ran on already-truncated text."""
    chunks = _chunks()
    before = [(c.text, c.kept, c.score, dict(c.meta)) for c in chunks]
    Compressor([SentencePruneStrategy(min_sentences=0)], budget=25).compress("p99 latency", chunks)
    assert [(c.text, c.kept, c.score, dict(c.meta)) for c in chunks] == before


def test_the_same_chunk_list_can_be_compressed_twice_with_the_same_result():
    chunks = _chunks()
    comp = Compressor([SentencePruneStrategy(min_sentences=0)], budget=25)
    first = comp.compress("p99 latency", chunks)
    second = comp.compress("p99 latency", chunks)
    assert first.text == second.text
    assert first.compressed_tokens == second.compressed_tokens


def test_a_tighter_budget_after_a_looser_one_starts_from_the_original_text():
    chunks = _chunks()
    Compressor([SentencePruneStrategy(min_sentences=0)], budget=15).compress("p99 latency", chunks)
    roomy = Compressor([TopKStrategy()], budget=500).compress("p99 latency", chunks)
    assert "office plants" in roomy.text  # would have been pruned away on shared state


def test_result_chunks_are_copies_not_the_callers_objects():
    chunks = _chunks()
    result = Compressor([TruncateStrategy()], budget=500).compress("q", chunks)
    assert all(r is not c for r in result.chunks for c in chunks)


def test_protected_chunks_survive_copying_unchanged():
    chunks = [
        Chunk(text="Always answer in English.", role=Role.INSTRUCTION, position=0),
        *[Chunk(text=p, position=i + 1) for i, p in enumerate(PASSAGES)],
    ]
    result = Compressor([TopKStrategy()], budget=60).compress("p99 latency", chunks)
    assert result.chunks[0].text == "Always answer in English."
    assert result.chunks[0].kept


def test_exact_and_fuzzy_fidelity_disagree_when_words_survive_out_of_order():
    """The two checks the benchmark now reports side by side."""
    chunks = [Chunk(text="45ms was the p99. Latency fell.", position=0)]
    result = Compressor([TruncateStrategy()], budget=500).compress("q", chunks)
    fact = "p99 latency fell to 45ms"
    assert FidelityEvaluator(fuzzy=True).evaluate(result, [fact]).recall == 1.0
    assert FidelityEvaluator(fuzzy=False).evaluate(result, [fact]).recall == 0.0


def test_the_committed_benchmark_table_reproduces():
    """benchmarks/results.json (and so the README table) is exactly what the script prints."""
    repo = Path(__file__).resolve().parent.parent
    path = repo / "benchmarks" / "results.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    try:
        proc = subprocess.run(
            [sys.executable, "-B", str(repo / "benchmarks" / "run.py")],
            capture_output=True, text=True, cwd=repo, check=False, timeout=120,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        assert "exact" in proc.stdout
        regenerated = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.write_text(json.dumps(committed, indent=2), encoding="utf-8")
    assert regenerated == committed
    rows = regenerated["rows"]
    assert rows["truncate"]["saved_pct"] > rows["summarize"]["saved_pct"]
    assert rows["truncate"]["recall_exact"] < 0.5
