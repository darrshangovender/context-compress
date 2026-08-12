<div align="center">

# context-compress — fit more signal into fewer tokens

[![tests](https://github.com/darrshangovender/context-compress/actions/workflows/tests.yml/badge.svg)](https://github.com/darrshangovender/context-compress/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Zero dependencies](https://img.shields.io/badge/dependencies-none-22c55e)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-Working%20code-blue)](#)

</div>

---

> Prompt and RAG context compression that **measures whether it broke the answer**.
> Five strategies, a composable pipeline, and a fidelity evaluator that reports
> fact recall alongside token savings — because a compressor that halves your
> tokens and loses the answer is not a saving, it's an outage.

**Why this exists.** Long contexts are the largest controllable line item in an
LLM bill, and retrieved RAG context is mostly padding — overlapping chunks,
near-duplicates, and framing prose around one or two load-bearing sentences.
Everyone truncates. Almost nobody measures what truncation destroyed.

> **Completes the inference-economics set.**
> [`cascade`](https://github.com/darrshangovender/cascade) picks the cheapest *model*.
> [`thinking-loop`](https://github.com/darrshangovender/thinking-loop) spends more compute when a question is *hard*.
> `context-compress` shrinks the *input* to every call. Route → scale → compress.

---

## The result

60 synthetic RAG scenarios, 120-token context budget, fully offline
(`python benchmarks/run.py`, seed 42 — reproducible, no API keys, no model downloads):

| strategy | mean tokens | saved | **fact recall** |
|---|---:|---:|---:|
| no compression | 186.4 | 0% | 100.0% |
| truncate | 79.8 | **57%** | 21.7% ❌ |
| truncate + tail | 99.3 | 45% | 10.0% ❌ |
| top-k | 113.1 | 38% | 100.0% |
| sentence-prune (min=1) | 118.3 | 35% | 48.3% ⚠️ |
| sentence-prune (min=0) | 115.3 | 37% | 95.0% |
| **summarize** | **96.1** | **48%** | **100.0%** ✅ |
| dedupe → top-k | 111.5 | 39% | 100.0% |
| dedupe → sentence | 115.9 | 36% | 100.0% |

**Read the last column first.** Truncation is the biggest saver and the worst
compressor — it throws away 78% of the answers. `summarize` gets **48% fewer
tokens at full recall**: 186 → 96 tokens per request, with nothing lost.

That gap is the entire argument for this library. Savings without a fidelity
number is a metric you cannot act on.

---

## Quick start

```bash
pip install -e ".[dev]"      # zero runtime dependencies
python benchmarks/run.py     # reproduce the table above
```

```python
from context_compress import Compressor, FidelityEvaluator
from context_compress.strategies import DedupeStrategy, SummarizeStrategy

chunks = Compressor.chunks_from_passages(retrieved_passages)

result = Compressor(
    [DedupeStrategy(), SummarizeStrategy()],   # dedupe first, then condense
    budget=2000,
).compress(user_query, chunks)

print(result.text)          # the compressed context, in document order
print(f"{result.saved_pct:.0%} fewer tokens")

# Did it survive? Check against facts the answer requires.
report = FidelityEvaluator().evaluate(result, required_facts=["p99 fell to 45ms"])
print(f"recall {report.recall:.0%}, missing: {report.missing}")
```

---

## The five strategies

| Strategy | Granularity | What it does | Best for |
|---|---|---|---|
| `TruncateStrategy` | chunk | Fill from head (and optionally tail) | The baseline to beat; conversation history with `tail_ratio` |
| `TopKStrategy` | chunk | Score vs query, greedily keep the best | General RAG; the safe default |
| `SentencePruneStrategy` | **sentence** | Keep the highest-value sentences *within* chunks | Verbose passages where signal is 10–20% of text |
| `DedupeStrategy` | chunk | Drop near-duplicates via MinHash/Jaccard shingles | Sliding-window retrieval (always compose this first) |
| `SummarizeStrategy` | tiered | Keep / condense / drop by relevance band | The best all-rounder — see benchmark |

Strategies compose, and **order is load-bearing**:

```python
Compressor([DedupeStrategy(), TopKStrategy()], budget=2000)
```

Dedupe first so the selector never spends budget ranking three copies of the
same passage. Reversed, you pay to rank redundancy.

---

## Two invariants every strategy honours

**1. Protected roles are never touched.** `SYSTEM`, `INSTRUCTION` and `QUERY`
chunks pass through untouched. Dropping an instruction to save tokens silently
changes the task rather than the evidence — the worst failure mode a compressor
has, and one that's easy to ship by accident.

**2. Budget is never exceeded.** If protected content alone exceeds the budget,
you get `BudgetTooSmall` rather than a silently mangled prompt.

Both are enforced by parametrised tests across all five strategies.

---

## An honest limitation, surfaced in the benchmark

`SentencePruneStrategy(min_sentences=1)` — the intuitive default — scores
**48.3% recall**, far worse than its `min_sentences=0` sibling at 95%.

The reason: pinning the lead sentence of *every* chunk spends the budget on
framing prose before the scorer can reach the sentence that actually holds the
answer. On RAG workloads, lead sentences are usually setup, not substance.

**Use `min_sentences=0` for retrieved passages.** Both variants stay in the
benchmark so the cost of the wrong default is visible rather than buried.

---

## Design decisions

| Decision | Why |
|---|---|
| **Zero runtime dependencies** | Installs and runs anywhere, including offline CI. `tiktoken` is optional for exact counts; a blended char/word heuristic (within ~10% on prose) covers the rest. |
| **BM25, not raw TF-IDF** | Term saturation matters: a chunk repeating a query term 20× isn't 20× more relevant. Without it, keyword-stuffed boilerplate beats the passage that answers the question. |
| **Extractive, never abstractive** | An LLM-written summary costs a call per chunk and introduces a hallucination surface *inside your context* — the one place ground truth must stay intact. |
| **Position scoring is U-shaped** | Models recall the head and tail of long contexts far better than the middle (Liu et al., 2023), so dropping from the weak middle costs least. |
| **Kept chunks re-emitted in document order** | Reordering context to match relevance rank measurably hurts multi-hop questions where narrative sequence carries meaning. |
| **Fidelity is a first-class output** | Compression ratio alone is a vanity metric — deleting everything scores 100%. |

---

## Project layout

```
context_compress/
├── types.py            # Chunk, Role, CompressionResult, FidelityReport
├── tokenizer.py        # tiktoken-optional counting + sentence splitting
├── scoring.py          # BM25 lexical · U-shaped position · hybrid
├── fidelity.py         # fact-recall evaluator (the metric that matters)
├── compressor.py       # the composable pipeline
└── strategies/         # truncate · topk · sentence · dedupe · summarize
benchmarks/             # reproducible offline benchmark + synthetic RAG set
tests/                  # 69 tests, all offline
docs/                   # architecture · strategies · fidelity
```

## Tests

```bash
pytest tests/ -q      # 69 tests, no API keys, no network
python benchmarks/run.py
```

CI runs both on every push.

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za) · Durban, South Africa
