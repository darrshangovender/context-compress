# Choosing a strategy

A decision guide, grounded in the benchmark in `benchmarks/run.py`.

## The short answer

```python
Compressor([DedupeStrategy(), SummarizeStrategy()], budget=N)
```

Best measured savings at full fact recall (48% fewer tokens, 100% recall). Start
here; deviate when you have a reason below.

---

## When each one wins

### `TruncateStrategy`
**Use when:** you need zero scoring overhead, or you're compressing conversation
history where recency dominates (`tail_ratio=0.3`).

**Do not use for RAG.** The benchmark is unambiguous: 57% savings at 21.7%
recall. It is the biggest saver and the worst compressor, because it is blind to
the query. It's in the library as the honest baseline every other strategy must
beat.

### `TopKStrategy`
**Use when:** you want a safe, predictable default. Query-aware, keeps whole
chunks, never mangles prose.

**Limitation:** all-or-nothing granularity. A 400-token passage containing one
critical sentence costs 400 tokens or zero.

### `SentencePruneStrategy`
**Use when:** passages are verbose and the signal density is low — long
documentation pages, transcripts, scraped articles.

**Set `min_sentences=0` for RAG.** The default of 1 pins every chunk's lead
sentence and drops recall from 95% to 48.3% (see benchmark). Lead sentences in
retrieved passages are usually framing, not answer.

**Trade-off:** aggressive pruning yields disjointed fragments. If output quality
degrades, raise `min_sentences` or switch to `SummarizeStrategy`.

### `DedupeStrategy`
**Always compose it first.** It is not budget-driven — it removes redundancy and
stops. Sliding-window chunking reliably returns the same passage two or three
times; every copy after the first is pure waste that also crowds out distinct
evidence.

Tune `threshold` down (0.6–0.7) for aggressive dedup, up (0.9) if you're losing
genuinely distinct passages that share boilerplate.

### `SummarizeStrategy`
**The best all-rounder.** Tiers each chunk into keep-verbatim / condense / drop
by relevance, so marginal material is compressed rather than binary-dropped.

Tune `keep_threshold` and `drop_threshold` to move along the savings/recall
curve; `condense_ratio` controls how hard the middle tier is squeezed.

---

## Composition patterns

| Goal | Pipeline |
|---|---|
| Best general RAG | `[DedupeStrategy(), SummarizeStrategy()]` |
| Safe and predictable | `[DedupeStrategy(), TopKStrategy()]` |
| Maximum squeeze | `[DedupeStrategy(), SentencePruneStrategy(min_sentences=0)]` |
| Conversation history | `[TruncateStrategy(tail_ratio=0.4)]` |
| Cheapest possible | `[TruncateStrategy()]` — only if you've measured recall |

**Rule:** cheap filters before expensive selectors. Dedupe removes candidates
for free; running it after selection means you paid to rank duplicates.

---

## Tuning against your own data

Compression ratios are workload-specific. Do not trust the numbers in this
README for your corpus — reproduce them:

1. Collect 50–100 real (query, retrieved passages, required facts) triples.
2. Swap them into `benchmarks/dataset.py`.
3. Run `python benchmarks/run.py` and read the recall column.
4. Pick the highest-saving strategy that holds recall above your bar.

The bar depends on the product. A support assistant might tolerate 95%; a
system quoting figures to a customer should not go below 100%.
