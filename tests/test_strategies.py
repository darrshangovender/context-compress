import pytest

from context_compress.strategies import (
    BudgetTooSmall,
    DedupeStrategy,
    SentencePruneStrategy,
    SummarizeStrategy,
    TopKStrategy,
    TruncateStrategy,
)
from context_compress.strategies.dedupe import jaccard, shingles
from context_compress.tokenizer import count_tokens
from context_compress.types import Chunk, Role


def ctx(*texts):
    return [Chunk(text=t, role=Role.CONTEXT, position=i) for i, t in enumerate(texts)]


def with_protected(*texts):
    cs = [Chunk(text="SYSTEM PROMPT HERE", role=Role.SYSTEM, position=0)]
    cs += [Chunk(text=t, role=Role.CONTEXT, position=i + 1) for i, t in enumerate(texts)]
    cs.append(Chunk(text="the user question", role=Role.QUERY, position=99))
    return cs


# --- shared invariants across every strategy -------------------------------

ALL = [
    TruncateStrategy(),
    TopKStrategy(),
    SentencePruneStrategy(),
    DedupeStrategy(),
    SummarizeStrategy(),
]


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.name)
def test_never_drops_protected_chunks(strategy):
    chunks = with_protected(*[f"filler passage number {i} " * 20 for i in range(8)])
    result = strategy.compress("question", chunks, budget=200)
    for c in result.chunks:
        if c.protected:
            assert c.kept, f"{strategy.name} dropped a protected chunk"


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.name)
def test_raises_when_protected_exceeds_budget(strategy):
    chunks = [Chunk(text="x " * 500, role=Role.SYSTEM, position=0)]
    with pytest.raises(BudgetTooSmall):
        strategy.compress("q", chunks, budget=10)


@pytest.mark.parametrize(
    "strategy", [TruncateStrategy(), TopKStrategy(), SentencePruneStrategy(), SummarizeStrategy()],
    ids=lambda s: s.name,
)
def test_respects_budget(strategy):
    chunks = ctx(*[f"passage {i} with a reasonable amount of words in it " * 10 for i in range(10)])
    budget = 150
    result = strategy.compress("passage", chunks, budget=budget)
    assert result.compressed_tokens <= budget


@pytest.mark.parametrize(
    "strategy",
    [TruncateStrategy(), TopKStrategy(), SentencePruneStrategy(), DedupeStrategy()],
    ids=lambda s: s.name,
)
def test_generous_budget_keeps_everything(strategy):
    """Budget-driven strategies keep everything when budget is not binding.

    SummarizeStrategy is deliberately excluded: it is relevance-tiered, not
    budget-driven, and drops sub-threshold chunks even when budget remains.
    That behaviour is covered by test_summarize_drops_irrelevant_despite_budget.
    """
    chunks = ctx("alpha content here", "beta content here", "gamma content here")
    result = strategy.compress("alpha", chunks, budget=100_000)
    assert result.ratio == pytest.approx(1.0, abs=0.01)


# --- truncate ---------------------------------------------------------------

def test_truncate_keeps_head_first():
    # Budget fits roughly one chunk (~43 tokens each), so the head survives and
    # the tail does not.
    chunks = ctx("first " * 30, "second " * 30, "third " * 30)
    result = TruncateStrategy().compress("q", chunks, budget=50)
    assert chunks[0].kept
    assert not chunks[-1].kept


def test_truncate_keeps_nothing_when_no_chunk_fits():
    """Chunk-level granularity means a budget smaller than the first chunk
    yields an empty context rather than a partial chunk. Documented contract:
    use SentencePruneStrategy when sub-chunk granularity is needed."""
    chunks = ctx("word " * 100)
    result = TruncateStrategy().compress("q", chunks, budget=5)
    assert result.compressed_tokens == 0


def test_truncate_tail_ratio_keeps_last():
    chunks = ctx(*[f"chunk{i} " * 20 for i in range(6)])
    result = TruncateStrategy(tail_ratio=0.5).compress("q", chunks, budget=80)
    kept = [c for c in result.chunks if c.kept]
    assert any(c.position == 5 for c in kept), "tail_ratio should preserve the final chunk"


def test_truncate_rejects_bad_ratio():
    with pytest.raises(ValueError):
        TruncateStrategy(tail_ratio=1.0)


# --- topk -------------------------------------------------------------------

def test_topk_keeps_relevant_over_head():
    chunks = ctx(
        "irrelevant filler about weather " * 8,
        "irrelevant filler about sports " * 8,
        "the postgres index rebuild fixed the latency problem " * 3,
    )
    result = TopKStrategy().compress("postgres index latency", chunks, budget=45)
    assert chunks[2].kept, "top-k must beat truncation on relevance"


def test_topk_min_score_leaves_budget_unused():
    chunks = ctx("completely unrelated text about gardening")
    result = TopKStrategy(min_score=1.5).compress("quantum computing", chunks, budget=10_000)
    assert result.compressed_tokens == 0, "noise should not be kept just because budget remains"


def test_topk_populates_scores():
    chunks = ctx("alpha", "beta")
    TopKStrategy().compress("alpha", chunks, budget=1000)
    assert all(c.score is not None for c in chunks)


# --- sentence prune ---------------------------------------------------------

def test_sentence_prune_compresses_within_chunk():
    text = (
        "The system uses Postgres. "
        "Weather in Durban is warm. "
        "Cats sleep a lot. "
        "Bananas are yellow. "
        "The sky is blue."
    )
    chunks = ctx(text)
    result = SentencePruneStrategy(min_sentences=1).compress("postgres", chunks, budget=12)
    assert result.compressed_tokens < count_tokens(text)
    assert "Postgres" in result.text


def test_sentence_prune_min_sentences_honoured():
    chunks = ctx("Alpha one. Beta two. Gamma three. Delta four.")
    result = SentencePruneStrategy(min_sentences=2).compress("nothing relevant", chunks, budget=1000)
    assert "Alpha one." in result.text and "Beta two." in result.text


# --- dedupe -----------------------------------------------------------------

def test_shingles_and_jaccard_identity():
    a = shingles("the quick brown fox jumps over")
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint_is_zero():
    assert jaccard(shingles("alpha beta gamma"), shingles("delta epsilon zeta")) == 0.0


def test_dedupe_removes_exact_duplicate():
    dup = "The migration completed successfully after the retry logic was fixed."
    chunks = ctx(dup, "Unrelated distinct content about something else entirely.", dup)
    result = DedupeStrategy().compress("q", chunks, budget=10_000)
    assert chunks[0].kept and not chunks[2].kept
    assert chunks[2].meta["dropped_reason"] == "near_duplicate"


def test_dedupe_keeps_distinct():
    chunks = ctx(
        "Postgres indexes speed up reads considerably.",
        "Redis caches hot keys in memory for fast access.",
    )
    DedupeStrategy().compress("q", chunks, budget=10_000)
    assert all(c.kept for c in chunks)


def test_dedupe_rejects_bad_threshold():
    with pytest.raises(ValueError):
        DedupeStrategy(threshold=0.0)


# --- summarize --------------------------------------------------------------

def test_summarize_tiers_chunks():
    chunks = ctx(
        "postgres index latency tuning explained in detail here",
        "somewhat related database background information for context",
        "utterly unrelated content concerning tropical fruit varieties",
    )
    result = SummarizeStrategy().compress("postgres index latency", chunks, budget=10_000)
    tiers = {c.meta.get("tier") for c in result.chunks}
    assert tiers & {"verbatim", "condensed", "dropped"}


def test_summarize_validates_thresholds():
    with pytest.raises(ValueError):
        SummarizeStrategy(keep_threshold=0.2, drop_threshold=0.8)


def test_summarize_drops_irrelevant_despite_budget():
    """Unlike the budget-driven strategies, summarize prunes noise even when
    there is room to keep it — irrelevant context degrades answer quality."""
    chunks = ctx(
        "postgres index latency tuning explained thoroughly in this passage",
        "utterly unrelated content concerning tropical fruit cultivation methods",
    )
    result = SummarizeStrategy().compress("postgres index latency", chunks, budget=100_000)
    assert result.ratio < 1.0


def test_sentence_prune_min_zero_frees_budget_for_relevant():
    """min_sentences=0 lets the scorer spend the whole budget on relevant
    sentences instead of pinning every chunk's lead sentence. This is the right
    default for RAG, where lead sentences are usually framing, not answer."""
    passages = [
        "General preamble sentence. More framing here. Yet more setup text.",
        "Another preamble sentence. The p99 latency fell to 45ms after tuning. Trailing note.",
    ]
    chunks = ctx(*passages)
    pinned = SentencePruneStrategy(min_sentences=1).compress("p99 latency 45ms", chunks, budget=20)

    chunks2 = ctx(*passages)
    free = SentencePruneStrategy(min_sentences=0).compress("p99 latency 45ms", chunks2, budget=20)

    assert "45ms" in free.text
    assert "45ms" not in pinned.text
