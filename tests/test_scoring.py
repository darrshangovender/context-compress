import pytest

from context_compress.scoring import (
    HybridScorer,
    LexicalScorer,
    PositionScorer,
    tokenize,
)
from context_compress.types import Chunk


def chunks(*texts):
    return [Chunk(text=t, position=i) for i, t in enumerate(texts)]


def test_tokenize_strips_stopwords():
    assert "the" not in tokenize("the quick brown fox")
    assert "quick" in tokenize("the quick brown fox")


def test_lexical_ranks_relevant_highest():
    cs = chunks(
        "The mitochondria is the powerhouse of the cell.",
        "Postgres query latency dropped after adding an index.",
        "Bananas are yellow.",
    )
    scores = LexicalScorer().score("postgres index latency", cs)
    assert scores[1] == max(scores)


def test_lexical_all_scores_bounded():
    cs = chunks("alpha beta", "gamma delta", "alpha gamma")
    for s in LexicalScorer().score("alpha", cs):
        assert 0.0 <= s <= 1.0


def test_lexical_empty_query_is_zero():
    cs = chunks("some text", "other text")
    assert LexicalScorer().score("", cs) == [0.0, 0.0]


def test_lexical_empty_chunks():
    assert LexicalScorer().score("query", []) == []


def test_rare_term_beats_common_term():
    """BM25 idf should favour the chunk with the rare identifier."""
    cs = chunks(
        "system system system system",
        "system error code XZ9142 occurred",
        "system system system",
    )
    scores = LexicalScorer().score("XZ9142", cs)
    assert scores[1] == max(scores)


def test_position_is_u_shaped():
    scores = PositionScorer().score("q", chunks("a", "b", "c", "d", "e"))
    assert scores[0] == 1.0        # head
    assert scores[-1] == 1.0       # tail
    assert scores[2] == min(scores)  # middle is weakest


def test_position_single_chunk():
    assert PositionScorer().score("q", chunks("only")) == [1.0]


def test_hybrid_blends_both():
    cs = chunks("alpha relevant", "filler", "filler", "filler")
    lex_only = HybridScorer(lexical_weight=1.0).score("alpha", cs)
    blended = HybridScorer(lexical_weight=0.5).score("alpha", cs)
    # With position mixed in, the tail chunk gains relative to pure lexical.
    assert blended[-1] > lex_only[-1]


def test_hybrid_rejects_bad_weight():
    with pytest.raises(ValueError):
        HybridScorer(lexical_weight=1.5)


def test_uniform_scores_normalise_to_one_not_zero():
    """All-equal relevance must not collapse to zero, or everything gets dropped."""
    cs = chunks("alpha", "alpha", "alpha")
    scores = LexicalScorer().score("alpha", cs)
    assert all(s == 1.0 for s in scores)
