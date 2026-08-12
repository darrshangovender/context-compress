from context_compress.tokenizer import count_tokens, split_sentences, truncate_to_tokens


def test_empty_is_zero():
    assert count_tokens("") == 0


def test_count_scales_with_length():
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 50)
    assert long > short * 10


def test_truncate_respects_budget():
    text = "word " * 200
    out = truncate_to_tokens(text, 20)
    assert count_tokens(out) <= 20


def test_truncate_noop_when_under_budget():
    text = "short text"
    assert truncate_to_tokens(text, 1000) == text


def test_truncate_zero_returns_empty():
    assert truncate_to_tokens("anything", 0) == ""


def test_split_sentences_basic():
    s = split_sentences("First one. Second one! Third one?")
    assert len(s) == 3


def test_split_sentences_single():
    assert split_sentences("No terminator here") == ["No terminator here"]


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_split_does_not_break_on_decimals():
    # A naive split on "." would shatter this; the regex requires whitespace +
    # capital/digit after the terminator.
    s = split_sentences("Latency fell to 0.5 seconds. That is fast.")
    assert len(s) == 2
