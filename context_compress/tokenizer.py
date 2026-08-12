"""Token counting.

Uses ``tiktoken`` when installed for exact counts against OpenAI-family
tokenizers; otherwise falls back to a character-ratio heuristic that is within
roughly 10% on English prose. The fallback matters: it keeps the package
dependency-free so the tests and benchmark run anywhere, including CI with no
network access.
"""

from __future__ import annotations

import re
from functools import lru_cache

#: Empirical chars-per-token for English prose across GPT/Claude tokenizers.
_CHARS_PER_TOKEN = 3.8


@lru_cache(maxsize=4)
def _encoder(model: str):
    """Return a tiktoken encoder, or None if tiktoken isn't available."""
    try:
        import tiktoken  # lazy — optional dependency
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in ``text``.

    Exact when tiktoken is installed, heuristic otherwise. The heuristic blends
    a character estimate with a word estimate — punctuation-dense text tokenises
    differently from prose, and averaging the two is more stable than either.
    """
    if not text:
        return 0
    enc = _encoder(model)
    if enc is not None:
        return len(enc.encode(text))
    char_est = len(text) / _CHARS_PER_TOKEN
    word_est = len(text.split()) * 1.3
    return max(1, int(round((char_est + word_est) / 2)))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate ``text`` to at most ``max_tokens``, cutting on a word boundary."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text, model) <= max_tokens:
        return text
    enc = _encoder(model)
    if enc is not None:
        return enc.decode(enc.encode(text)[:max_tokens])
    # Heuristic path: binary search on words for the longest fitting prefix.
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid]), model) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences.

    Deliberately regex-based rather than pulling in nltk/spacy: sentence
    segmentation only needs to be good enough to give the scorers addressable
    units, and a heavyweight NLP dependency would make the package hostile to
    install.
    """
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])
