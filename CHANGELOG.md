# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-10-14

### Fixed
- `Compressor.compress` mutated the caller's chunks (strategies write `kept`, `score` and
  rewritten `text` onto them), so a chunk list was single-use: a second compression ran on
  already-pruned text. The pipeline now works on copies and leaves the input untouched.
- Docstrings that did not match the code: `DedupeStrategy` claimed MinHash (it is exact pairwise
  Jaccard, O(n^2)); `SentencePruneStrategy` documented `min_sentences` as a guarantee (it is a
  priority, and the token budget wins).
- The benchmark reported a single recall column from the lenient bag-of-words fidelity check. It
  now reports `recall` (fuzzy) and `exact` (contiguous substring) side by side, picks the "best
  faithful compressor" only if both hold at least 0.95, and the committed
  `benchmarks/results.json` is regenerated with both.
- README: the stale test count (69; it is 71, now 78) and the statement that no results file was
  committed (`benchmarks/results.json` is, and a test checks it reproduces).

### Added
- `tests/test_regressions.py` (71 to 78 tests), `CHANGELOG.md`, an explicit ruff rule set, and lint
  as a CI gate.

## [0.1.0] - 2026-10-09

### Added
- Initial public reference implementation: five compression strategies, a composable pipeline,
  BM25 + position scoring, a fact-recall fidelity evaluator, an offline benchmark.
