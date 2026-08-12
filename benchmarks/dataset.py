"""Synthetic RAG benchmark with a controlled signal-to-noise ratio.

Each item is a realistic retrieval result: a handful of passages where one or
two carry the answer and the rest are plausible-but-irrelevant neighbours, plus
some near-duplicates (which real vector search reliably produces from sliding
-window chunking).

Deterministic under a seed, so the benchmark is reproducible offline with no
API keys and no model downloads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_TOPICS = [
    ("postgres", "index", "query latency", "p99 fell to {v}ms after the index rebuild"),
    ("redis", "cache", "hit rate", "cache hit rate reached {v} percent after tuning"),
    ("kafka", "consumer", "lag", "consumer lag dropped to {v} messages under the new partitioning"),
    ("s3", "upload", "throughput", "multipart upload throughput hit {v} MB per second"),
    ("nginx", "tls", "handshake", "TLS handshake time settled at {v}ms with session resumption"),
    ("docker", "image", "build time", "image build completed in {v} seconds using layer caching"),
    ("python", "gil", "concurrency", "worker pool sustained {v} requests per second"),
    ("dns", "resolver", "lookup", "resolver lookup averaged {v}ms after enabling the local cache"),
]

_NOISE = [
    "The quarterly planning meeting has been moved to the first Tuesday of the month.",
    "Our office relocated to the third floor of the adjacent building last spring.",
    "The onboarding checklist now includes a security awareness training module.",
    "Vendor invoices are processed on a net-thirty basis unless otherwise agreed.",
    "The company all-hands is recorded and posted to the internal wiki afterwards.",
    "Parking permits must be renewed annually through the facilities portal.",
    "Coffee machine maintenance is scheduled for the second week of each quarter.",
    "The employee handbook was last revised to clarify remote work expectations.",
]


@dataclass
class BenchItem:
    """One retrieval scenario."""

    query: str
    passages: list[str]
    required_facts: list[str]
    meta: dict = field(default_factory=dict)


def make_dataset(n: int = 60, seed: int = 42) -> list[BenchItem]:
    rng = random.Random(seed)
    items: list[BenchItem] = []

    for i in range(n):
        system, component, metric, fact_tpl = _TOPICS[i % len(_TOPICS)]
        value = rng.randint(10, 999)
        fact = fact_tpl.format(v=value)

        query = f"What happened to {system} {metric}?"

        # The answer-bearing passage, padded with genuine surrounding prose so
        # sentence-level pruning has something real to strip.
        signal = (
            f"During the {system} performance review the team examined the {component} path. "
            f"{fact.capitalize()}. "
            f"Follow-up monitoring confirmed the improvement held across the following week. "
            f"The change was documented in the internal runbook."
        )

        # Same-domain distractors: topically adjacent, do not contain the answer.
        distractors = [
            f"The {system} deployment pipeline runs nightly and publishes artefacts to the registry.",
            f"Historically the {component} configuration was managed by hand before automation.",
            f"A separate {system} cluster serves the staging environment on smaller instances.",
        ]

        noise = rng.sample(_NOISE, k=3)

        passages = [*noise[:2], *distractors[:2], signal, distractors[2], noise[2]]

        # Real retrieval returns overlapping chunks — duplicate the signal in
        # ~40% of items so dedupe has something to earn its place against.
        if rng.random() < 0.4:
            passages.insert(rng.randint(0, len(passages)), signal)

        items.append(
            BenchItem(
                query=query,
                passages=passages,
                required_facts=[fact],
                meta={"system": system, "has_duplicate": passages.count(signal) > 1},
            )
        )

    return items
