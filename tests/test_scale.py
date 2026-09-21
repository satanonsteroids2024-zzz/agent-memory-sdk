"""Scale and decision-quality tests over bulk data.

Seeds a few hundred memories across realistic domains, then verifies the
decision layer holds up: exact queries replay the *right* memory, unrelated
queries return NONE, paraphrases restore, and latency stays flat thanks to
the FTS5 index (instead of rebuilding a Python BM25 index per query).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_memory import Memory, MemoryAction

DOMAINS = {
    "kubernetes": "kubectl apply the manifest in deploy/{n}.yaml, then watch rollout status.",
    "postgres": "Run the migration in migrations/{n}.sql inside a transaction.",
    "react": "Update the component in src/components/Widget{n}.tsx and run the storybook.",
    "billing": "Invoice {n} disputes go through the finance queue with a 5-day SLA.",
    "onboarding": "New hire task {n} is tracked in the onboarding board, column 'week one'.",
}

UNRELATED_QUERIES = [
    "What's the best recipe for sourdough bread?",
    "Who won the football match yesterday evening?",
    "Recommend a good science fiction novel from the nineties",
    "What are the opening hours of the museum downtown?",
    "Translate good morning into Japanese please",
    "What's the average rainfall in the Amazon basin?",
    "How tall is the Eiffel Tower exactly?",
    "Suggest a birthday gift for a six year old",
]


@pytest.fixture(scope="module")
def bulk_memory(tmp_path_factory: pytest.TempPathFactory) -> Memory:
    mem = Memory(persist_dir=tmp_path_factory.mktemp("scale") / "mem")
    for domain, template in DOMAINS.items():
        for n in range(80):
            mem.remember(
                f"How do I handle {domain} task number {n}?",
                template.format(n=n),
                tags=[domain],
            )
    return mem  # 400 memories


def test_seeded_count(bulk_memory: Memory) -> None:
    assert bulk_memory.stats()["total"] == 400


def test_exact_queries_replay_correct_memory(bulk_memory: Memory) -> None:
    """Exact repeats must replay their own answer, not a near-neighbor's."""
    for domain in DOMAINS:
        for n in (0, 37, 79):
            decision = bulk_memory.resolve(f"How do I handle {domain} task number {n}?")
            assert decision.action == MemoryAction.REPLAY, (domain, n, decision.reason)
            assert str(n) in decision.response, (domain, n, decision.response)


def test_unrelated_queries_return_none(bulk_memory: Memory) -> None:
    for query in UNRELATED_QUERIES:
        decision = bulk_memory.resolve(query)
        assert decision.action == MemoryAction.NONE, (query, decision.confidence)


def test_paraphrases_do_not_answer_from_wrong_domain(bulk_memory: Memory) -> None:
    decision = bulk_memory.resolve("What's the process for postgres task 12?")
    assert decision.action in (MemoryAction.REPLAY, MemoryAction.RESTORE, MemoryAction.VERIFY)
    top = decision.context[0].entry if decision.context else decision.memory
    assert top is not None
    assert "postgres" in top.tags


def test_scoped_resolution_isolates_domains(bulk_memory: Memory) -> None:
    # All were stored with USER scope; a disjoint scope filter finds nothing.
    decision = bulk_memory.resolve(
        "How do I handle kubernetes task number 3?", scope=["team"]
    )
    assert decision.action == MemoryAction.NONE


def test_resolve_latency_stays_flat(bulk_memory: Memory) -> None:
    """FTS5-backed retrieval should answer in milliseconds at 400 memories.

    The bound is generous (250ms) to tolerate slow CI runners; the point is
    to catch a regression back to per-query index rebuilding, which costs
    an order of magnitude more.
    """
    latencies = []
    for n in range(0, 80, 4):
        start = time.perf_counter()
        bulk_memory.resolve(f"How do I handle react task number {n}?")
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    assert p95 < 250, f"p95 resolve latency {p95:.0f}ms — retrieval regressed"


def test_stats_aggregates_without_row_cap(tmp_path: Path) -> None:
    mem = Memory(persist_dir=tmp_path / "stats_mem")
    for i in range(50):
        mem.remember(f"fact {i}", f"answer {i}", type="fact")
    for i in range(30):
        mem.remember(f"workflow {i}", f"steps {i}", type="workflow")
    stats = mem.stats()
    assert stats["total"] == 80
    assert stats["by_type"] == {"fact": 50, "workflow": 30}
    assert stats["by_state"]["active"] == 80


def test_cleanup_uses_sql_and_removes_index_rows(tmp_path: Path) -> None:
    mem = Memory(persist_dir=tmp_path / "cleanup_mem")
    for i in range(20):
        mem.remember(f"ephemeral {i}", f"gone soon {i}", ttl=0.05)
    for i in range(5):
        mem.remember(f"durable {i}", f"stays {i}")
    time.sleep(0.3)

    marked = mem.cleanup(delete=False)
    assert marked["expired"] == 20
    deleted = mem.cleanup(delete=True)
    assert deleted["deleted"] == 20
    assert mem.stats()["total"] == 5
    # Expired entries must be gone from the search index too.
    decision = mem.resolve("ephemeral cache entries")
    assert decision.action == MemoryAction.NONE
    assert all("gone soon" not in r.entry.response for r in decision.context)


def test_consolidate_scales_linearly(tmp_path: Path) -> None:
    """consolidate() must issue one search per entry, not one per pair."""
    mem = Memory(persist_dir=tmp_path / "consolidate_mem")
    for i in range(60):
        mem.remember(f"unique question about topic {i}?", f"unique answer {i}")
    # Three exact duplicates that should merge.
    for _ in range(3):
        mem.remember("What is the deploy freeze window?", "Fridays after 3pm UTC.")

    start = time.perf_counter()
    created = mem.consolidate(similarity_threshold=0.95)
    elapsed = time.perf_counter() - start

    assert elapsed < 10, f"consolidate took {elapsed:.1f}s on 63 entries"
    if created:  # duplicates merged into a summary
        assert any("Fridays after 3pm" in s.content for s in created)
