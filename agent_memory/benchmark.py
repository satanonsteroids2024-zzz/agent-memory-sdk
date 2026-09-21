from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean

from agent_memory.manager import Memory
from agent_memory.models import MemoryAction


@dataclass
class BenchmarkResult:
    queries: int = 0
    replay_count: int = 0
    restore_count: int = 0
    verify_count: int = 0
    none_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    memory_hits: int = 0

    @property
    def replay_rate(self) -> float:
        return self.replay_count / self.queries if self.queries else 0.0

    @property
    def restore_rate(self) -> float:
        return self.restore_count / self.queries if self.queries else 0.0

    @property
    def verify_rate(self) -> float:
        return self.verify_count / self.queries if self.queries else 0.0

    @property
    def hit_rate(self) -> float:
        return self.memory_hits / self.queries if self.queries else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(0.95 * (len(sorted_lat) - 1))
        return sorted_lat[idx]

    def to_dict(self) -> dict:
        return {
            "queries": self.queries,
            "replay_rate": round(self.replay_rate, 4),
            "restore_rate": round(self.restore_rate, 4),
            "verify_rate": round(self.verify_rate, 4),
            "hit_rate": round(self.hit_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "action_counts": {
                "replay": self.replay_count,
                "restore": self.restore_count,
                "verify": self.verify_count,
                "none": self.none_count,
            },
        }


def run_benchmark(
    memory: Memory,
    queries: list[str],
    *,
    baseline_no_memory_ms: float | None = None,
) -> tuple[BenchmarkResult, dict]:
    """
    Run resolve() over queries and collect action distribution + latency.

    Returns (result, comparison) where comparison contrasts with a no-memory baseline.
    """
    result = BenchmarkResult()

    for query in queries:
        start = time.perf_counter()
        decision = memory.resolve(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result.queries += 1
        result.latencies_ms.append(elapsed_ms)

        if decision.action == MemoryAction.REPLAY:
            result.replay_count += 1
            result.memory_hits += 1
        elif decision.action == MemoryAction.RESTORE:
            result.restore_count += 1
            result.memory_hits += 1
        elif decision.action == MemoryAction.VERIFY:
            result.verify_count += 1
            result.memory_hits += 1
        else:
            result.none_count += 1

    # Only compare against a baseline the caller actually measured — we have
    # no honest way to invent an LLM round-trip time.
    comparison: dict = {"memory_hit_pct": round(result.hit_rate * 100, 1)}
    if baseline_no_memory_ms:
        with_memory_avg = result.avg_latency_ms
        comparison.update(
            {
                "user_measured_baseline_ms": baseline_no_memory_ms,
                "with_memory_avg_latency_ms": round(with_memory_avg, 2),
                "latency_reduction_pct": round(
                    max(0.0, (1 - with_memory_avg / baseline_no_memory_ms) * 100), 1
                ),
            }
        )

    return result, comparison


def format_benchmark_report(result: BenchmarkResult, comparison: dict) -> str:
    lines = [
        "Agent Memory Benchmark",
        "========================",
        f"Queries:           {result.queries}",
        f"Replay rate:       {result.replay_rate:.1%}",
        f"Restore rate:      {result.restore_rate:.1%}",
        f"Verify rate:       {result.verify_rate:.1%}",
        f"Memory hit rate:   {result.hit_rate:.1%}",
        f"Avg latency:       {result.avg_latency_ms:.2f} ms",
        f"P95 latency:       {result.p95_latency_ms:.2f} ms",
    ]
    if "user_measured_baseline_ms" in comparison:
        lines += [
            "",
            "Against your measured no-memory baseline",
            "----------------------------------------",
            f"Your baseline:     {comparison['user_measured_baseline_ms']:.0f} ms (user-supplied)",
            f"With agent-memory: {comparison['with_memory_avg_latency_ms']:.2f} ms",
            f"Latency reduction: {comparison['latency_reduction_pct']:.1f}% "
            "(applies to replayed queries only)",
        ]
    else:
        lines += [
            "",
            "Tip: pass --baseline-ms with your app's measured no-memory latency",
            "to see the comparison. No synthetic baseline is assumed.",
        ]
    return "\n".join(lines)
