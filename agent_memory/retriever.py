from __future__ import annotations

from dataclasses import dataclass

from agent_memory.models import MemoryEntry, MemoryScope, RetrievalResult
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.store import MemoryStore


class MemoryRetriever:
    """Hybrid retrieval: BM25 keyword search + vector search + policy rerank."""

    def __init__(self, store: MemoryStore, policy: DecisionPolicy | None = None) -> None:
        self._store = store
        self._policy = policy or DefaultPolicy()

    @property
    def policy(self) -> DecisionPolicy:
        return self._policy

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> list[RetrievalResult]:
        vector_hits = self._store.search(query, top_k=top_k * 2, scopes=scopes)
        keyword_hits = self._store.keyword_search(query, top_k=top_k * 2, scopes=scopes)

        fused = self._reciprocal_rank_fusion(vector_hits, keyword_hits, k=60)

        results: list[RetrievalResult] = []
        for rank, (entry, semantic, keyword) in enumerate(fused[:top_k], start=1):
            final_score = self._policy.score(entry, semantic, keyword)
            results.append(
                RetrievalResult(
                    entry=entry,
                    semantic_score=semantic,
                    keyword_score=keyword,
                    final_score=final_score,
                    rank=rank,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        for index, result in enumerate(results, start=1):
            result.rank = index
        return results

    def retrieve_best(
        self,
        query: str,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> RetrievalResult | None:
        results = self.retrieve(query, top_k=1, scopes=scopes)
        return results[0] if results else None

    def record_access(self, entry: MemoryEntry) -> MemoryEntry:
        entry.touch()
        return self._store.update(entry)

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_hits: list[tuple[MemoryEntry, float]],
        keyword_hits: list[tuple[MemoryEntry, float]],
        *,
        k: int = 60,
    ) -> list[tuple[MemoryEntry, float, float]]:
        @dataclass
        class _Bucket:
            entry: MemoryEntry
            semantic: float = 0.0
            keyword: float = 0.0
            rrf: float = 0.0

        buckets: dict[str, _Bucket] = {}

        for rank, (entry, score) in enumerate(vector_hits, start=1):
            bucket = buckets.setdefault(entry.id, _Bucket(entry=entry))
            bucket.semantic = max(bucket.semantic, score)
            bucket.rrf += 1.0 / (k + rank)

        for rank, (entry, score) in enumerate(keyword_hits, start=1):
            bucket = buckets.setdefault(entry.id, _Bucket(entry=entry))
            bucket.keyword = max(bucket.keyword, score)
            bucket.rrf += 1.0 / (k + rank)

        fused = sorted(buckets.values(), key=lambda b: b.rrf, reverse=True)
        return [(b.entry, b.semantic, b.keyword) for b in fused]
