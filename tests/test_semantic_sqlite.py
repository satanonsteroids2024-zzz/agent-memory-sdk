"""Tests for the optional sqlite-vec semantic backend.

Skipped automatically when the ``semantic`` extra (sqlite-vec + fastembed /
sentence-transformers) is not installed, so the base test suite stays light.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from agent_memory import Memory, MemoryAction  # noqa: E402
from agent_memory.embeddings import get_default_embedder  # noqa: E402

if get_default_embedder() is None:
    pytest.skip("no embedding model installed", allow_module_level=True)


@pytest.fixture
def semantic_memory(tmp_path: Path) -> Memory:
    mem = Memory(persist_dir=tmp_path / "sem", enable_embeddings=True)
    assert mem.store.semantic_search_enabled  # type: ignore[union-attr]
    return mem


def test_paraphrase_with_no_shared_words(semantic_memory: Memory) -> None:
    """The lexical backend cannot do this; embeddings can."""
    semantic_memory.remember(
        "How do I reset my password?",
        "Go to Settings → Security → Reset Password.",
    )
    decision = semantic_memory.resolve(
        "I can't remember my login credentials, help me regain access"
    )
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.VERIFY)
    assert decision.context
    assert decision.context[0].entry.query == "How do I reset my password?"


def test_exact_match_still_replays(semantic_memory: Memory) -> None:
    semantic_memory.remember("What is the deploy freeze window?", "Fridays after 3pm UTC.")
    decision = semantic_memory.resolve("What is the deploy freeze window?")
    assert decision.action == MemoryAction.REPLAY
    assert decision.response == "Fridays after 3pm UTC."


def test_unrelated_query_returns_none(semantic_memory: Memory) -> None:
    semantic_memory.remember("What is the deploy freeze window?", "Fridays after 3pm UTC.")
    decision = semantic_memory.resolve("What is the capital city of Mongolia?")
    assert decision.action == MemoryAction.NONE


def test_vector_rows_follow_entry_lifecycle(semantic_memory: Memory) -> None:
    entry = semantic_memory.remember("lifecycle probe", "vector should be cleaned up")
    # Update re-embeds without violating the vec table's primary key.
    entry.response = "updated response"
    entry.content = "updated response"
    semantic_memory.store.update(entry)

    assert semantic_memory.forget(entry.id) is True
    decision = semantic_memory.resolve("lifecycle probe")
    assert decision.action == MemoryAction.NONE


def test_persists_and_backfills_across_instances(tmp_path: Path) -> None:
    persist = tmp_path / "reopen"
    m1 = Memory(persist_dir=persist, enable_embeddings=True)
    m1.remember("Where is the runbook for incident response?", "wiki/runbooks/incidents")

    m2 = Memory(persist_dir=persist, enable_embeddings=True)
    decision = m2.resolve("Where is the runbook for incident response?")
    assert decision.action == MemoryAction.REPLAY


def test_scope_filtering_applies_after_knn(semantic_memory: Memory) -> None:
    semantic_memory.remember("team secret", "team data", scope="team")
    semantic_memory.remember("user note", "user data", scope="user")
    results = semantic_memory.store.search("team secret", top_k=5, scopes=None)
    assert results
    from agent_memory.models import MemoryScope

    team_only = semantic_memory.store.search(
        "team secret", top_k=5, scopes=[MemoryScope.USER]
    )
    assert all(entry.scope == MemoryScope.USER for entry, _ in team_only)
