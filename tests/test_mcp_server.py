"""Tests for the MCP server layer.

Exercises every exposed tool against bulk data and verifies the JSON reply
contract that MCP clients (Cursor, Claude Code, etc.) actually consume:
action, instruction, context payloads, pagination, and lifecycle operations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server import server

# Realistic knowledge base: (query, response, type, tags)
KNOWLEDGE_BASE = [
    ("How do I reset my password?",
     "Go to Settings → Security → Reset Password and follow the email link.",
     "conversation", ["auth", "faq"]),
    ("What payment methods do you support?",
     "We accept Visa, Mastercard, and PayPal.",
     "conversation", ["billing"]),
    ("What is the API rate limit?",
     "1000 requests/minute per API key.",
     "fact", ["api", "limits"]),
    ("How do we deploy to production?",
     "Merge to main, tag a release, and the GitHub Actions pipeline deploys automatically.",
     "workflow", ["deploy", "ci"]),
    ("What Python version does the project require?",
     "Python 3.10 or newer; see pyproject.toml.",
     "fact", ["python"]),
    ("How do I rotate the database credentials?",
     "Run scripts/rotate-creds.sh, then restart the API pods.",
     "workflow", ["database", "security"]),
    ("Where are the Grafana dashboards?",
     "https://grafana.internal/d/service-overview",
     "fact", ["monitoring"]),
    ("What is the on-call escalation policy?",
     "Page the secondary after 15 minutes without acknowledgement.",
     "fact", ["oncall"]),
]


@pytest.fixture
def mcp_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the MCP server at a fresh temp store and seed the knowledge base."""
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(tmp_path / "mcp_mem"))
    server.reset_memory()
    for query, response, memory_type, tags in KNOWLEDGE_BASE:
        server.remember_memory(query, response, tags=tags, memory_type=memory_type)
    yield server.get_memory()
    server.reset_memory()


def test_remember_returns_full_entry(mcp_memory) -> None:
    reply = json.loads(
        server.remember_memory(
            "What is the SLA?", "99.9% monthly uptime.", memory_type="fact", ttl="90d"
        )
    )
    assert reply["query"] == "What is the SLA?"
    assert reply["response"] == "99.9% monthly uptime."
    assert reply["type"] == "fact"
    assert reply["expires_at"] is not None
    assert reply["id"]


def test_resolve_replay_contract(mcp_memory) -> None:
    reply = json.loads(server.resolve_memory("How do I reset my password?"))
    assert reply["action"] == "replay"
    assert reply["response"].startswith("Go to Settings")
    assert reply["memory_id"]
    assert "instruction" in reply
    assert reply["confidence"] >= 0.85


def test_resolve_restore_contract(mcp_memory) -> None:
    reply = json.loads(server.resolve_memory("I forgot my password, what should I do?"))
    assert reply["action"] in ("restore", "verify")
    if reply["action"] == "restore":
        assert reply["context"], "restore reply must carry retrieved context"
        top = reply["context"][0]
        assert {"memory_id", "score", "prior_query", "prior_response"} <= set(top)
        assert "Reset Password" in reply["prompt_context"]


def test_resolve_none_contract_for_unrelated_query(mcp_memory) -> None:
    reply = json.loads(server.resolve_memory("What's the weather forecast for Tokyo?"))
    assert reply["action"] == "none"
    assert "instruction" in reply


def test_resolve_does_not_replay_cross_topic(mcp_memory) -> None:
    # Shares "support" with the payment memory but is a different question.
    reply = json.loads(
        server.resolve_memory("Does your platform support two-factor authentication?")
    )
    assert reply["action"] != "replay"
    assert reply.get("response") != "We accept Visa, Mastercard, and PayPal."


def test_resolve_verify_contract_for_stale_facts(mcp_memory) -> None:
    reply = json.loads(server.resolve_memory("What's the current API rate limit?"))
    assert reply["action"] in ("verify", "restore", "replay")
    if reply["action"] == "verify":
        assert reply["memory"]["response"] == "1000 requests/minute per API key."
        assert "Validate" in reply["instruction"]


def test_list_pagination(mcp_memory) -> None:
    page1 = json.loads(server.list_memories(limit=3, offset=0))
    page2 = json.loads(server.list_memories(limit=3, offset=3))
    assert page1["total_in_page"] == 3
    assert page2["total_in_page"] == 3
    ids1 = {m["id"] for m in page1["memories"]}
    ids2 = {m["id"] for m in page2["memories"]}
    assert ids1.isdisjoint(ids2)


def test_get_forget_archive_lifecycle(mcp_memory) -> None:
    entry = json.loads(server.remember_memory("temp entry", "temp response"))
    memory_id = entry["id"]

    fetched = json.loads(server.get_memory_by_id(memory_id))
    assert fetched["response"] == "temp response"

    archived = json.loads(server.archive_memory(memory_id))
    assert archived["archived"] is True
    # Archived memories must not resolve.
    reply = json.loads(server.resolve_memory("temp entry"))
    assert reply["action"] == "none"

    deleted = json.loads(server.forget_memory(memory_id))
    assert deleted["deleted"] is True
    missing = json.loads(server.get_memory_by_id(memory_id))
    assert "error" in missing


def test_get_missing_memory_returns_error(mcp_memory) -> None:
    reply = json.loads(server.get_memory_by_id("nonexistent-id"))
    assert "error" in reply


def test_consolidate_merges_duplicates(mcp_memory) -> None:
    server.remember_memory("How do I reset my password?", "Settings → Security → Reset.")
    reply = json.loads(server.consolidate_memories())
    assert "consolidated_count" in reply
    assert isinstance(reply["summaries"], list)


def test_bulk_roundtrip_100_memories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Store 100 memories through the MCP layer and verify every reply."""
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(tmp_path / "bulk_mem"))
    server.reset_memory()
    try:
        ids = []
        for i in range(100):
            reply = json.loads(
                server.remember_memory(
                    f"How do I configure service number {i}?",
                    f"Edit config/service-{i}.yaml and restart.",
                    tags=[f"service-{i}"],
                )
            )
            ids.append(reply["id"])
        assert len(set(ids)) == 100

        listed = json.loads(server.list_memories(limit=200))
        assert listed["total_in_page"] == 100

        # Exact queries replay their own answer, not a neighbor's.
        for i in (0, 42, 99):
            reply = json.loads(server.resolve_memory(f"How do I configure service number {i}?"))
            assert reply["action"] == "replay"
            assert f"service-{i}.yaml" in reply["response"]
    finally:
        server.reset_memory()
