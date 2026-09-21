"""Regression tests for decision quality: partially-matching queries must not replay.

These pin the fix for the score-inflation bug where the top BM25 hit was
always normalized to 1.0, causing REPLAY of unrelated answers.
"""

from pathlib import Path

import pytest

from agent_memory import Memory, MemoryAction


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(persist_dir=tmp_path / "mem")


def test_single_shared_word_does_not_replay(memory: Memory) -> None:
    memory.remember(
        "How do I reset my password?",
        "Go to Settings → Security → Reset Password.",
        confidence=0.95,
    )
    memory.remember(
        "What payment methods do you support?",
        "We accept Visa, Mastercard, and PayPal.",
    )

    # Shares only "support" with the payment memory.
    decision = memory.resolve("Does your platform support two-factor authentication?")
    assert decision.action != MemoryAction.REPLAY
    assert decision.response != "We accept Visa, Mastercard, and PayPal."

    # Shares only "password" with the reset memory.
    decision = memory.resolve("Why is my password rejected as too weak when signing up?")
    assert decision.action != MemoryAction.REPLAY


def test_low_confidence_memory_is_not_replayed(memory: Memory) -> None:
    memory.remember("What is the refund window?", "14 days.", confidence=0.1)
    decision = memory.resolve("What is the refund window?")
    assert decision.action != MemoryAction.REPLAY


def test_exact_match_still_replays(memory: Memory) -> None:
    memory.remember("What is the refund window?", "14 days.")
    decision = memory.resolve("What is the refund window?")
    assert decision.action == MemoryAction.REPLAY
    assert decision.response == "14 days."


def test_replay_does_not_refresh_recency(memory: Memory) -> None:
    entry = memory.remember("What is the refund window?", "14 days.")
    original_updated_at = entry.updated_at

    decision = memory.resolve("What is the refund window?")
    assert decision.action == MemoryAction.REPLAY

    fetched = memory.get(entry.id)
    assert fetched is not None
    assert fetched.access_count >= 1
    assert fetched.updated_at == original_updated_at


def test_paraphrase_restores_as_context(memory: Memory) -> None:
    memory.remember(
        "How do I reset my password?",
        "Go to Settings → Security → Reset Password.",
    )
    decision = memory.resolve("I forgot my password, what should I do?")
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.VERIFY)
