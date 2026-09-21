# Contributing to Agent Memory

Thanks for considering a contribution! This project is young, issues get
answered quickly, and there is real, well-scoped work available.

## Good first issues

These are genuinely useful and don't require touching the decision engine:

- **Eval cases**: add adversarial cases to `benchmarks/datasets/` — queries
  that share words with a stored memory but mean something different. Every
  trap case that survives review makes the eval suite more credible.
- **Framework adapters**: a LangChain `BaseMemory` or LlamaIndex wrapper
  around `Memory.resolve()` / `Memory.remember()`.
- **External benchmark harness**: a runner that seeds LongMemEval or LoCoMo
  conversations and scores `resolve()` decisions against them.
- **Docs**: a worked example for a domain you know (support bot, coding
  agent, research assistant).

Open a [Discussion](https://github.com/TheProdSDE/agent-memory-sdk/discussions)
first for anything bigger than a file or two.

## Development setup

```bash
git clone https://github.com/TheProdSDE/agent-memory-sdk.git
cd agent-memory-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # base install
pip install -e ".[dev,semantic]"  # + vector search (sqlite-vec, fastembed)
pre-commit install
```

## Before you open a PR

All four must pass — CI enforces them (mypy is not advisory):

```bash
ruff check agent_memory/ mcp_server/ tests/
mypy agent_memory/ mcp_server/
python -m pytest tests/
agent-memory --data-dir /tmp/eval eval   # decision quality must stay at 100%
```

Guidelines:

- **Decision quality over recall.** A change that makes more queries match
  is wrong if it lets a trap case replay the wrong answer. When in doubt,
  add the trap case to `benchmarks/datasets/decision_traps.json` first.
- **Both backends.** Retrieval changes need to pass with and without the
  `semantic` extra (`pip uninstall sqlite-vec fastembed` to test the
  lexical path).
- **No synthetic numbers.** Anything in docs/benchmarks.md must be
  reproducible by the commands shown next to it.
- Tests live next to the behavior they pin: `tests/test_decision_quality.py`
  for scoring, `tests/test_scale.py` for performance regressions,
  `tests/test_mcp_server.py` for the MCP reply contract.

## Release process

Maintainers: push a `v*` tag; the release workflow builds, tests, publishes
to PyPI, and creates the GitHub Release. See the README's Release Process
section.
