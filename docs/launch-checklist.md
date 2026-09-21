# Launch checklist

Everything below is prepared in this repo; each step needs your account and
is one command or one click. Order matters roughly top to bottom.

## 0. Commit and push the pending work

The demo GIF, `server.json`, README updates, and blog post are in the
working tree — review, commit, push. The blog post is in the website repo:
`theprodsde.github.io/content/blog/agent-memory-decision-layer.mdx`.

## 1. Enable GitHub Discussions — ✅ DONE (2026-09-21)

Discussions are enabled and the roadmap thread is live:
https://github.com/theprodsde/agent-memory-sdk/discussions/1

Remaining manual click: open that discussion and choose **Pin discussion**
in the right sidebar (GitHub removed the API for pinning discussions).

Starter issues are also created and labeled (#2 trap cases, #3 LangChain
adapter, #4 LongMemEval harness, #5 worked example).

## 2. Publish to the MCP Registry

`server.json` at the repo root is already validated against the official
2025-09-29 schema, and the `agent-memory-sdk` console script exists so the
registry's default `uvx agent-memory-sdk` launch works.

Prerequisite: the version in `server.json` must exist on PyPI, and the
registry verifies PyPI ownership via a README marker — make sure the README
on the published PyPI version contains the string:
`mcp-name: io.github.theprodsde/agent-memory`
(add it near the bottom of README.md, cut a release, then publish).

```bash
brew install mcp-publisher
mcp-publisher login github        # authenticates io.github.theprodsde/* namespace
mcp-publisher publish             # reads ./server.json
```

Docs: https://github.com/modelcontextprotocol/registry/blob/main/docs/guides/publishing/publish-server.md

## 3. Publish the blog post

The post is written in your site's format at
`content/blog/agent-memory-decision-layer.mdx`. Push the site repo and it
goes live at `/blog/agent-memory-decision-layer`.

## 4. Show HN / r/LocalLLaMA

Suggested Show HN title (under 80 chars):

> Show HN: Agent Memory – memory for AI agents that knows when NOT to answer

Body: link the blog post, lead with the trap-query example, mention
"25/25 on adversarial eval cases, reproducible with `agent-memory eval`",
and invite people to submit trap cases that break it. Post the same piece to
r/LocalLLaMA with flair "Resources". Best posting windows: Tue–Thu,
14:00–16:00 UTC.

## 5. After launch

- Respond to the first issues/comments within hours — early responsiveness
  is the strongest liveness signal.
- Label 3–4 issues `good first issue` from CONTRIBUTING.md's list (eval trap
  cases, LangChain adapter, LongMemEval harness, worked examples).
- Add the GIF and registry badge to the PyPI page by cutting a release with
  the updated README.
