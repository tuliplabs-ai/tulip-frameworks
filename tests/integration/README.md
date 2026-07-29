# Integration tests — real frameworks, real LLM

These exercise the bridges end-to-end: a real agent, built on a real framework,
driven by a **real model**, decides to take an action — and we assert Tulip's gate
holds it. No mocks. This is the test that proves the claim "the model can be fooled;
it still can't get the action executed."

They are **skipped by default** and only run when both the framework extra and an
API key are present, so they never break a plain `pytest` or a keyless CI:

- `test_langchain_live.py` — needs `langchain-anthropic` + `langgraph` and
  `ANTHROPIC_API_KEY`.
- `test_openai_agents_live.py` — needs `openai-agents` and `OPENAI_API_KEY`.

## Run them

```bash
# from the repo root, with the core SDK importable (tulip-agents>=2)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
pip install "tulip-frameworks[langchain,langgraph,openai-agents]" langchain-anthropic
pytest tests/integration -q
```

Override the models with `TULIP_TEST_ANTHROPIC_MODEL` / `TULIP_TEST_OPENAI_MODEL`
(defaults: a small, cheap model on each provider).

## `test_admit_rpc_live.py` — the gate over the wire, no key needed

Unlike the two above, this one needs **no API key** — the model is scripted; what is
real is the *gateway*. A LangGraph agent holds a tool whose `policy=` is a
`RemotePolicy`, so the decision is taken by `POST /v1/admit` in another process. It
asserts an allowed call runs, a held call does not, the hold carries the gateway's
`approval_id`, a human on a different principal can approve it, the claim is
single-use, and the agent's own principal is refused when it tries to approve itself.

It is skipped unless a gateway answers `/health`:

```bash
TULIP_GATEWAY_URL=http://127.0.0.1:8420 TULIP_GATEWAY_TENANT=acme pytest tests/integration -q
```

It is idempotent and safe against a shared, persistent stack: each run uses fresh
asset ids and only touches approvals it opened itself.
