# tulip-frameworks

**Put Tulip's control gate around the actions your existing agent already takes —
without rebuilding it on Tulip.**

[Tulip](https://tulipagents.ai) is an open-source **agentic harness** — the
control runtime for agents that act: a consequential action runs only after your
policy clears it, waits for a person when it matters, and lands on a record you
can prove. `tulip-frameworks` lets you keep the agent you already have — in
**LangChain, LangGraph, CrewAI, the OpenAI Agents SDK, LlamaIndex, or Google
ADK** — and bring **just that control layer** to the tools it calls, without
rebuilding on the full Tulip SDK.

You wrap a tool once. From then on, when the agent decides to refund an order,
disable an account, or run a deploy, the gate decides whether that action is
allowed, held for a human, or denied — and writes the decision down either way.

---

## Why this exists

An agent that can only read is easy to trust. An agent that can *act* is not: the
thing deciding is a language model, and a bad retrieval, a confused chain of
thought, or a prompt injection can end with `refund(order, 1_000_000)`. A system
prompt is a guideline the model can ignore. Tulip's answer is a check in **real
code, outside the model**, between the decision and the side effect — the model
can be fooled; the action still doesn't execute unless your policy allows.

`tulip-frameworks` is how you get that check without leaving the framework you
already build in.

---

## Install

```bash
pip install "tulip-frameworks[langchain]"        # LangChain
pip install "tulip-frameworks[langgraph]"        # LangGraph
pip install "tulip-frameworks[openai-agents]"    # OpenAI Agents SDK
pip install "tulip-frameworks[crewai]"           # CrewAI
pip install "tulip-frameworks[llama-index]"      # LlamaIndex
pip install "tulip-frameworks[adk]"              # Google ADK
pip install "tulip-frameworks[all]"              # everything
```

`import tulip_frameworks` pulls in **no** framework package. Each bridge imports its
framework lazily and, if it's missing, tells you exactly which extra to install.

---

## Quickstart — LangChain

```python
from langchain_core.tools import tool
from tulip.control import Action, AuditTrail
from tulip_frameworks.langchain import gate_langchain_tool
from tulip_frameworks.policy_presets import action_gate_policy

@tool
def refund(order_id: str, amount_usd: float) -> str:
    "Issue a customer refund."
    return payments.refund(order_id, amount_usd)

trail = AuditTrail()

safe_refund = gate_langchain_tool(
    refund,
    # Describe the action's risk from the call's arguments.
    action=lambda name, a: Action(
        name=name, asset=a["order_id"],
        blast_radius=1, kind="payment", environment="production",
    ),
    policy=action_gate_policy(),   # production actions → held for a human
    trail=trail,
)

# Drop it into the agent exactly where `refund` went — same name, schema, description.
# agent = create_react_agent(model, tools=[safe_refund])
```

`safe_refund` is a real LangChain `StructuredTool`. Give it to your agent in place
of `refund`. Now:

- A refund in a non-production environment **runs** and is recorded.
- A **production** refund is **held for a human** — the function never executes;
  the agent gets back a structured "held for approval" result it can act on.
- Every decision lands on `trail`, which you can `trail.verify()` (tamper-evident —
  editing any record breaks verification) and `trail.export_jsonl()` (ship to a
  SIEM — a security team's log platform — or a warehouse).

Whatever leads the model to attempt a thousand production refunds — a bad
retrieval, a runaway loop, or a prompt injection — the result is a thousand
*held* actions and zero executed ones.

---

## The mental model

Every bridge is a thin wrapper over one primitive — `gate_callable` — which is the
only thing that actually calls the core SDK's `admit()`. Three inputs decide what
happens:

| Input | What it is | Who supplies it |
|---|---|---|
| **Action** | What the agent is about to do — its `asset`, `blast_radius`, `environment`, `kind`, `tags`. | You, as a constant `Action` or a `(name, kwargs) -> Action` callable that reads the call's arguments. |
| **Policy** | The rule. `action_gate_policy()` gates on labels + blast radius; or bring a full `ControlPolicy`. | You. |
| **Trail** | Where decisions are recorded, hash-chained so tampering is detectable. | Optional `AuditTrail`. |

The gate's output is one of three outcomes — **allow**, **require_human**, or
**deny** — and the side effect runs only on *allow*. The audit record is written
**before** a hold or denial surfaces, so a held action can never slip through as an
un-recorded side effect.

Deriving the `Action` from the arguments is the recommended form, because risk
usually depends on them — a refund of $5 in staging is not the same action as a
refund of $50,000 in production.

---

## Every framework, one line

The shape is identical across frameworks: pass the tool you already have, an
`action`, and a `policy`; get back a gated tool in that framework's native type.

```python
from tulip_frameworks.langchain     import gate_langchain_tool   # -> StructuredTool
from tulip_frameworks.langgraph      import gate_langchain_tool, gate_node
from tulip_frameworks.openai_agents  import gate_openai_tool      # -> FunctionTool
from tulip_frameworks.crewai         import gate_crewai_tool      # -> BaseTool
from tulip_frameworks.llamaindex     import gate_llamaindex_tool  # -> FunctionTool
from tulip_frameworks.adk            import gate_adk_tool         # -> FunctionTool
```

- **LangChain** — `gate_langchain_tool(tool, …)` returns a `StructuredTool` with the
  same name, description, and args schema. Drop-in.
- **LangGraph** — `ToolNode` consumes a gated LangChain tool unchanged, so
  `gate_langchain_tool` is re-exported here. For a raw graph **node** that performs a
  side effect directly, use `gate_node(node_fn, name=…, action=…, policy=…)`; its
  `action` reads the graph `state`.
- **OpenAI Agents SDK** — `gate_openai_tool(function_tool, …)` returns a
  `FunctionTool` that drops into `Agent(tools=[…])`.
- **CrewAI** — `gate_crewai_tool(tool, …)` returns a `BaseTool`. CrewAI runs tools
  synchronously; the bridge drives the async gate for you.
- **LlamaIndex** — `gate_llamaindex_tool(tool, …)` returns a `FunctionTool`.
- **Google ADK** — `gate_adk_tool(tool_or_fn, …)` returns a `FunctionTool` and
  preserves the original signature, so ADK builds the right function declaration.

---

## Held or raised — your choice

When an action doesn't admit, you pick how it surfaces with `mode`:

- **`mode="soft"`** (default) — the gated call returns a structured *held-for-approval*
  result the agent loop can read and react to (explain to the user, try a safer path,
  poll for the human decision). The run stays alive. For LLM-facing bridges the result
  is JSON, because a tool result has to be a string the model can see.

  ```python
  {"status": "held_for_approval", "outcome": "require_human",
   "action": "refund", "asset": "ord-9", "reason": "production action needs a human"}
  ```

- **`mode="raise"`** — the gate re-raises `AdmissionError` to stop a deterministic
  pipeline cold.

Either way, the decision is on the trail first.

---

## Out-of-band approval

A held action can be routed to a human on a side channel the agent can't reach.
Pass an `ApprovalBridge` and the held result carries an `approval_id` the agent can
poll while a person approves or denies elsewhere:

```python
safe_refund = gate_langchain_tool(refund, action=…, policy=…, trail=trail,
                                  approval=my_bridge)
# held result now includes: "approval_id": "appr-123",
#                           "next": "call approval_status(approval_id) once a human decides"
```

`ApprovalBridge` is a small structural `Protocol` (`submit` + `state`) with **no**
import-time dependency on any broker. The [`tulip-gateway`](https://tulipagents.ai)
approval broker satisfies it; `gateway_bridge(broker)` adapts one when you have it —
that adapter wraps a broker **object in your process**. For a gate on the other side
of the network, see below.

---

## The gate as a plain network call

By default the decision is taken in your process. Swap `policy=` for a `RemotePolicy`
and it moves to a gateway instead — the wrapper marshals the `Action`, `POST /v1/admit`
weighs it against a **server-side** policy, and the side effect runs locally only on
ALLOW. Nothing else about your agent changes:

```python
from tulip_frameworks.gateway import RemotePolicy

policy = RemotePolicy("http://gateway:8420", policy_ref="default",
                      principal="agent:support", tenant="acme")
safe_refund = gate_langchain_tool(refund, action=…, policy=policy, trail=trail)
```

Every bridge accepts either kind (`GatePolicy = ControlPolicy | RemotePolicy`). The
held result then carries the **gateway's** `approval_id`, and the lifecycle is driven
over the same connection:

```python
await policy.approval_state(approval_id)   # pending / approved / denied / consumed
await policy.consume(approval_id)          # single-use claim; False if already used
```

No policy logic ships from this side — `policy_ref` names a policy the gateway holds —
and the transport is stdlib `urllib`, so this adds no dependency. Pass `transport=` to
supply your own (custom auth, proxies, an HTTP client you already have).

> The gateway does **not** authenticate `/v1/admit` today, and the "a principal may not
> approve its own action" rule compares a caller-supplied `decided_by` string. Put the
> gateway on a trusted network until that changes.

---

## Test your gate offline — no LLM, no network

`tulip_frameworks.testing` lets you assert a tool admits or holds deterministically,
without running a model:

```python
from tulip_frameworks import gate_callable, action_gate_policy
from tulip_frameworks.testing import Spy, assert_allowed, assert_held
from tulip.control import Action

async def test_production_refund_is_held():
    spy = Spy()                      # stands in for the real side effect
    gated = gate_callable(
        spy, name="refund",
        action=Action(name="refund", environment="production", kind="payment"),
        policy=action_gate_policy(),
    )
    result = await gated(order_id="ord-9")
    assert_held(result, spy)         # the side effect never ran; result is "held"
```

This is the same pattern the package's own unit tests use, so the behaviour you
assert in CI is the behaviour your agent gets in production.

---

## Three ways Tulip meets the ecosystem (don't conflate them)

This package is for the **first** one. It helps to see all three, because the same
names people call "integrations" are actually three different relationships:

- **Gate** — agent *frameworks* whose tools take actions (LangChain, LangGraph,
  CrewAI, the OpenAI Agents SDK, LlamaIndex, Google ADK). This package's `gate_*_tool`.
- **Compose** — *model-call* gateways (LiteLLM, Portkey): they route the model call;
  Tulip gates the action. They stack, they don't compete — see
  [`examples/litellm_two_layer.py`](examples/litellm_two_layer.py).
- **Assure** — *another agent* you don't control (a chatbot, an endpoint, an
  OpenClaw-style runtime): red-team it as a `Target` with the core SDK's `red_team()`.

---

## Examples

Runnable, per-framework, under [`examples/`](examples/):

| File | Shows |
|---|---|
| [`langchain_refund_gate.py`](examples/langchain_refund_gate.py) | Gate a refund tool; production held for a human |
| [`langgraph_soc_containment.py`](examples/langgraph_soc_containment.py) | Gate a graph node that performs a side effect |
| [`crewai_account_gate.py`](examples/crewai_account_gate.py) | Gate a CrewAI tool (sync execution model) |
| [`adk_account_gate.py`](examples/adk_account_gate.py) | Gate a Google ADK FunctionTool |
| [`litellm_two_layer.py`](examples/litellm_two_layer.py) | Compose: LiteLLM routes the model, Tulip gates the action |

---

## Status

v0.1 ships gate bridges for **LangChain, LangGraph, the OpenAI Agents SDK, CrewAI,
LlamaIndex, and Google ADK**, plus the framework-agnostic core, an out-of-band
approval protocol, and offline testing helpers. The LangChain/LangGraph and OpenAI
Agents bridges are the most exercised, including end-to-end tests against a real LLM;
CrewAI, LlamaIndex, and ADK follow the identical pattern.

One-way dependency on `tulip-agents` (the same direction as the
langchain-core / langchain-community split). Apache-2.0.

→ Full docs: **<https://tulipagents.ai/integrations/frameworks/>** ·
Core SDK: **<https://github.com/tuliplabs-ai/sdk-python>**
