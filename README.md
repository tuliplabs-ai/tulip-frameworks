# tulip-frameworks

**Drop Tulip's control runtime into the agent framework you already use.**

Tulip is the SDK for *agents you can let act* — every risky action policy-gated,
human-approvable, and recorded in a tamper-evident audit trail. You don't have to
rebuild your agent on Tulip to get that. If you already have an agent in
**LangChain / LangGraph, CrewAI, the OpenAI Agents SDK, or LlamaIndex**, keep it —
and wrap the *actions* it takes with Tulip's admission gate.

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
    action=lambda name, a: Action(name=name, asset=a["order_id"],
        blast_radius=1, kind="payment", environment="production"),
    policy=action_gate_policy(),   # production → human
    trail=trail,
)
# agent = create_react_agent(model, tools=[safe_refund])
```

Now a production refund is **held for a human**, the decision is on a hash-chained
trail you can `verify()` and `export_jsonl()`, and a prompt injection that talks the
model into a thousand refunds still can't get one *executed*. Fool the model; you
can't talk past the runtime.

## How it works

Every bridge is a thin wrapper over one primitive, `gate_callable`, which is the only
thing that calls the core SDK's `admit()`:

- **Action derivation** — pass a constant `Action`, or a callable
  `(name, kwargs) -> Action` that derives blast radius / environment / kind from the
  call's arguments.
- **Policy** — `action_gate_policy()` gates on labels + blast radius (use this for
  ordinary tools); or bring a full `ControlPolicy` and pass a grounded `finding`
  (`Evidence`) / `verdict` (`VerificationResult`) for the complete trust chain.
- **Soft vs raise** — `mode="soft"` (default) returns an LLM-readable
  *held-for-approval* result the agent loop can react to; `mode="raise"` re-raises
  `AdmissionError` to stop a deterministic pipeline. The audit record is written
  **before** either, so a held action can never become an un-audited side effect.
- **Out-of-band approval** — optionally pass an `ApprovalBridge` (the `tulip-gateway`
  broker satisfies it) to submit holds for a human to approve on a side channel.

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

`import tulip_frameworks` pulls in **no** framework package; each bridge imports its
framework lazily and tells you which extra to install if it's missing.

## Three ways Tulip meets the ecosystem (don't conflate them)

This package is for the **first** one — but it helps to see all three, because the same
five "integrations" people name are actually three different relationships:

- **Gate** — agent *frameworks* whose tools take actions: **LangChain, LangGraph, CrewAI,
  the OpenAI Agents SDK, LlamaIndex, Google ADK.** This package's `gate_*_tool`.
- **Compose** — *model-call* gateways (**LiteLLM**, Portkey): they route the model call;
  Tulip gates the action. They stack, they don't compete. See
  [`examples/litellm_two_layer.py`](examples/litellm_two_layer.py).
- **Assure** — *another agent* you don't control (a chatbot, an OpenClaw-style runtime, an
  endpoint): red-team it as a `Target` with the core SDK's `red_team()`.

## Status

v0.1 ships gate bridges for **LangChain, LangGraph, the OpenAI Agents SDK, CrewAI,
LlamaIndex, and Google ADK**, plus the framework-agnostic core. The LangChain/LangGraph
and OpenAI Agents bridges are the most exercised; CrewAI, LlamaIndex, and ADK follow the
identical pattern.

One-way dependency on `tulip-agents` (the langchain-core / langchain-community split).
Apache-2.0. See [the docs](https://tulipagents.ai/integrations/frameworks/).
