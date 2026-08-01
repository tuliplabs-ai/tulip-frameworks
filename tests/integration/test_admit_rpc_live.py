# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The gate as a plain network call, against a REAL running Tulip gateway.

A real LangGraph agent holds a gated refund tool whose policy lives in **another
process**. Nothing about the decision is local: the wrapper marshals the action,
``POST /v1/admit`` weighs it server-side, and the refund runs only if that answer is
ALLOW. Then the hold is approved out of band by a human principal and claimed once —
the lifecycle a held action actually has to survive.

Skipped unless a gateway is reachable. Point it somewhere with::

    TULIP_GATEWAY_URL=http://127.0.0.1:8420 pytest tests/integration -q

Idempotent and safe against a shared, persistent stack: it only appends approvals of
its own (each keyed to a unique asset id) and never mutates anyone else's.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

GATEWAY_URL = os.environ.get("TULIP_GATEWAY_URL", "http://127.0.0.1:8420")
TENANT = os.environ.get("TULIP_GATEWAY_TENANT", "acme")
#: Bearer for an AUTHENTICATED gateway (e.g. the compose stack). Unset = the
#: README's local unauthenticated quickstart. The cases adapt: separation of
#: duties is a property of an authenticated deployment, so what is asserted
#: about the self-approval attempt differs by mode — and both modes are real.
TOKEN = os.environ.get("TULIP_GATEWAY_TOKEN", "")


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as response:  # noqa: S310
            return bool(200 <= response.status < 300)
    except (urllib.error.URLError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(GATEWAY_URL),
    reason=f"no Tulip gateway at {GATEWAY_URL} — set TULIP_GATEWAY_URL to run this",
)

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from tulip.control import Action, AuditTrail  # noqa: E402

from tulip_frameworks.gateway import RemotePolicy  # noqa: E402
from tulip_frameworks.langchain import gate_langchain_tool  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Deterministic model — the point of this test is the gate, not the LLM."""

    script: list[Any] = []
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        message = self.script[min(self.index, len(self.script) - 1)]
        object.__setattr__(self, "index", self.index + 1)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _agent(
    order_id: str, environment: str, executed: list[str]
) -> tuple[Any, AuditTrail, RemotePolicy]:
    @tool
    def refund(order_id: str) -> str:
        """Issue a customer refund for an order."""
        executed.append(order_id)
        return f"refunded {order_id}"

    trail = AuditTrail()
    policy = RemotePolicy(
        GATEWAY_URL,
        principal="agent:tulip-frameworks:test",
        tenant=TENANT,
        token=TOKEN or None,
    )
    gated = gate_langchain_tool(
        refund,
        action=lambda name, args: Action(
            name=name,
            asset=str(args["order_id"]),
            blast_radius=1,
            kind="payment",
            environment=environment,
        ),
        policy=policy,  # the decision is NOT in this process
        trail=trail,
    )
    model = ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "refund", "args": {"order_id": order_id}, "id": "call-1"}],
            ),
            AIMessage(content="finished"),
        ]
    )
    return create_react_agent(model, tools=[gated]), trail, policy


def _tool_content(out: dict[str, Any]) -> str:
    messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert messages, "the agent loop never dispatched the tool"
    return str(messages[0].content)


def _post(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{GATEWAY_URL}{path}",
        data=json.dumps(payload).encode(),
        # Tenancy is an authentication result: the gateway reads its own header
        # (or the verified token), never a caller-supplied query param.
        headers={
            "content-type": "application/json",
            "x-tulip-tenant": TENANT,
            **({"authorization": f"Bearer {TOKEN}"} if TOKEN else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return int(response.status), json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read() or b"{}")


async def test_remote_allow_lets_the_agent_act() -> None:
    executed: list[str] = []
    order_id = f"ord-dev-{uuid.uuid4().hex[:8]}"
    agent, trail, _ = _agent(order_id, "dev", executed)

    out = await agent.ainvoke({"messages": [("user", f"refund {order_id}")]})

    assert executed == [order_id], "the gateway allowed the action but it did not run"
    assert _tool_content(out) == f"refunded {order_id}"
    assert trail.records()[0].payload["outcome"] == "allow"
    assert trail.verify() is True


async def test_remote_hold_stops_the_side_effect_then_survives_the_human_lifecycle() -> None:
    executed: list[str] = []
    order_id = f"ord-prod-{uuid.uuid4().hex[:8]}"
    agent, trail, policy = _agent(order_id, "production", executed)

    out = await agent.ainvoke({"messages": [("user", f"refund {order_id}")]})

    # 1. The side effect did not happen — the decision was taken in another process.
    assert executed == [], f"a remotely-held refund executed anyway: {executed}"
    payload = json.loads(_tool_content(out))
    assert payload["status"] == "held_for_approval"
    assert payload["asset"] == order_id
    approval_id = payload["approval_id"]
    assert approval_id, "a remote hold must carry the gateway's approval_id"

    # 2. The decision is on the local tamper-evident trail with the same shape the
    #    in-process gate writes.
    assert trail.records()[0].payload["outcome"] == "require_human"
    assert trail.verify() is True

    # 3. The hold is live on the gateway and starts pending.
    assert await policy.approval_state(approval_id) == "pending"

    # 4. A human — a DIFFERENT principal — decides out of band.
    status, _ = _post(
        f"/v1/admit/approval/{approval_id}/decision",
        {
            "decision": "approve",
            "decided_by": "approver@local",
            # Mandatory since M-B: a verdict nobody explained is not a decision.
            "justification": "framework conformance: reviewed against the case",
        },
    )
    assert status == 200
    assert await policy.approval_state(approval_id) == "approved"

    # 5. The approval is single-use: claiming it twice must fail. (The native runtime
    #    shipped a bug where an approval was redeemable by any later call.)
    assert await policy.consume(approval_id) is True
    assert await policy.approval_state(approval_id) == "consumed"
    assert await policy.consume(approval_id) is False


async def test_the_agents_own_principal_cannot_approve_its_hold() -> None:
    """A requester must not be able to sign off on its own action.

    With auth OFF (the single-operator quickstart) the claimed name IS the
    identity, so claiming the agent's principal is refused outright. With auth
    ON the claimed name is worthless by construction: the gateway attributes
    the decision to the VERIFIED caller and the body string is ignored — the
    spoof does not even reach the rule. Both assertions are the same sentence
    enforced at different layers.
    """
    executed: list[str] = []
    order_id = f"ord-self-{uuid.uuid4().hex[:8]}"
    agent, _, policy = _agent(order_id, "production", executed)

    out = await agent.ainvoke({"messages": [("user", f"refund {order_id}")]})
    approval_id = json.loads(_tool_content(out))["approval_id"]

    status, body = _post(
        f"/v1/admit/approval/{approval_id}/decision",
        {
            "decision": "approve",
            "decided_by": "agent:tulip-frameworks:test",
            "justification": "the agent vouches for itself",
        },
    )

    if TOKEN:
        # Authenticated mode: the body's decided_by is IGNORED — the decision
        # is attributed to the verified caller, so the agent's name cannot be
        # spoofed onto it.
        assert status == 200, body
        assert body.get("decided_by") != "agent:tulip-frameworks:test"
        assert executed == []
    else:
        assert status == 403, f"the gateway let an agent approve its own action: {body}"
        assert await policy.approval_state(approval_id) == "pending"
        assert executed == []


# ── the same policy, the same verdict — natively and wrapped (M-D) ───────────


async def test_the_same_action_gets_the_same_verdict_natively_and_wrapped() -> None:
    """The C12 sentence, executed: one action, two callers, one answer.

    The NATIVE path is any agent POSTing the action straight to ``/v1/admit``;
    the WRAPPED path is a real LangGraph agent whose tool tulip-frameworks
    gated with a ``RemotePolicy``. Same server-side ``policy_ref``, so if the
    verdicts ever diverge, the wrapper is editorialising — which is exactly
    what it must never do.
    """
    for environment, expected in (("dev", "allow"), ("production", "require_human")):
        order_id = f"ord-parity-{environment}-{uuid.uuid4().hex[:8]}"

        # Native: the raw RPC, no framework anywhere.
        status, native = _post(
            "/v1/admit",
            {
                "principal": "agent:tulip-frameworks:test",
                "policy_ref": "default",
                "action": {
                    "name": "refund",
                    "asset": order_id,
                    "blast_radius": 1,
                    "environment": environment,
                    "kind": "payment",
                    "tags": [],
                },
            },
        )
        assert status == 200, native
        assert native["outcome"] == expected

        # Wrapped: the identical action declared on a LangGraph tool.
        executed: list[str] = []
        agent, trail, _ = _agent(order_id, environment, executed)
        out = await agent.ainvoke({"messages": [("user", f"refund {order_id}")]})

        wrapped_outcome = trail.records()[0].payload["outcome"]
        assert wrapped_outcome == native["outcome"], (
            f"{environment}: native said {native['outcome']!r}, "
            f"the wrapped framework got {wrapped_outcome!r} — same policy, same action"
        )
        if expected == "allow":
            assert executed == [order_id]
        else:
            assert executed == []
            assert json.loads(_tool_content(out))["status"] == "held_for_approval"


# ── a real CrewAI crew against the live gateway (M-D) ────────────────────────


def test_a_crewai_crew_is_governed_by_the_remote_gate() -> None:
    """CrewAI's own crew/agent/tool-dispatch loop; the decision in another process.

    Same shape as the LangGraph cases above: the crew is real, only the LLM is
    scripted; the wrapper marshals the declared Action to ``POST /v1/admit``
    and the side effect runs locally only on ALLOW. Sync on purpose: CrewAI
    refuses ``kickoff()`` from inside a running event loop, and the remote
    gate is driven through the bridge's own sync shim — the same path a real
    crew uses.
    """
    pytest.importorskip("crewai")
    from crewai import Agent as CrewAgent
    from crewai import Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import BaseTool

    from tulip_frameworks.crewai import gate_crewai_tool

    class ScriptedLLM(BaseLLM):
        def __init__(self, script: list[str]) -> None:
            super().__init__(model="scripted")
            self.script = script
            self.index = 0

        def call(self, messages: Any, **kwargs: Any) -> str:
            answer = self.script[min(self.index, len(self.script) - 1)]
            self.index += 1
            return answer

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

    def crew_for(order_id: str, environment: str, executed: list[str]) -> tuple[Crew, AuditTrail]:
        class Refund(BaseTool):
            name: str = "refund"
            description: str = "Issue a customer refund. Argument: order_id (string)."

            def _run(self, order_id: str) -> str:
                executed.append(order_id)
                return f"refunded {order_id}"

        trail = AuditTrail()
        gated = gate_crewai_tool(
            Refund(),
            action=lambda name, args: Action(
                name=name,
                asset=str(args["order_id"]),
                blast_radius=1,
                kind="payment",
                environment=environment,
            ),
            policy=RemotePolicy(
                GATEWAY_URL,
                principal="agent:tulip-frameworks:crewai",
                tenant=TENANT,
                token=TOKEN or None,
            ),
            trail=trail,
        )
        llm = ScriptedLLM(
            [
                'Thought: I will refund.\nAction: refund\nAction Input: {"order_id": "'
                + order_id
                + '"}',
                "Thought: done\nFinal Answer: handled",
            ]
        )
        agent = CrewAgent(
            role="support",
            goal="handle refunds",
            backstory="A support agent.",
            tools=[gated],
            llm=llm,
        )
        task = Task(description=f"Refund {order_id}", expected_output="confirmation", agent=agent)
        return Crew(agents=[agent], tasks=[task]), trail

    # ALLOW: the crew's dispatch reaches the side effect.
    executed: list[str] = []
    ok_id = f"ord-crew-dev-{uuid.uuid4().hex[:8]}"
    crew, trail = crew_for(ok_id, "dev", executed)
    crew.kickoff()
    assert executed == [ok_id], "the gateway allowed the action but the crew never ran it"
    assert trail.records()[0].payload["outcome"] == "allow"

    # HOLD: the decision came from another process and the side effect never ran.
    executed = []
    held_id = f"ord-crew-prod-{uuid.uuid4().hex[:8]}"
    crew, trail = crew_for(held_id, "production", executed)
    crew.kickoff()
    assert executed == [], f"a remotely-held refund executed anyway: {executed}"
    assert trail.records()[0].payload["outcome"] == "require_human"
    assert trail.verify() is True
