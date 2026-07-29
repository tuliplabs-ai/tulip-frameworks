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
        headers={"content-type": "application/json"},
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
        f"/v1/admit/approval/{approval_id}/decision?tenant={TENANT}",
        {"decision": "approve", "decided_by": "approver@local"},
    )
    assert status == 200
    assert await policy.approval_state(approval_id) == "approved"

    # 5. The approval is single-use: claiming it twice must fail. (The native runtime
    #    shipped a bug where an approval was redeemable by any later call.)
    assert await policy.consume(approval_id) is True
    assert await policy.approval_state(approval_id) == "consumed"
    assert await policy.consume(approval_id) is False


async def test_the_agents_own_principal_cannot_approve_its_hold() -> None:
    """A requester must not be able to sign off on its own action."""
    executed: list[str] = []
    order_id = f"ord-self-{uuid.uuid4().hex[:8]}"
    agent, _, policy = _agent(order_id, "production", executed)

    out = await agent.ainvoke({"messages": [("user", f"refund {order_id}")]})
    approval_id = json.loads(_tool_content(out))["approval_id"]

    status, body = _post(
        f"/v1/admit/approval/{approval_id}/decision?tenant={TENANT}",
        {"decision": "approve", "decided_by": "agent:tulip-frameworks:test"},
    )

    assert status == 403, f"the gateway let an agent approve its own action: {body}"
    assert await policy.approval_state(approval_id) == "pending"
    assert executed == []
