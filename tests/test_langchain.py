# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""LangChain bridge — runs only when the ``langchain`` extra is installed.

No LLM is involved: we invoke the gated tool directly and assert the gate decision.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool  # noqa: E402
from tulip.security import Action, AuditTrail  # noqa: E402

from tulip_frameworks.langchain import gate_langchain_tool  # noqa: E402
from tulip_frameworks.policy_presets import action_gate_policy  # noqa: E402


def _refund_tool(ran: list[str]):
    @tool
    def refund(order_id: str) -> str:
        """Issue a customer refund."""
        ran.append(order_id)
        return f"refunded {order_id}"

    return refund


async def test_gated_tool_preserves_metadata() -> None:
    gated = gate_langchain_tool(
        _refund_tool([]),
        action=lambda n, a: Action(name=n, asset=a["order_id"], environment="dev"),
        policy=action_gate_policy(),
    )
    assert gated.name == "refund"
    assert "refund" in gated.description.lower()


async def test_allow_runs() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_langchain_tool(
        _refund_tool(ran),
        action=lambda n, a: Action(name=n, asset=a["order_id"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated.ainvoke({"order_id": "ord-1"})
    assert out == "refunded ord-1"
    assert ran == ["ord-1"]
    assert trail.verify() is True


async def test_production_held() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_langchain_tool(
        _refund_tool(ran),
        action=lambda n, a: Action(
            name=n, asset=a["order_id"], environment="production", kind="payment"
        ),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated.ainvoke({"order_id": "ord-9"})
    payload = json.loads(out)  # serialize=True -> JSON string
    assert payload["status"] == "held_for_approval"
    assert ran == []  # refund did NOT run
    assert trail.verify() is True
