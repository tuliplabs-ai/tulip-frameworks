# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""CrewAI bridge — runs only when the ``crewai`` extra is installed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("crewai")

from crewai.tools import BaseTool  # noqa: E402
from tulip.control import (
    Action,
    AuditTrail,  # noqa: E402
)

from tulip_frameworks.crewai import gate_crewai_tool  # noqa: E402
from tulip_frameworks.policy_presets import action_gate_policy  # noqa: E402


def _refund_tool(ran: list[str]) -> BaseTool:
    class Refund(BaseTool):
        name: str = "refund"
        description: str = "Issue a customer refund."

        def _run(self, order_id: str) -> str:
            ran.append(order_id)
            return f"refunded {order_id}"

    return Refund()


def test_allow_runs() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_crewai_tool(
        _refund_tool(ran),
        action=lambda n, a: Action(name=n, asset=a["order_id"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = gated._run(order_id="ord-1")
    assert out == "refunded ord-1"
    assert ran == ["ord-1"]
    assert trail.verify() is True


def test_production_held() -> None:
    ran: list[str] = []
    gated = gate_crewai_tool(
        _refund_tool(ran),
        action=lambda n, a: Action(
            name=n, asset=a["order_id"], environment="production", kind="payment"
        ),
        policy=action_gate_policy(),
    )
    payload = json.loads(gated._run(order_id="ord-9"))
    assert payload["status"] == "held_for_approval"
    assert ran == []
