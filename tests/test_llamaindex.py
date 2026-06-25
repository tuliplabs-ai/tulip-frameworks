# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex bridge — runs only when the ``llama-index`` extra is installed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.tools import FunctionTool  # noqa: E402
from tulip.control import (
    Action,
    AuditTrail,  # noqa: E402
)

from tulip_frameworks.llamaindex import gate_llamaindex_tool  # noqa: E402
from tulip_frameworks.policy_presets import action_gate_policy  # noqa: E402


def _refund_tool(ran: list[str]) -> FunctionTool:
    def refund(order_id: str) -> str:
        """Issue a customer refund."""
        ran.append(order_id)
        return f"refunded {order_id}"

    return FunctionTool.from_defaults(fn=refund, name="refund")


async def test_allow_runs() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_llamaindex_tool(
        _refund_tool(ran),
        action=lambda n, a: Action(name=n, asset=a["order_id"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated.async_fn(order_id="ord-1")
    assert out == "refunded ord-1"
    assert ran == ["ord-1"]
    assert trail.verify() is True


async def test_production_held() -> None:
    ran: list[str] = []
    gated = gate_llamaindex_tool(
        _refund_tool(ran),
        action=lambda n, a: Action(
            name=n, asset=a["order_id"], environment="production", kind="payment"
        ),
        policy=action_gate_policy(),
    )
    payload = json.loads(await gated.async_fn(order_id="ord-9"))
    assert payload["status"] == "held_for_approval"
    assert ran == []
