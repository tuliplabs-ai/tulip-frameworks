# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""`gate_node` — gate a LangGraph node that performs a side effect directly.

`gate_node` is framework-agnostic (it never imports langgraph), so these run without
the langgraph extra installed. A gated node returns updated *state*, not an LLM tool
message, so soft-mode holds come back as a dict (serialize=False)."""

from __future__ import annotations

import pytest
from tulip.control import Action, AdmissionError, AuditTrail

from tulip_frameworks import action_gate_policy, is_held
from tulip_frameworks.langgraph import gate_node


def _contain_node(ran: list[str]):
    """A typical (state) -> state node that performs a side effect."""

    def contain(state: dict) -> dict:
        ran.append(state["host"])
        return {**state, "contained": True}

    return contain


async def test_node_allows_and_returns_state() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_node(
        _contain_node(ran),
        name="contain_host",
        action=lambda s: Action(name="contain_host", asset=s["host"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated({"host": "dev-1"})
    assert out["contained"] is True
    assert ran == ["dev-1"]
    assert trail.verify() is True


async def test_node_held_returns_dict_not_raise() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_node(
        _contain_node(ran),
        name="contain_host",
        action=lambda s: Action(name="contain_host", asset=s["host"], environment="production"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated({"host": "prod-db-01"})
    assert is_held(out)  # dict form, the graph can branch on it
    assert out["status"] == "held_for_approval"
    assert ran == []  # the node body never ran
    assert trail.verify() is True


async def test_node_constant_action() -> None:
    ran: list[str] = []
    gated = gate_node(
        _contain_node(ran),
        name="contain_host",
        action=Action(name="contain_host", asset="prod", environment="production"),
        policy=action_gate_policy(),
    )
    out = await gated({"host": "prod-1"})
    assert out["status"] == "held_for_approval"
    assert ran == []


async def test_node_raise_mode_propagates() -> None:
    ran: list[str] = []
    gated = gate_node(
        _contain_node(ran),
        name="contain_host",
        action=lambda s: Action(name="contain_host", asset=s["host"], environment="production"),
        policy=action_gate_policy(),
        mode="raise",
    )
    with pytest.raises(AdmissionError):
        await gated({"host": "prod-db-01"})
    assert ran == []
