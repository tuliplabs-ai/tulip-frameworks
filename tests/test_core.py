# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Core gate behaviour — offline, no framework, no LLM."""

from __future__ import annotations

import json

import pytest
from tulip.security import Action, AdmissionError, AuditTrail

from tulip_frameworks import action_gate_policy, gate_callable
from tulip_frameworks.testing import Spy, assert_allowed, assert_held


async def test_allow_runs_and_audits() -> None:
    spy = Spy(result="contained")
    trail = AuditTrail()
    gated = gate_callable(
        spy,
        name="isolate_host",
        action=Action(name="isolate_host", asset="dev-1", environment="dev", kind="containment"),
        policy=action_gate_policy(),  # dev is not in require_human_for -> allow
        trail=trail,
    )
    result = await gated(host="dev-1")
    assert_allowed(result, spy)
    assert spy.calls == [{"host": "dev-1"}]
    assert trail.verify() is True


async def test_production_holds_for_human_soft() -> None:
    spy = Spy()
    trail = AuditTrail()
    gated = gate_callable(
        spy,
        name="isolate_host",
        action=lambda name, a: Action(
            name=name, asset=a["host"], environment="production", kind="containment"
        ),
        policy=action_gate_policy(),  # production -> require_human
        trail=trail,
        mode="soft",
    )
    result = await gated(host="prod-db-01")
    assert_held(result, spy)  # did NOT run; status held_for_approval
    assert result["asset"] == "prod-db-01"
    assert trail.verify() is True  # the hold is on the record


async def test_deny_label_denies() -> None:
    spy = Spy()
    gated = gate_callable(
        spy,
        name="wipe",
        action=Action(name="wipe", asset="db", tags=frozenset({"irreversible"})),
        policy=action_gate_policy(),  # 'irreversible' is in deny_for
        mode="soft",
    )
    result = await gated()
    assert_held(result, spy, outcome="denied")
    assert "approval_id" not in result  # a denial offers no approval path


async def test_raise_mode_propagates() -> None:
    spy = Spy()
    trail = AuditTrail()
    gated = gate_callable(
        spy,
        name="refund",
        action=Action(name="refund", asset="c1", environment="production", kind="payment"),
        policy=action_gate_policy(),
        trail=trail,
        mode="raise",
    )
    with pytest.raises(AdmissionError):
        await gated(order_id="c1")
    assert not spy.ran
    assert trail.verify() is True  # admit() recorded the decision before raising


async def test_serialize_returns_json_string() -> None:
    spy = Spy()
    gated = gate_callable(
        spy,
        name="refund",
        action=Action(name="refund", environment="production", kind="payment"),
        policy=action_gate_policy(),
        mode="soft",
        serialize=True,
    )
    result = await gated()
    assert isinstance(result, str)
    payload = json.loads(result)
    assert payload["status"] == "held_for_approval"


async def test_approval_bridge_embeds_id() -> None:
    class _Broker:
        def __init__(self) -> None:
            self.submitted: list[tuple[str, str]] = []

        def submit(self, principal: str, tool: str, args: dict) -> str:
            self.submitted.append((principal, tool))
            return "appr-123"

        def state(self, approval_id: str) -> str | None:
            return "pending"

    broker = _Broker()
    gated = gate_callable(
        Spy(),
        name="disable_user",
        action=Action(name="disable_user", environment="production", kind="identity"),
        policy=action_gate_policy(),
        principal="agent-7",
        approval=broker,
    )
    result = await gated(email="m@corp")
    assert result["approval_id"] == "appr-123"
    assert broker.submitted == [("agent-7", "disable_user")]
