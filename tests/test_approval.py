# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Out-of-band approval — the `ApprovalBridge` protocol, the `gateway_bridge`
adapter, and how a held action carries an approval id."""

from __future__ import annotations

from collections.abc import Mapping

from tulip.control import Action

from tulip_frameworks import action_gate_policy, gate_callable
from tulip_frameworks.approval import ApprovalBridge, gateway_bridge
from tulip_frameworks.testing import Spy


class _Record:
    def __init__(self, approval_id: str, state: str) -> None:
        self.approval_id = approval_id
        self.state = state


class _Broker:
    """Stands in for the tulip-gateway ApprovalBroker (submit -> record, get -> record)."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, str, dict]] = []
        self._records: dict[str, _Record] = {}

    def submit(self, principal: str, tool: str, args: Mapping) -> _Record:
        self.submitted.append((principal, tool, dict(args)))
        rec = _Record("appr-9", "pending")
        self._records[rec.approval_id] = rec
        return rec

    def get(self, approval_id: str) -> _Record | None:
        return self._records.get(approval_id)


def test_gateway_bridge_unwraps_record() -> None:
    broker = _Broker()
    bridge = gateway_bridge(broker)
    approval_id = bridge.submit("agent", "refund", {"order_id": "ord-1"})
    assert approval_id == "appr-9"  # the record's approval_id, not the record
    assert bridge.state("appr-9") == "pending"
    assert bridge.state("does-not-exist") is None
    assert broker.submitted == [("agent", "refund", {"order_id": "ord-1"})]


def test_approval_bridge_is_structural() -> None:
    class Good:
        def submit(self, principal: str, tool: str, args: Mapping) -> str:
            return "id"

        def state(self, approval_id: str) -> str | None:
            return "pending"

    class MissingState:
        def submit(self, principal: str, tool: str, args: Mapping) -> str:
            return "id"

    assert isinstance(Good(), ApprovalBridge)
    assert not isinstance(MissingState(), ApprovalBridge)


async def test_hold_embeds_approval_id_and_next() -> None:
    broker = _Broker()
    gated = gate_callable(
        Spy(),
        name="disable_user",
        action=Action(
            name="disable_user", asset="m@corp", environment="production", kind="identity"
        ),
        policy=action_gate_policy(),
        principal="agent-7",
        approval=gateway_bridge(broker),
    )
    result = await gated(email="m@corp")
    assert result["status"] == "held_for_approval"
    assert result["approval_id"] == "appr-9"
    assert "approval_status" in result["next"]
    assert broker.submitted[0][0] == "agent-7"  # principal recorded


async def test_denied_action_offers_no_approval() -> None:
    broker = _Broker()
    gated = gate_callable(
        Spy(),
        name="wipe",
        action=Action(name="wipe", asset="db", tags=frozenset({"irreversible"})),
        policy=action_gate_policy(),
        approval=gateway_bridge(broker),
    )
    result = await gated()
    assert result["status"] == "denied"
    assert "approval_id" not in result  # a hard denial has no approval path
    assert broker.submitted == []
