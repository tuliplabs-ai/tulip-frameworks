# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""One governance case, shared by every framework's agent-loop test.

The per-framework tests differ only in how the agent is built and driven. What is
asserted is identical, so "the wrapper governs your framework" means the same thing
for all of them:

* an **allowed** tool call reaches the side effect;
* a **held** tool call does not — the side effect never runs;
* the decision is on the tamper-evident :class:`~tulip.control.AuditTrail`;
* the labels the gate weighed are the ones the *tool's action* declared.

That last point is enforced by construction. The policy holds on a bespoke tag
(``payment-rail``) and on nothing else — not on ``production``, not on blast radius.
:func:`tulip_frameworks.actions.default_action` cannot produce that tag, so a hold is
only reachable if the declared :class:`~tulip.control.Action` actually crossed into
the gate. If the bridge ever fell back to a default action, the held tests would
allow the refund and fail loudly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tulip.control import Action, AuditTrail

from tulip_frameworks.policy_presets import action_gate_policy

#: The label the policy holds on. Nothing derives it — the tool declares it.
HELD_TAG = "payment-rail"

#: Holds on ``HELD_TAG`` alone: no environment rule, and a blast-radius ceiling well
#: above anything these tests use. The ONLY way to a hold is the declared tag.
POLICY = action_gate_policy(require_human_for=(HELD_TAG,), deny_for=(), max_blast_radius=100)


def held_action(name: str, kwargs: Mapping[str, Any]) -> Action:
    """The action a production refund tool declares — carries ``HELD_TAG``."""
    return Action(
        name=name,
        asset=str(kwargs["order_id"]),
        blast_radius=1,
        kind="payment",
        environment="production",
        tags=frozenset({HELD_TAG}),
    )


def allowed_action(name: str, kwargs: Mapping[str, Any]) -> Action:
    """The same tool against a sandbox order — no ``HELD_TAG``, so it admits."""
    return Action(
        name=name,
        asset=str(kwargs["order_id"]),
        blast_radius=1,
        kind="payment",
        environment="sandbox",
        tags=frozenset({"reversible"}),
    )


def _admissions(trail: AuditTrail) -> list[Any]:
    return [r for r in trail.records() if r.event_type == "action-admission"]


def assert_recorded(trail: AuditTrail, *, outcome: str, asset: str, name: str = "refund") -> None:
    """Assert the gate's decision is on the trail, with the declared labels."""
    assert trail.verify() is True, "the audit chain does not verify"
    records = _admissions(trail)
    assert len(records) == 1, f"expected exactly one admission record, got {len(records)}"
    payload = records[0].payload
    assert payload["action"] == name, payload
    assert payload["asset"] == asset, payload
    assert payload["outcome"] == outcome, payload
    if outcome == "require_human":
        # The reason names the label that fired — proof the DECLARED tag reached the
        # policy rather than a default action's empty label set.
        assert HELD_TAG in payload["reason"], payload


def assert_held_payload(payload: Mapping[str, Any], *, asset: str) -> None:
    """Assert the structured result the agent was handed describes the hold."""
    assert payload["status"] == "held_for_approval", payload
    assert payload["outcome"] == "require_human", payload
    assert payload["asset"] == asset, payload
    assert HELD_TAG in payload["reason"], payload
