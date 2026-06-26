# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""`action_gate_policy` — the preset that gates ordinary tools on labels + blast
radius rather than a verification score. Exercised end-to-end through the gate so
the assertions reflect the real admission engine, not the preset in isolation."""

from __future__ import annotations

from tulip.control import Action

from tulip_frameworks import action_gate_policy, gate_callable
from tulip_frameworks.testing import Spy


async def _run(action: Action, policy=None, **kwargs):
    """Gate a Spy with ``action``/``policy`` and report (ran, result)."""
    spy = Spy(result="done")
    gated = gate_callable(
        spy, name=action.name, action=action, policy=policy or action_gate_policy()
    )
    result = await gated(**kwargs)
    return spy.ran, result


def test_preset_fields() -> None:
    policy = action_gate_policy()
    # The whole point of the preset: verification is optional (score 0), so an
    # un-verified tool call is NOT auto-held the way the stock SOC policy would.
    assert policy.require_verification_score == 0.0
    assert policy.max_blast_radius == 1
    assert "production" in policy.require_human_for
    assert "irreversible" in policy.deny_for


async def test_low_risk_action_allows() -> None:
    ran, _ = await _run(Action(name="read", asset="x", blast_radius=1, environment="dev"))
    assert ran is True  # dev, single asset, no deny label -> runs


async def test_production_label_holds() -> None:
    ran, result = await _run(Action(name="refund", asset="ord-1", environment="production"))
    assert ran is False
    assert result["status"] == "held_for_approval"
    assert result["outcome"] == "require_human"


async def test_deny_label_denies() -> None:
    ran, result = await _run(Action(name="wipe", asset="db", tags=frozenset({"irreversible"})))
    assert ran is False
    assert result["status"] == "denied"
    assert result["outcome"] == "deny"


async def test_blast_radius_over_max_holds() -> None:
    ran, result = await _run(
        Action(name="rotate", asset="fleet", blast_radius=5, environment="dev")
    )
    assert ran is False
    assert result["status"] == "held_for_approval"
    assert "blast radius" in result["reason"].lower()


async def test_custom_require_human_label() -> None:
    # Gate any "payment" action on a human, regardless of environment.
    policy = action_gate_policy(require_human_for=("payment",))
    ran, result = await _run(
        Action(name="pay", asset="x", kind="payment", environment="dev"), policy
    )
    assert ran is False
    assert result["outcome"] == "require_human"


async def test_custom_max_blast_radius_raises_ceiling() -> None:
    policy = action_gate_policy(max_blast_radius=10)
    ran, _ = await _run(
        Action(name="rotate", asset="fleet", blast_radius=5, environment="dev"), policy
    )
    assert ran is True  # 5 <= 10 now, and dev is not a hold label
