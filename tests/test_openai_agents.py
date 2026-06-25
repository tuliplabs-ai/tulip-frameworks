# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Agents SDK bridge — runs only when the ``openai-agents`` extra is installed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agents")

from agents import function_tool  # noqa: E402
from tulip.security import Action, AuditTrail  # noqa: E402

from tulip_frameworks.openai_agents import gate_openai_tool  # noqa: E402
from tulip_frameworks.policy_presets import action_gate_policy  # noqa: E402


def _disable_tool(ran: list[str]):
    @function_tool
    def disable_user(email: str) -> str:
        """Disable an account."""
        ran.append(email)
        return f"disabled {email}"

    return disable_user


async def test_allow_runs() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_openai_tool(
        _disable_tool(ran),
        action=lambda n, a: Action(name=n, asset=a["email"], environment="dev", kind="identity"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated.on_invoke_tool(None, json.dumps({"email": "u@corp"}))
    assert "disabled u@corp" in out
    assert ran == ["u@corp"]
    assert trail.verify() is True


async def test_production_held() -> None:
    ran: list[str] = []
    gated = gate_openai_tool(
        _disable_tool(ran),
        action=lambda n, a: Action(
            name=n, asset=a["email"], environment="production", kind="identity"
        ),
        policy=action_gate_policy(),
    )
    out = await gated.on_invoke_tool(None, json.dumps({"email": "m@corp"}))
    payload = json.loads(out)
    assert payload["status"] == "held_for_approval"
    assert ran == []
