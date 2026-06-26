# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Google ADK bridge — runs only when the ``adk`` extra is installed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("google.adk")

from google.adk.tools import FunctionTool  # noqa: E402
from tulip.control import (
    Action,
    AuditTrail,  # noqa: E402
)

from tulip_frameworks.adk import gate_adk_tool  # noqa: E402
from tulip_frameworks.policy_presets import action_gate_policy  # noqa: E402


def _disable(ran: list[str]):
    def disable_user(email: str) -> str:
        """Disable an account."""
        ran.append(email)
        return f"disabled {email}"

    return disable_user


def test_gated_tool_keeps_name_and_signature() -> None:
    gated = gate_adk_tool(
        FunctionTool(func=_disable([])),
        action=lambda n, a: Action(name=n, asset=a["email"], environment="dev"),
        policy=action_gate_policy(),
    )
    assert gated.name == "disable_user"
    # ADK introspects the function signature for its declaration — must see `email`.
    import inspect

    assert "email" in inspect.signature(gated.func).parameters


async def test_allow_runs() -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_adk_tool(
        FunctionTool(func=_disable(ran)),
        action=lambda n, a: Action(name=n, asset=a["email"], environment="dev", kind="identity"),
        policy=action_gate_policy(),
        trail=trail,
    )
    out = await gated.func(email="u@corp")
    assert "disabled u@corp" in out
    assert ran == ["u@corp"]
    assert trail.verify() is True


async def test_production_held() -> None:
    ran: list[str] = []
    gated = gate_adk_tool(
        FunctionTool(func=_disable(ran)),
        action=lambda n, a: Action(
            name=n, asset=a["email"], environment="production", kind="identity"
        ),
        policy=action_gate_policy(),
    )
    payload = json.loads(await gated.func(email="m@corp"))
    assert payload["status"] == "held_for_approval"
    assert ran == []
