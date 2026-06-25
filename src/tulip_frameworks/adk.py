# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Google ADK bridge — gate an Agent Development Kit tool behind Tulip's gate.

    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
    from tulip_frameworks.adk import gate_adk_tool
    from tulip_frameworks.policy_presets import action_gate_policy
    from tulip.security import Action, AuditTrail

    def disable_user(email: str) -> str:
        "Disable an account."
        return idp.disable(email)

    trail = AuditTrail()
    gated = gate_adk_tool(
        FunctionTool(func=disable_user),
        action=lambda n, a: Action(name=n, asset=a["email"],
            environment="production", kind="identity"),
        policy=action_gate_policy(), trail=trail)
    # agent = Agent(model="gemini-2.0-flash", tools=[gated])

Accepts an ADK ``FunctionTool`` or a bare Python function. Needs the ``adk``
extra: ``pip install tulip-frameworks[adk]``.
"""

from __future__ import annotations

from typing import Any

from tulip.security import AuditTrail, Finding, SecurityPolicy

from tulip_frameworks.actions import ActionSpec
from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.core import Mode, Verdict, gate_callable


def _require_adk() -> Any:
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Google ADK support needs: pip install tulip-frameworks[adk]"
        ) from exc
    return FunctionTool


def gate_adk_tool(
    tool: Any,
    *,
    action: ActionSpec,
    policy: SecurityPolicy,
    trail: AuditTrail | None = None,
    mode: Mode = "soft",
    finding: Finding | None = None,
    verdict: Verdict | None = None,
    principal: str = "agent",
    approval: ApprovalBridge | None = None,
) -> Any:
    """Return an ADK ``FunctionTool`` whose action is gated by :func:`admit`.

    ``tool`` may be an ADK ``FunctionTool`` or a plain function. The gated callable
    keeps the original name, docstring, and signature (so ADK builds the right
    function declaration), and runs the action only after it clears policy.
    """
    function_tool = _require_adk()
    inner = getattr(tool, "func", tool)  # FunctionTool.func, or a bare function
    name = str(getattr(tool, "name", None) or getattr(inner, "__name__", "tool"))
    gated = gate_callable(
        inner,
        name=name,
        action=action,
        policy=policy,
        trail=trail,
        mode=mode,
        finding=finding,
        verdict=verdict,
        principal=principal,
        approval=approval,
        serialize=True,  # a held result must be a string for the model
    )
    return function_tool(func=gated)


__all__ = ["gate_adk_tool"]
