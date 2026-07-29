# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Agents SDK bridge — gate a ``function_tool`` behind Tulip's admission gate.

    from agents import Agent, function_tool
    from tulip_frameworks.openai_agents import gate_openai_tool
    from tulip_frameworks.policy_presets import action_gate_policy
    from tulip.control import Action, AuditTrail

    @function_tool
    def disable_user(email: str) -> str:
        "Disable an account."
        return idp.disable(email)

    trail = AuditTrail()
    gated = gate_openai_tool(disable_user,
        action=lambda n, a: Action(name=n, asset=a["email"],
            environment="production", kind="identity"),
        policy=action_gate_policy(), trail=trail)
    # agent = Agent(name="IT Help", instructions="Help with accounts.", tools=[gated])

Needs the ``openai-agents`` extra: ``pip install tulip-frameworks[openai-agents]``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tulip.control import AuditTrail
from tulip.security import Evidence

from tulip_frameworks.actions import ActionSpec
from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.core import GatePolicy, Mode, VerificationResult, gate_callable

if TYPE_CHECKING:
    from agents import FunctionTool


def _require_agents() -> Any:
    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "OpenAI Agents support needs: pip install tulip-frameworks[openai-agents]"
        ) from exc
    return FunctionTool


def gate_openai_tool(
    tool: Any,
    *,
    action: ActionSpec,
    policy: GatePolicy,
    trail: AuditTrail | None = None,
    mode: Mode = "soft",
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    principal: str = "agent",
    approval: ApprovalBridge | None = None,
) -> FunctionTool:
    """Return an OpenAI-Agents ``FunctionTool`` whose action is gated by :func:`admit`.

    Keeps the original ``name``, ``description`` and ``params_json_schema``; only the
    invocation is wrapped, so it drops into ``Agent(tools=[...])`` unchanged. The
    invoker always returns a string (the gate's held result is JSON-serialized), as the
    Agents runtime requires.
    """
    function_tool = _require_agents()

    async def on_invoke(ctx: Any, args_json: str) -> str:
        kwargs = json.loads(args_json or "{}")

        async def call(**kw: Any) -> Any:
            return await tool.on_invoke_tool(ctx, json.dumps(kw))

        gated = gate_callable(
            call,
            name=tool.name,
            action=action,
            policy=policy,
            trail=trail,
            mode=mode,
            finding=finding,
            verdict=verdict,
            principal=principal,
            approval=approval,
            serialize=True,
        )
        out: str = await gated(**kwargs)  # serialize=True -> always a string
        return out

    return function_tool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.params_json_schema,
        on_invoke_tool=on_invoke,
    )


__all__ = ["gate_openai_tool"]
