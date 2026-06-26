# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""LangChain bridge — gate a LangChain tool's action behind Tulip's admission gate.

    from langchain_core.tools import tool
    from tulip_frameworks.langchain import gate_langchain_tool
    from tulip_frameworks.policy_presets import action_gate_policy
    from tulip.control import Action, AuditTrail

    @tool
    async def refund(order_id: str, amount_usd: float) -> str:
        "Issue a customer refund."
        return payments.refund(order_id, amount_usd)

    trail = AuditTrail()
    safe_refund = gate_langchain_tool(
        refund,
        action=lambda name, a: Action(name=name, asset=a["order_id"],
            blast_radius=1, kind="payment", environment="production"),
        policy=action_gate_policy(), trail=trail)
    # agent = create_react_agent(model, tools=[safe_refund])

Needs the ``langchain`` extra: ``pip install tulip-frameworks[langchain]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.control import AuditTrail, ControlPolicy
from tulip.security import Evidence

from tulip_frameworks.actions import ActionSpec
from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.core import Mode, VerificationResult, gate_callable

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


def _require_langchain() -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "LangChain support needs: pip install tulip-frameworks[langchain]"
        ) from exc
    return StructuredTool


def gate_langchain_tool(
    tool: Any,
    *,
    action: ActionSpec,
    policy: ControlPolicy,
    trail: AuditTrail | None = None,
    mode: Mode = "soft",
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    principal: str = "agent",
    approval: ApprovalBridge | None = None,
) -> StructuredTool:
    """Return a LangChain ``StructuredTool`` whose action is gated by :func:`admit`.

    The returned tool keeps the original ``name``, ``description`` and ``args_schema``,
    so it is a drop-in replacement in any agent/graph that consumed the original. It is
    async (``coroutine``); use it where the runtime awaits tools (e.g. LangGraph).
    """
    structured_tool = _require_langchain()
    # The underlying callable: prefer the async impl, else the sync one.
    inner = getattr(tool, "coroutine", None) or tool.func
    gated = gate_callable(
        inner,
        name=tool.name,
        action=action,
        policy=policy,
        trail=trail,
        mode=mode,
        finding=finding,
        verdict=verdict,
        principal=principal,
        approval=approval,
        serialize=True,  # a held result must be a string for the LLM's tool message
    )
    return structured_tool.from_function(
        coroutine=gated,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


__all__ = ["gate_langchain_tool"]
