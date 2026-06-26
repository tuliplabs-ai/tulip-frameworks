# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex bridge — gate a LlamaIndex FunctionTool's action behind Tulip's gate.

    from llama_index.core.tools import FunctionTool
    from tulip_frameworks.llamaindex import gate_llamaindex_tool
    from tulip_frameworks.policy_presets import action_gate_policy
    from tulip.control import Action, AuditTrail

    trail = AuditTrail()
    safe_tool = gate_llamaindex_tool(
        refund_tool,
        action=lambda n, a: Action(name=n, asset=a["order_id"],
            kind="payment", environment="production"),
        policy=action_gate_policy(), trail=trail)
    # agent = FunctionAgent(tools=[safe_tool], llm=llm)

Needs the ``llama-index`` extra: ``pip install tulip-frameworks[llama-index]``.
"""

from __future__ import annotations

from typing import Any

from tulip.control import AuditTrail, ControlPolicy
from tulip.security import Evidence

from tulip_frameworks.actions import ActionSpec
from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.core import Mode, VerificationResult, gate_callable


def _require_llamaindex() -> Any:
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "LlamaIndex support needs: pip install tulip-frameworks[llama-index]"
        ) from exc
    return FunctionTool


def gate_llamaindex_tool(
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
) -> Any:
    """Return a LlamaIndex ``FunctionTool`` whose action is gated by :func:`admit`.

    Keeps the original ``name``, ``description`` and ``fn_schema``; the gated callable
    is installed as the tool's async function.
    """
    function_tool = _require_llamaindex()
    meta = tool.metadata
    inner = getattr(tool, "async_fn", None) or tool.fn
    gated = gate_callable(
        inner,
        name=meta.name or "tool",
        action=action,
        policy=policy,
        trail=trail,
        mode=mode,
        finding=finding,
        verdict=verdict,
        principal=principal,
        approval=approval,
        serialize=True,  # a held result must be a string for the LLM
    )
    return function_tool.from_defaults(
        async_fn=gated,
        name=meta.name,
        description=meta.description,
        fn_schema=getattr(meta, "fn_schema", None),
    )


__all__ = ["gate_llamaindex_tool"]
