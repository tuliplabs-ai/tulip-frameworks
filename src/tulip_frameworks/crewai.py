# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""CrewAI bridge — gate a CrewAI tool's action behind Tulip's admission gate.

    from crewai.tools import BaseTool
    from tulip_frameworks.crewai import gate_crewai_tool
    from tulip_frameworks.policy_presets import action_gate_policy
    from tulip.control import Action, AuditTrail

    trail = AuditTrail()
    safe_tool = gate_crewai_tool(
        my_refund_tool,
        action=lambda n, a: Action(name=n, asset=a["order_id"],
            kind="payment", environment="production"),
        policy=action_gate_policy(), trail=trail)
    # crew = Crew(agents=[agent], tasks=[task])  # agent uses safe_tool

CrewAI runs tools synchronously, so the gated coroutine is driven via a sync bridge.
Needs the ``crewai`` extra: ``pip install tulip-frameworks[crewai]``.
"""

from __future__ import annotations

from typing import Any

from tulip.control import AuditTrail, ControlPolicy
from tulip.security import Evidence

from tulip_frameworks._sync import run_sync
from tulip_frameworks.actions import ActionSpec
from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.core import Mode, VerificationResult, gate_callable


def _require_crewai() -> Any:
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("CrewAI support needs: pip install tulip-frameworks[crewai]") from exc
    return BaseTool


def gate_crewai_tool(
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
    """Return a CrewAI ``BaseTool`` whose action is gated by :func:`admit`.

    Keeps the original ``name``, ``description`` and ``args_schema``. The wrapped
    ``_run`` drives the async gate synchronously (CrewAI's execution model).
    """
    base_tool = _require_crewai()
    inner = getattr(tool, "_run", None) or tool.run
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
        serialize=True,  # a held result must be a string for the LLM
    )

    class _GatedTool(base_tool):  # type: ignore[misc, valid-type]
        name: str = tool.name
        description: str = tool.description
        args_schema: Any = getattr(tool, "args_schema", None)

        def _run(self, **kwargs: Any) -> Any:
            return run_sync(gated(**kwargs))

    return _GatedTool()


__all__ = ["gate_crewai_tool"]
