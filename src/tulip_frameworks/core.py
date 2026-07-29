# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The framework-agnostic gate — the only place that calls :func:`tulip.security.admit`.

Every per-framework bridge (``langchain``, ``langgraph``, ``openai_agents``, …) is a
thin re-wrapper around :func:`gate_callable`. Give it the callable that performs a
side effect plus the :class:`~tulip.security.Action` that describes it, and it returns
an async callable that runs the side effect **only** if the action clears policy —
holding for a human or denying otherwise, and recording the decision on a
tamper-evident :class:`~tulip.security.AuditTrail` either way.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from tulip.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    admit,
)
from tulip.security import (
    Evidence,
    as_json,
)
from tulip.security import (
    VerificationResult as VerificationResult,
    # re-exported for callers that pass a verification result,
)

from tulip_frameworks.actions import ActionSpec, resolve_action
from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.gateway import RemotePolicy

#: Where the decision is taken: in process (:class:`ControlPolicy`) or on a gateway
#: over HTTP (:class:`~tulip_frameworks.gateway.RemotePolicy`). Every bridge accepts
#: either, so moving the gate across a process boundary is a one-argument change.
GatePolicy = ControlPolicy | RemotePolicy

#: How a held/denied action surfaces. ``"soft"`` returns an LLM-readable result the
#: agent loop can react to; ``"raise"`` re-raises :class:`~tulip.security.AdmissionError`.
Mode = Literal["soft", "raise"]

#: ``status`` value on a soft-mode result when an action is held for a human.
HELD = "held_for_approval"
#: ``status`` value on a soft-mode result when an action is denied outright.
DENIED = "denied"


def _ensure_async(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Adapt a sync-or-async callable into an awaitable one (called with kwargs)."""
    if inspect.iscoroutinefunction(fn):
        return fn

    async def _wrapped(**kwargs: Any) -> Any:
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return _wrapped


def held_result(
    decision: Any,
    *,
    kwargs: Mapping[str, Any],
    principal: str,
    approval: ApprovalBridge | None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Build the structured result returned in soft mode when an action doesn't admit.

    On ``require_human`` and an ``approval`` bridge is given, an approval is submitted
    out of band and its id is embedded so the agent can poll for the human decision.
    """
    action: Action = decision.action
    out: dict[str, Any] = {
        "status": DENIED if decision.outcome == "deny" else HELD,
        "outcome": decision.outcome,
        "action": action.name,
        "asset": action.asset,
        "reason": decision.reason,
    }
    if decision.outcome == "deny":
        return out
    # A remote gate has already opened the approval and told us its id; an in-process
    # gate needs a bridge to open one. Either way the agent gets a pollable id.
    if approval_id is not None:
        out["approval_id"] = approval_id
    elif approval is not None:
        out["approval_id"] = approval.submit(principal, action.name, dict(kwargs))
    if "approval_id" in out:
        out["next"] = "call approval_status(approval_id) once a human decides"
    return out


def gate_callable(
    fn: Callable[..., Any],
    *,
    name: str,
    action: ActionSpec,
    policy: GatePolicy,
    trail: AuditTrail | None = None,
    mode: Mode = "soft",
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    principal: str = "agent",
    approval: ApprovalBridge | None = None,
    serialize: bool = False,
) -> Callable[..., Awaitable[Any]]:
    """Wrap ``fn`` so it runs only after its :class:`Action` clears the admission gate.

    Args:
        fn: The callable that performs the side effect (sync or async). Invoked with
            the gated callable's keyword arguments.
        name: A stable tool name used in the audit record and held result.
        action: An :class:`~tulip.security.Action`, or a callable
            ``(name, kwargs) -> Action`` that derives one per call (the recommended
            form — risk usually varies with the arguments).
        policy: The governing :class:`~tulip.security.ControlPolicy`. For arbitrary
            tools with no verification, prefer
            :func:`tulip_frameworks.policy_presets.action_gate_policy`.
        trail: An :class:`~tulip.security.AuditTrail` to record every decision on.
        mode: ``"soft"`` (default) returns a held/denied result the LLM can read;
            ``"raise"`` re-raises :class:`~tulip.security.AdmissionError`.
        finding / verdict: Optional grounded evidence + verification, forwarded to the
            gate so a fully grounded caller gets the complete trust chain.
        principal: Identity recorded against an out-of-band approval.
        approval: Optional :class:`~tulip_frameworks.approval.ApprovalBridge` to submit
            a ``require_human`` hold to (e.g. the gateway's broker).
        serialize: If ``True``, the soft-mode held/denied result is returned as a JSON
            string (what a string-typed tool result needs). If ``False`` (default), it
            is returned as a ``dict``.

    Returns:
        An async callable. On allow it returns whatever ``fn`` returns; in soft mode
        on hold/deny it returns the held result; in raise mode it raises.
    """
    perform_async = _ensure_async(fn)

    async def gated(**kwargs: Any) -> Any:
        act = resolve_action(action, name, kwargs)

        async def perform() -> Any:
            return await perform_async(**kwargs)

        try:
            if isinstance(policy, RemotePolicy):
                # The decision crosses the wire; ``perform`` never leaves this process.
                return await policy.admit(act, perform, trail=trail)
            return await admit(
                act, perform, policy=policy, finding=finding, verdict=verdict, trail=trail
            )
        except AdmissionError as exc:
            if mode == "raise":
                raise
            result = held_result(
                exc.decision,
                kwargs=kwargs,
                principal=principal,
                approval=approval,
                approval_id=getattr(exc, "approval_id", None),
            )
            return as_json(result) if serialize else result

    gated.__name__ = name
    # Preserve the wrapped tool's signature + docstring so frameworks that build
    # their arg schema by introspecting the callable (ADK, LlamaIndex, …) see the
    # real parameters rather than the gated wrapper's ``**kwargs``.
    try:
        gated.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        pass
    if getattr(fn, "__doc__", None):
        gated.__doc__ = fn.__doc__
    return gated


def is_held(result: Any) -> bool:
    """Whether a soft-mode result represents a held/denied action (dict form)."""
    return isinstance(result, Mapping) and result.get("status") in {HELD, DENIED}


__all__ = [
    "DENIED",
    "HELD",
    "GatePolicy",
    "Mode",
    "VerificationResult",
    "gate_callable",
    "held_result",
    "is_held",
]
