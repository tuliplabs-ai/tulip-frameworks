# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""LangGraph bridge.

LangGraph's ``ToolNode`` consumes LangChain tools unchanged, so gating a tool is just
:func:`tulip_frameworks.langchain.gate_langchain_tool` — re-exported here for
discoverability. For a raw graph **node** that performs a side effect directly (rather
than via a tool), use :func:`gate_node`.

Needs the ``langgraph`` extra: ``pip install tulip-frameworks[langgraph]``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from tulip.control import Action, AuditTrail, ControlPolicy
from tulip.security import Evidence

from tulip_frameworks.approval import ApprovalBridge
from tulip_frameworks.core import Mode, VerificationResult, gate_callable
from tulip_frameworks.langchain import gate_langchain_tool

#: An :class:`Action`, or a callable deriving one from the graph ``state``.
NodeActionSpec = Action | Callable[[Any], Action]


def gate_node(
    node_fn: Callable[[Any], Any],
    *,
    name: str,
    action: NodeActionSpec,
    policy: ControlPolicy,
    trail: AuditTrail | None = None,
    mode: Mode = "soft",
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    principal: str = "agent",
    approval: ApprovalBridge | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """Gate a LangGraph node that performs a side effect.

    ``node_fn`` is the usual ``(state) -> state`` (or partial-state) node. It runs only
    if its :class:`Action` clears policy; otherwise the node returns the held/denied
    result (soft mode) or raises (raise mode), and the decision is on ``trail``.

    ``action`` may be a constant :class:`Action` or a callable ``(state) -> Action``.
    """

    def spec(_name: str, kwargs: Mapping[str, Any]) -> Action:
        state = kwargs["state"]
        if isinstance(action, Action):
            return action
        return action(state)

    gated = gate_callable(
        node_fn,
        name=name,
        action=spec,
        policy=policy,
        trail=trail,
        mode=mode,
        finding=finding,
        verdict=verdict,
        principal=principal,
        approval=approval,
        serialize=False,  # a node returns state, not an LLM tool message
    )

    async def run(state: Any) -> Any:
        return await gated(state=state)

    run.__name__ = name
    return run


__all__ = ["NodeActionSpec", "gate_langchain_tool", "gate_node"]
