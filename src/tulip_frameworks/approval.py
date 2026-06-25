# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Out-of-band approval — a small Protocol the gateway's broker already satisfies.

When a tool is gated in ``mode="soft"`` and an action is held for a human, you can
hand :func:`tulip_frameworks.core.gate_callable` an :class:`ApprovalBridge`. The held
result then carries an ``approval_id`` the agent can poll, while a human approves or
denies on a side channel the agent has no access to.

This is a structural Protocol on purpose: it has **no import-time dependency** on
``tulip-gateway`` (which is not yet on PyPI). ``tulip_gateway.policy.approval``'s
``ApprovalBroker`` matches it in shape; :func:`gateway_bridge` adapts one when present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ApprovalBridge(Protocol):
    """Submit a held action for out-of-band approval and check its state."""

    def submit(self, principal: str, tool: str, args: Mapping[str, Any]) -> str:
        """Record a pending approval; return an id the agent can poll."""
        ...

    def state(self, approval_id: str) -> str | None:
        """Current state for an id — e.g. ``"pending"`` / ``"approved"`` / ``"denied"``."""
        ...


def gateway_bridge(broker: Any) -> ApprovalBridge:
    """Adapt a ``tulip_gateway`` ``ApprovalBroker`` to the :class:`ApprovalBridge` shape.

    The broker's ``submit`` returns an ``ApprovalRecord``; this thin adapter returns
    its ``approval_id`` instead, and maps ``get`` to :meth:`ApprovalBridge.state`.
    """

    class _Bridge:
        def submit(self, principal: str, tool: str, args: Mapping[str, Any]) -> str:
            record = broker.submit(principal, tool, dict(args))
            return str(getattr(record, "approval_id", record))

        def state(self, approval_id: str) -> str | None:
            record = broker.get(approval_id)
            if record is None:
                return None
            return str(getattr(record, "state", record))

    return _Bridge()


__all__ = ["ApprovalBridge", "gateway_bridge"]
