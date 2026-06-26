# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Deriving a Tulip :class:`~tulip.security.Action` from a tool call.

The gate reasons over an :class:`Action` — its ``asset``, ``blast_radius``,
``environment``, ``kind`` and ``tags``. A bridge can supply that three ways:

1. a constant :class:`Action` (risk is the same regardless of arguments),
2. a callable ``(name, kwargs) -> Action`` (the recommended form — risk usually
   varies with the call's arguments), or
3. nothing, in which case :func:`default_action` builds a conservative one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tulip.control import Action

#: An :class:`Action`, or a callable that derives one from ``(name, kwargs)``.
ActionSpec = Action | Callable[[str, Mapping[str, Any]], Action]

#: Argument names commonly naming the asset an action touches, in priority order.
_ASSET_KEYS = (
    "asset",
    "host",
    "hostname",
    "target",
    "resource",
    "resource_id",
    "account",
    "user",
    "username",
    "email",
    "order_id",
    "id",
    "ip",
    "url",
)


def asset_from_args(kwargs: Mapping[str, Any]) -> str:
    """Best-effort asset label from a tool call's arguments."""
    for key in _ASSET_KEYS:
        value = kwargs.get(key)
        if value:
            return str(value)
    return ""


def default_action(
    name: str,
    kwargs: Mapping[str, Any],
    *,
    environment: str = "unknown",
    kind: str = "",
) -> Action:
    """A conservative :class:`Action` for ``name`` when none was supplied.

    Fail-safe by construction: ``environment="unknown"`` plus the stock
    :class:`~tulip.security.ControlPolicy` (which requires a verification score)
    lands an un-verified call on ``require_human`` rather than auto-allowing it.
    """
    return Action(
        name=name,
        asset=asset_from_args(kwargs),
        blast_radius=1,
        environment=environment,
        kind=kind,
    )


def resolve_action(spec: ActionSpec | None, name: str, kwargs: Mapping[str, Any]) -> Action:
    """Resolve an :class:`ActionSpec` (or ``None``) into a concrete :class:`Action`."""
    if spec is None:
        return default_action(name, kwargs)
    if isinstance(spec, Action):
        return spec
    return spec(name, kwargs)


__all__ = ["ActionSpec", "asset_from_args", "default_action", "resolve_action"]
