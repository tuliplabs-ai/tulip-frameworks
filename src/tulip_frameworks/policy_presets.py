# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Policy presets tuned for gating arbitrary framework tools.

The stock :class:`~tulip.security.SecurityPolicy` is tuned for the grounded-finding
SOC flow: ``require_verification_score=0.8`` means a call with **no** ``Verdict`` is
forced to ``require_human``. That is correct when every action answers a verified
finding — but when you are simply gating ordinary agent tools (a refund, a delete),
you usually have no ``Verdict``, and the default would hold *everything*.

:func:`action_gate_policy` flips that: verification is optional, and risk is governed
by the action's labels (environment / kind / tags) and blast radius instead. A caller
that *does* have grounded evidence can still pass ``finding`` / ``verdict`` through the
gate and get the full trust chain.
"""

from __future__ import annotations

from collections.abc import Iterable

from tulip.security import SecurityPolicy


def action_gate_policy(
    *,
    require_human_for: Iterable[str] = ("production",),
    deny_for: Iterable[str] = ("irreversible",),
    max_blast_radius: int = 1,
) -> SecurityPolicy:
    """A policy that gates on labels + blast radius, not on a verification score.

    Args:
        require_human_for: Labels (environment / kind / tag) that always need a human.
        deny_for: Labels that are hard-denied outright.
        max_blast_radius: Most assets an action may affect to auto-allow.

    Returns:
        A :class:`~tulip.security.SecurityPolicy` with ``require_verification_score=0``
        so an un-verified tool call is allowed unless a label or the blast radius
        trips a hold/deny.
    """
    return SecurityPolicy(
        require_verification_score=0.0,
        max_blast_radius=max_blast_radius,
        require_human_for=frozenset(require_human_for),
        deny_for=frozenset(deny_for),
    )


__all__ = ["action_gate_policy"]
