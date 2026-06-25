# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Offline test helpers — assert a gated tool admits/holds without running an LLM.

Mirrors ``tulip.security.testing``: deterministic, no network, no model. Use :class:`Spy`
as the side effect, gate it, invoke it directly, and assert whether it ran.
"""

from __future__ import annotations

from typing import Any

from tulip_frameworks.core import DENIED, HELD, is_held


class Spy:
    """A stand-in side effect that records whether (and how) it was invoked."""

    def __init__(self, result: Any = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self.result

    @property
    def ran(self) -> bool:
        """Whether the side effect actually executed."""
        return bool(self.calls)


def assert_allowed(result: Any, spy: Spy) -> None:
    """Assert the action ran and returned the spy's result."""
    assert spy.ran, "expected the gated action to run, but it did not"
    assert result == spy.result, f"expected {spy.result!r}, got {result!r}"


def assert_held(result: Any, spy: Spy, *, outcome: str = HELD) -> None:
    """Assert the action did NOT run and a held/denied result was returned.

    ``outcome`` is ``HELD`` (require_human) or ``DENIED``. Accepts the structured dict
    form; if ``result`` is a JSON string (serialize=True), decode it first.
    """
    assert not spy.ran, "expected the gated action to be held, but it ran"
    if isinstance(result, str):
        import json

        result = json.loads(result)
    assert is_held(result), f"expected a held/denied result, got {result!r}"
    assert result.get("status") == outcome, f"expected status {outcome!r}, got {result!r}"


__all__ = ["DENIED", "HELD", "Spy", "assert_allowed", "assert_held"]
