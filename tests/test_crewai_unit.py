# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""CrewAI bridge — unit coverage WITHOUT the heavy ``crewai`` dependency.

CrewAI's tree is the heaviest and churniest of the supported frameworks, so neither
CI nor the hatch gate installs it. The bridge is decoupled from CrewAI's ``BaseTool``
by a lazy import, so here we inject a minimal stand-in ``crewai.tools`` module and
exercise the bridge's REAL wrapping logic end to end: inner-callable resolution,
``gate_callable``, the ``_GatedTool`` subclass, and the synchronous ``run_sync`` drive
(CrewAI executes tools synchronously). The real-framework path is additionally covered
by ``test_crewai.py`` whenever the ``crewai`` extra happens to be installed.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest
from tulip.control import Action, AuditTrail

from tulip_frameworks.crewai import gate_crewai_tool
from tulip_frameworks.policy_presets import action_gate_policy


class _FakeBaseTool:
    """A minimal stand-in for ``crewai.tools.BaseTool`` — subclassable + instantiable."""

    name: str = ""
    description: str = ""
    args_schema: Any = None

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeCrewTool:
    """The CrewAI tool object the bridge wraps: has ``name``/``description``/``_run``."""

    def __init__(self, ran: list[str]) -> None:
        self.name = "refund"
        self.description = "Issue a customer refund."
        self.args_schema = None
        self._ran = ran

    def _run(self, order_id: str) -> str:
        self._ran.append(order_id)
        return f"refunded {order_id}"


@pytest.fixture
def fake_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``crewai.tools`` module exposing ``BaseTool``."""
    crewai_mod = types.ModuleType("crewai")
    tools_mod = types.ModuleType("crewai.tools")
    tools_mod.BaseTool = _FakeBaseTool  # type: ignore[attr-defined]
    crewai_mod.tools = tools_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", crewai_mod)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools_mod)


def test_gated_tool_preserves_metadata(fake_crewai: None) -> None:
    gated = gate_crewai_tool(
        _FakeCrewTool([]),
        action=lambda n, a: Action(name=n, asset=a["order_id"], environment="dev"),
        policy=action_gate_policy(),
    )
    assert isinstance(gated, _FakeBaseTool)  # it really subclasses crewai's BaseTool
    assert gated.name == "refund"
    assert gated.description == "Issue a customer refund."


def test_allow_runs_via_sync_bridge(fake_crewai: None) -> None:
    ran: list[str] = []
    trail = AuditTrail()
    gated = gate_crewai_tool(
        _FakeCrewTool(ran),
        action=lambda n, a: Action(name=n, asset=a["order_id"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    # _run drives the async gate synchronously (run_sync -> asyncio.run, no loop here).
    out = gated._run(order_id="ord-1")
    assert out == "refunded ord-1"
    assert ran == ["ord-1"]
    assert trail.verify() is True


def test_production_held_is_serialized(fake_crewai: None) -> None:
    ran: list[str] = []
    gated = gate_crewai_tool(
        _FakeCrewTool(ran),
        action=lambda n, a: Action(
            name=n, asset=a["order_id"], environment="production", kind="payment"
        ),
        policy=action_gate_policy(),
    )
    payload = json.loads(gated._run(order_id="ord-9"))  # serialize=True -> JSON string
    assert payload["status"] == "held_for_approval"
    assert ran == []  # the refund never ran


def test_missing_crewai_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the extra, the bridge raises a clear ``pip install`` hint."""
    # None in sys.modules makes `from crewai.tools import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "crewai", None)
    monkeypatch.setitem(sys.modules, "crewai.tools", None)
    with pytest.raises(ImportError, match=r"tulip-frameworks\[crewai\]"):
        gate_crewai_tool(
            _FakeCrewTool([]),
            action=lambda n, a: Action(name=n, asset="x", environment="dev"),
            policy=action_gate_policy(),
        )
