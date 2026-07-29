# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A real CrewAI crew, kicked off for real, is governed by the wrapper.

Skipped unless the ``crewai`` extra is installed — the default hatch env and CI omit
CrewAI's (very heavy) tree, matching the existing ``tests/test_crewai.py``. Run it
with::

    pip install "tulip-frameworks[crewai]" && pytest tests/agent_loop -q

The crew, the agent executor and the ReAct tool-dispatch loop are CrewAI's own; only
the LLM is scripted, so no key and no paid call is needed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("crewai")

from _case import (  # noqa: E402
    POLICY,
    allowed_action,
    assert_held_payload,
    assert_recorded,
    held_action,
)
from crewai import Agent, Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402
from crewai.tools import BaseTool  # noqa: E402
from tulip.control import AuditTrail  # noqa: E402

from tulip_frameworks.crewai import gate_crewai_tool  # noqa: E402


class ScriptedLLM(BaseLLM):
    """Replays a fixed ReAct script — a tool call, then a final answer."""

    def __init__(self, script: list[str]) -> None:
        super().__init__(model="scripted")
        self.script = script
        self.index = 0

    def call(self, messages: Any, **kwargs: Any) -> str:
        answer = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return answer

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False


def _crew(order_id: str, executed: list[str], action: Any) -> tuple[Crew, AuditTrail, list[str]]:
    class Refund(BaseTool):
        name: str = "refund"
        description: str = "Issue a customer refund. Argument: order_id (string)."

        def _run(self, order_id: str) -> str:
            executed.append(order_id)
            return f"refunded {order_id}"

    trail = AuditTrail()
    gated = gate_crewai_tool(Refund(), action=action, policy=POLICY, trail=trail)
    observations: list[str] = []

    class Recording(type(gated)):  # type: ignore[misc]
        """Captures what the tool handed back to the agent loop."""

        def _run(self, **kwargs: Any) -> Any:
            out = gated._run(**kwargs)
            observations.append(str(out))
            return out

    llm = ScriptedLLM(
        [
            'Thought: I will refund.\nAction: refund\nAction Input: {"order_id": "'
            + order_id
            + '"}',
            "Thought: done\nFinal Answer: handled",
        ]
    )
    agent = Agent(
        role="support",
        goal="handle refunds",
        backstory="A support agent.",
        tools=[Recording()],
        llm=llm,
    )
    task = Task(description=f"Refund {order_id}", expected_output="confirmation", agent=agent)
    return Crew(agents=[agent], tasks=[task]), trail, observations


def test_allowed_call_reaches_the_side_effect() -> None:
    executed: list[str] = []
    crew, trail, observations = _crew("ord-sandbox", executed, allowed_action)

    crew.kickoff()

    assert executed == ["ord-sandbox"], "an admitted refund did not run"
    assert observations and "refunded ord-sandbox" in observations[0]
    assert_recorded(trail, outcome="allow", asset="ord-sandbox")


def test_held_call_never_reaches_the_side_effect() -> None:
    executed: list[str] = []
    crew, trail, observations = _crew("ord-9", executed, held_action)

    crew.kickoff()

    assert executed == [], f"the gate let a held refund execute: {executed}"
    assert observations, "the agent loop never dispatched the tool"
    assert_held_payload(json.loads(observations[0]), asset="ord-9")
    assert_recorded(trail, outcome="require_human", asset="ord-9")
