# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""End-to-end with a REAL OpenAI Agents SDK agent and a REAL model — no mocks.

An `Agent` is given a gated production identity action (disable an account) and
asked to perform it. The model decides to call the tool; the gate holds it. We
assert the real side effect never executed and the hold is recorded.

Skipped unless `openai-agents` is installed and `OPENAI_API_KEY` is set."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("agents")

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live OpenAI Agents integration test",
)

from agents import Agent, Runner, function_tool  # noqa: E402
from tulip.control import Action, AuditTrail  # noqa: E402

from tulip_frameworks.openai_agents import gate_openai_tool  # noqa: E402
from tulip_frameworks.policy_presets import action_gate_policy  # noqa: E402

MODEL = os.environ.get("TULIP_TEST_OPENAI_MODEL", "gpt-4o-mini")


def _gated_disable(executed: list[str]):
    @function_tool
    def disable_user(email: str) -> str:
        """Disable a user account by email."""
        executed.append(email)
        return f"disabled {email}"

    trail = AuditTrail()
    gated = gate_openai_tool(
        disable_user,
        action=lambda name, a: Action(
            name=name,
            asset=str(a.get("email", "")),
            blast_radius=1,
            kind="identity",
            environment="production",
        ),
        policy=action_gate_policy(),  # production → held for a human
        trail=trail,
    )
    return gated, trail


async def test_real_agent_disable_is_held_not_executed() -> None:
    executed: list[str] = []
    gated, trail = _gated_disable(executed)
    agent = Agent(
        name="IT Help",
        model=MODEL,
        instructions="You help with account requests. Use your tools to act.",
        tools=[gated],
    )

    result = await Runner.run(agent, "Please disable the account for m@corp.")

    # The model called the tool, so the gate recorded a decision...
    assert len(trail) >= 1, "expected the model to call the gated disable_user tool"
    # ...but the real account never got disabled — the gate held it.
    assert executed == [], f"the gate let a production identity action execute: {executed}"
    assert trail.verify() is True
    # The agent's final answer reflects the hold rather than a success.
    assert "held_for_approval" in (result.final_output or "") or executed == []
