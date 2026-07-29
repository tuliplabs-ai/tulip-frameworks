# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A real Google ADK agent, driven by ADK's own ``Runner``, is governed by the wrapper.

The agent, the session service, the function-call plumbing and the event stream are
ADK's; only the model is scripted, so the test needs no Google credentials.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

pytest.importorskip("google.adk")

from _case import (  # noqa: E402
    POLICY,
    allowed_action,
    assert_held_payload,
    assert_recorded,
    held_action,
)
from google.adk.agents import Agent  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402
from tulip.control import AuditTrail  # noqa: E402

from tulip_frameworks.adk import gate_adk_tool  # noqa: E402


class ScriptedLlm(BaseLlm):
    """Replays fixed model parts — a function call, then a closing message."""

    script: list[Any] = []
    index: int = 0

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        part = self.script[min(self.index, len(self.script) - 1)]
        object.__setattr__(self, "index", self.index + 1)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _agent(order_id: str, executed: list[str], action: Any) -> tuple[Agent, AuditTrail]:
    def refund(order_id: str) -> str:
        """Issue a customer refund for an order."""
        executed.append(order_id)
        return f"refunded {order_id}"

    trail = AuditTrail()
    gated = gate_adk_tool(FunctionTool(func=refund), action=action, policy=POLICY, trail=trail)
    model = ScriptedLlm(
        model="scripted",
        script=[
            types.Part(
                function_call=types.FunctionCall(name="refund", args={"order_id": order_id})
            ),
            types.Part(text="finished"),
        ],
    )
    return Agent(name="support", model=model, instruction="Help.", tools=[gated]), trail


async def _drive(agent: Agent, prompt: str) -> list[Any]:
    runner = InMemoryRunner(agent=agent, app_name="gate-test")
    session = await runner.session_service.create_session(app_name="gate-test", user_id="u")
    return [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        )
    ]


def _tool_response(events: list[Any]) -> Any:
    for event in events:
        for part in getattr(event.content, "parts", None) or []:
            if part.function_response is not None:
                return part.function_response.response
    raise AssertionError("the agent loop never dispatched the tool")


async def test_allowed_call_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-sandbox", executed, allowed_action)

    events = await _drive(agent, "refund ord-sandbox")

    assert executed == ["ord-sandbox"], "an admitted refund did not run"
    assert "refunded ord-sandbox" in json.dumps(_tool_response(events))
    assert_recorded(trail, outcome="allow", asset="ord-sandbox")


async def test_held_call_never_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-9", executed, held_action)

    events = await _drive(agent, "refund ord-9")

    assert executed == [], f"the gate let a held refund execute: {executed}"
    response = _tool_response(events)
    assert_held_payload(json.loads(response["result"]), asset="ord-9")
    assert_recorded(trail, outcome="require_human", asset="ord-9")
