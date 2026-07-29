# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A real OpenAI-Agents ``Runner`` loop is governed by the wrapper.

The agent, the ``function_tool``, the tool-dispatch machinery and the item stream are
all the SDK's own; only the model is scripted, so no key and no paid call is needed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("agents")

from _case import (  # noqa: E402
    POLICY,
    allowed_action,
    assert_held_payload,
    assert_recorded,
    held_action,
)
from agents import Agent, RunConfig, Runner, function_tool  # noqa: E402
from agents.items import ModelResponse, ToolCallOutputItem  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from tulip.control import AuditTrail  # noqa: E402

from tulip_frameworks.openai_agents import gate_openai_tool  # noqa: E402


class ScriptedModel(Model):
    """Replays a fixed list of model outputs — deterministic, offline."""

    def __init__(self, script: list[list[Any]]) -> None:
        self.script = script
        self.index = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        output = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


def _agent(order_id: str, executed: list[str], action: Any) -> tuple[Agent, AuditTrail]:
    @function_tool
    def refund(order_id: str) -> str:
        """Issue a customer refund for an order."""
        executed.append(order_id)
        return f"refunded {order_id}"

    trail = AuditTrail()
    gated = gate_openai_tool(refund, action=action, policy=POLICY, trail=trail)
    model = ScriptedModel(
        [
            [
                ResponseFunctionToolCall(
                    id="fc-1",
                    call_id="call-1",
                    name="refund",
                    arguments=json.dumps({"order_id": order_id}),
                    type="function_call",
                )
            ],
            [
                ResponseOutputMessage(
                    id="msg-1",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(text="finished", type="output_text", annotations=[])
                    ],
                )
            ],
        ]
    )
    return Agent(name="support", instructions="Help.", tools=[gated], model=model), trail


def _tool_output(result: Any) -> str:
    outputs = [i for i in result.new_items if isinstance(i, ToolCallOutputItem)]
    assert outputs, "the agent loop never dispatched the tool"
    return str(outputs[0].output)


async def test_allowed_call_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-sandbox", executed, allowed_action)

    result = await Runner.run(
        agent, "refund ord-sandbox", run_config=RunConfig(tracing_disabled=True)
    )

    assert executed == ["ord-sandbox"], "an admitted refund did not run"
    assert "refunded ord-sandbox" in _tool_output(result)
    assert_recorded(trail, outcome="allow", asset="ord-sandbox")


async def test_held_call_never_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-9", executed, held_action)

    result = await Runner.run(agent, "refund ord-9", run_config=RunConfig(tracing_disabled=True))

    assert executed == [], f"the gate let a held refund execute: {executed}"
    assert_held_payload(json.loads(_tool_output(result)), asset="ord-9")
    assert_recorded(trail, outcome="require_human", asset="ord-9")
