# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A real LangChain/LangGraph agent, driven end to end, is governed by the wrapper.

The existing bridge tests invoke the gated tool by hand. This one does not: a real
``create_react_agent`` graph decides to call the tool, the ToolNode dispatches it, and
the gate has to hold it *inside the agent's own loop*. The model is scripted — no
network, no key, no paid call — but every other component is the real one.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from _case import (  # noqa: E402
    POLICY,
    allowed_action,
    assert_held_payload,
    assert_recorded,
    held_action,
)
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from tulip.control import AuditTrail  # noqa: E402

from tulip_frameworks.langchain import gate_langchain_tool  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """A deterministic stand-in for the LLM: replays a fixed list of AI messages."""

    script: list[Any] = []
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        message = self.script[min(self.index, len(self.script) - 1)]
        object.__setattr__(self, "index", self.index + 1)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _agent(order_id: str, executed: list[str], action: Any) -> tuple[Any, AuditTrail]:
    @tool
    def refund(order_id: str) -> str:
        """Issue a customer refund for an order."""
        executed.append(order_id)
        return f"refunded {order_id}"

    trail = AuditTrail()
    gated = gate_langchain_tool(refund, action=action, policy=POLICY, trail=trail)
    model = ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "refund", "args": {"order_id": order_id}, "id": "call-1"}],
            ),
            AIMessage(content="finished"),
        ]
    )
    return create_react_agent(model, tools=[gated]), trail


def _tool_message(out: dict[str, Any]) -> ToolMessage:
    messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert messages, "the agent loop never dispatched the tool"
    return messages[0]


async def test_allowed_call_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-sandbox", executed, allowed_action)

    out = await agent.ainvoke({"messages": [("user", "refund ord-sandbox")]})

    assert executed == ["ord-sandbox"], "an admitted refund did not run"
    assert _tool_message(out).content == "refunded ord-sandbox"
    assert_recorded(trail, outcome="allow", asset="ord-sandbox")


async def test_held_call_never_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-9", executed, held_action)

    out = await agent.ainvoke({"messages": [("user", "refund ord-9")]})

    assert executed == [], f"the gate let a held refund execute: {executed}"
    assert_held_payload(json.loads(_tool_message(out).content), asset="ord-9")
    assert_recorded(trail, outcome="require_human", asset="ord-9")
