# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A real LlamaIndex ``FunctionAgent`` workflow is governed by the wrapper.

The agent workflow, tool-selection handling and tool dispatch are LlamaIndex's own;
only the function-calling LLM is scripted, so the test is offline and free.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("llama_index.core")

from _case import (  # noqa: E402
    POLICY,
    allowed_action,
    assert_held_payload,
    assert_recorded,
    held_action,
)
from llama_index.core.agent.workflow import FunctionAgent  # noqa: E402
from llama_index.core.base.llms.types import (  # noqa: E402
    ChatMessage,
    ChatResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback  # noqa: E402
from llama_index.core.llms.function_calling import FunctionCallingLLM  # noqa: E402
from llama_index.core.tools import FunctionTool, ToolSelection  # noqa: E402
from tulip.control import AuditTrail  # noqa: E402

from tulip_frameworks.llamaindex import gate_llamaindex_tool  # noqa: E402


class ScriptedLLM(FunctionCallingLLM):
    """A deterministic function-calling LLM: replays a fixed list of chat responses."""

    script: list[Any] = []
    index: int = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, model_name="scripted")

    def _next(self) -> ChatResponse:
        step = self.script[min(self.index, len(self.script) - 1)]
        object.__setattr__(self, "index", self.index + 1)
        return step

    def _prepare_chat_with_tools(self, tools: Any, **kwargs: Any) -> dict[str, Any]:
        return {"messages": list(kwargs.get("chat_history") or [])}

    def get_tool_calls_from_response(self, response: Any, **kwargs: Any) -> list[ToolSelection]:
        return list(response.message.additional_kwargs.get("tool_selections", []))

    @llm_chat_callback()
    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self._next()

    @llm_chat_callback()
    async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self._next()

    @llm_chat_callback()
    async def astream_chat(self, messages: Any, **kwargs: Any) -> Any:
        response = self._next()

        async def stream() -> Any:
            yield response

        return stream()

    @llm_chat_callback()
    def stream_chat(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kw: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kw: Any
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kw: Any
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kw: Any
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


def _agent(order_id: str, executed: list[str], action: Any) -> tuple[FunctionAgent, AuditTrail]:
    def refund(order_id: str) -> str:
        """Issue a customer refund for an order."""
        executed.append(order_id)
        return f"refunded {order_id}"

    trail = AuditTrail()
    gated = gate_llamaindex_tool(
        FunctionTool.from_defaults(fn=refund, name="refund"),
        action=action,
        policy=POLICY,
        trail=trail,
    )
    llm = ScriptedLLM(
        script=[
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={
                        "tool_selections": [
                            ToolSelection(
                                tool_id="call-1",
                                tool_name="refund",
                                tool_kwargs={"order_id": order_id},
                            )
                        ]
                    },
                )
            ),
            ChatResponse(message=ChatMessage(role="assistant", content="finished")),
        ]
    )
    return FunctionAgent(tools=[gated], llm=llm, system_prompt="Help."), trail


async def test_allowed_call_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-sandbox", executed, allowed_action)

    await agent.run("refund ord-sandbox")

    assert executed == ["ord-sandbox"], "an admitted refund did not run"
    assert_recorded(trail, outcome="allow", asset="ord-sandbox")


async def test_held_call_never_reaches_the_side_effect() -> None:
    executed: list[str] = []
    agent, trail = _agent("ord-9", executed, held_action)

    result = await agent.run("refund ord-9")

    assert executed == [], f"the gate let a held refund execute: {executed}"
    # The held result is what the agent's tool output carried back to the model.
    outputs = [json.loads(str(block.tool_output.content)) for block in result.tool_calls]
    assert outputs, "the agent loop never dispatched the tool"
    assert_held_payload(outputs[0], asset="ord-9")
    assert_recorded(trail, outcome="require_human", asset="ord-9")
