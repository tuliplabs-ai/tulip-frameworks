# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit: the Claude Code PreToolUse gate — event in, permission decision out.

The RPC is stubbed with the same transport double the RemotePolicy tests use;
what is asserted is the CONTRACT: verdict mapping, the action a tool call
becomes, fail-closed by default, and the unreadable-event path never silently
allowing anything.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any

import pytest

from tulip_frameworks import claude_code
from tulip_frameworks.gateway import RemotePolicy


class StubTransport:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes]:
        self.calls.append(
            {"method": method, "url": url, "body": json.loads(body or b"{}"), "headers": headers}
        )
        return self.status, json.dumps(self.body).encode()


def _policy(status: int, body: Any) -> tuple[RemotePolicy, StubTransport]:
    stub = StubTransport(status, body)
    return RemotePolicy("http://gw", tenant="acme", transport=stub), stub


def _event(tool: str = "Bash", **tool_input: Any) -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input or {"command": "rm -rf /prod-data"},
    }


def _verdict(out: dict[str, Any]) -> tuple[str, str]:
    inner = out["hookSpecificOutput"]
    assert inner["hookEventName"] == "PreToolUse"
    return inner["permissionDecision"], inner["permissionDecisionReason"]


def test_allow_maps_to_allow_and_the_action_crossed_the_wire() -> None:
    policy, stub = _policy(200, {"outcome": "allow", "reason": "all checks passed"})
    out = claude_code.decide(_event("Bash", command="ls -la"), policy=policy)
    decision, reason = _verdict(out)
    assert decision == "allow"
    assert "all checks passed" in reason
    sent = stub.calls[0]["body"]["action"]
    assert sent["name"] == "Bash"
    assert sent["asset"] == "ls -la"
    assert sent["kind"] == "command"


def test_hold_maps_to_ask_with_the_approval_id() -> None:
    policy, _ = _policy(
        200,
        {"outcome": "require_human", "reason": "production is gated", "approval_id": "appr-7"},
    )
    decision, reason = _verdict(claude_code.decide(_event(), policy=policy))
    assert decision == "ask"
    assert "appr-7" in reason


def test_deny_maps_to_deny_with_the_policy_reason() -> None:
    policy, _ = _policy(200, {"outcome": "deny", "reason": "irreversible actions are refused"})
    decision, reason = _verdict(claude_code.decide(_event(), policy=policy))
    assert decision == "deny"
    assert "irreversible" in reason


def test_an_unreachable_gateway_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TULIP_FAIL_OPEN", raising=False)
    policy, _ = _policy(503, {"detail": "down"})
    decision, reason = _verdict(claude_code.decide(_event(), policy=policy))
    assert decision == "deny"
    assert "fail-closed" in reason


def test_fail_open_is_a_choice_not_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TULIP_FAIL_OPEN", "1")
    policy, _ = _policy(503, {"detail": "down"})
    decision, _reason = _verdict(claude_code.decide(_event(), policy=policy))
    assert decision == "allow"


def test_an_unreadable_event_asks_the_human_instead_of_allowing() -> None:
    decision, reason = _verdict(claude_code.decide({"tool_name": "", "tool_input": None}))
    assert decision == "ask"
    assert "could not read" in reason


def test_no_gateway_url_asks_the_human(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TULIP_GATEWAY_URL", raising=False)
    decision, reason = _verdict(claude_code.decide(_event()))
    assert decision == "ask"
    assert "TULIP_GATEWAY_URL" in reason


def test_asset_extraction_prefers_the_specific_key() -> None:
    action = claude_code.action_for("Write", {"file_path": "/etc/passwd", "content": "x"})
    assert action.asset == "/etc/passwd"
    assert action.kind == "file"
    assert claude_code.action_for("WebFetch", {"url": "https://x.test"}).kind == "network"
    assert claude_code.action_for("MyTool", {}).asset == "(no asset)"


def test_main_speaks_the_stdio_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TULIP_GATEWAY_URL", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_event())))
    assert claude_code.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_main_survives_garbage_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json at all"))
    assert claude_code.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
