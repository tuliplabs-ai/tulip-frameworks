# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Claude Code's ``PreToolUse`` hook, gated by a Tulip gateway.

``pip install tulip-frameworks`` is the whole integration: the package installs
a ``tulip-claude-gate`` console script that speaks Claude Code's hook contract —
a JSON event on stdin, a JSON permission decision on stdout — and asks the
gateway's ``POST /v1/admit`` to weigh the tool call before Claude Code runs it.
The decision is therefore RECORDED: it lands on the same server-side policy and
tamper-evident audit chain as every other agent's actions, instead of living
and dying inside one developer's session.

Wire-up, in the project's ``.claude/settings.json``::

    {
      "hooks": {
        "PreToolUse": [
          {"matcher": "*",
           "hooks": [{"type": "command", "command": "tulip-claude-gate"}]}
        ]
      }
    }

Configuration is environment variables, because a hook has no argv worth
parsing: ``TULIP_GATEWAY_URL`` (required), ``TULIP_GATEWAY_TOKEN``,
``TULIP_TENANT``, ``TULIP_POLICY_REF`` (default ``default``),
``TULIP_PRINCIPAL`` (default ``agent:claude-code``), ``TULIP_ENVIRONMENT``
(default ``dev``), and ``TULIP_FAIL_OPEN=1`` to let tool calls through when the
gateway is unreachable — the default is fail-closed, because the dangerous
actions must never be the ones that slip through by accident.

Verdict mapping: ALLOW → ``allow``; DENY → ``deny`` with the policy's reason;
REQUIRE_HUMAN → ``ask`` — Claude Code's own permission prompt becomes the
human-approval step, with the gateway's ``approval_id`` in the reason so the
hold is traceable in the queue and the chain.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from tulip.control import Action

from tulip_frameworks.gateway import GatewayError, RemotePolicy

#: Claude Code tool name → the action ``kind`` a policy matches on.
_KINDS = {
    "Bash": "command",
    "Write": "file",
    "Edit": "file",
    "NotebookEdit": "file",
    "WebFetch": "network",
    "WebSearch": "network",
}

#: tool_input keys tried, in order, for the action's ``asset``.
_ASSET_KEYS = ("file_path", "command", "url", "path", "notebook_path", "query")


def _asset(tool_input: dict[str, Any]) -> str:
    for key in _ASSET_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return "(no asset)"


def action_for(tool_name: str, tool_input: dict[str, Any]) -> Action:
    """The Claude Code tool call as the :class:`~tulip.control.Action` a policy weighs."""
    return Action(
        name=tool_name,
        asset=_asset(tool_input),
        blast_radius=1,
        kind=_KINDS.get(tool_name, "tool"),
        environment=os.environ.get("TULIP_ENVIRONMENT", "dev"),
    )


def _respond(decision: str, reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def decide(event: dict[str, Any], *, policy: RemotePolicy | None = None) -> dict[str, Any]:
    """One hook event in, one permission decision out. Pure apart from the RPC."""
    tool_name = str(event.get("tool_name") or "")
    tool_input = event.get("tool_input")
    if not tool_name or not isinstance(tool_input, dict):
        # Not a tool call this hook understands — the human decides, with the
        # reason on screen. Never silently allow what could not be weighed.
        return _respond("ask", "tulip-claude-gate could not read this event; decide by hand")
    gateway_url = os.environ.get("TULIP_GATEWAY_URL", "")
    if policy is None:
        if not gateway_url:
            return _respond("ask", "TULIP_GATEWAY_URL is not set — the gate cannot weigh this call")
        policy = RemotePolicy(
            gateway_url,
            policy_ref=os.environ.get("TULIP_POLICY_REF", "default"),
            principal=os.environ.get("TULIP_PRINCIPAL", "agent:claude-code"),
            tenant=os.environ.get("TULIP_TENANT") or None,
            token=os.environ.get("TULIP_GATEWAY_TOKEN") or None,
        )
    action = action_for(tool_name, tool_input)
    try:
        remote = asyncio.run(policy.decide(action))
    except (GatewayError, OSError) as exc:
        if os.environ.get("TULIP_FAIL_OPEN") == "1":
            return _respond("allow", f"gateway unreachable and TULIP_FAIL_OPEN=1: {exc}")
        return _respond(
            "deny",
            f"the Tulip gateway could not render a decision ({exc}) — "
            "fail-closed; set TULIP_FAIL_OPEN=1 to choose otherwise",
        )
    outcome = remote.decision.outcome
    reason = remote.decision.reason
    if outcome == "allow":
        return _respond("allow", f"admitted by policy: {reason}")
    if outcome == "require_human":
        held = f" (approval {remote.approval_id})" if remote.approval_id else ""
        return _respond("ask", f"held by policy{held}: {reason}")
    return _respond("deny", f"denied by policy: {reason}")


def main() -> int:
    """``tulip-claude-gate`` — read the event, weigh it, answer Claude Code."""
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        event = {}
    if not isinstance(event, dict):
        event = {}
    print(json.dumps(decide(event)))
    return 0


__all__ = ["action_for", "decide", "main"]
