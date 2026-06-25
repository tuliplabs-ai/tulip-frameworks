#!/usr/bin/env python
"""Gate a Google ADK tool's action behind Tulip's admission gate.

Your ADK agent has a tool that disables accounts. Wrap it once and a production
disable is held for a human — with a tamper-evident record either way.

Run: pip install "tulip-frameworks[adk]" && python adk_account_gate.py
(No LLM/API key needed — we invoke the gated tool directly to show the gate.)
"""

from __future__ import annotations

import asyncio
import json

from google.adk.tools import FunctionTool
from tulip.security import Action, AuditTrail

from tulip_frameworks.adk import gate_adk_tool
from tulip_frameworks.policy_presets import action_gate_policy


def disable_account(email: str) -> str:
    """Disable a user account by email."""
    return f"disabled {email}"


async def main() -> None:
    trail = AuditTrail()
    gated = gate_adk_tool(
        FunctionTool(func=disable_account),
        action=lambda name, a: Action(
            name=name, asset=a["email"], kind="identity", environment="production"
        ),
        policy=action_gate_policy(),  # production -> human
        trail=trail,
    )
    # `gated` is an ADK FunctionTool — drop it into Agent(tools=[gated]) unchanged.
    # ADK still sees the real `email` parameter (signature is preserved).

    held = json.loads(await gated.func(email="mallory@corp.com"))
    print("prod disable:", held["status"], "-", held["reason"])  # held_for_approval

    dev = gate_adk_tool(
        FunctionTool(func=disable_account),
        action=lambda name, a: Action(name=name, asset=a["email"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    print("dev disable :", await dev.func(email="test@corp.com"))

    print("\naudit chain intact:", trail.verify())
    print(trail.export_jsonl())


if __name__ == "__main__":
    asyncio.run(main())
