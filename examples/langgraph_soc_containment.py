#!/usr/bin/env python
"""A LangGraph SOC agent whose containment tool is gated by Tulip.

This is the LangGraph port of the SDK's ``governed_soc_action`` demo: keep your
graph, wrap the one tool that *acts*, and a production host-isolation is held for a
human with a forge-proof record — proof that the value is the control runtime, not
the model.

Run: pip install "tulip-frameworks[langgraph]" && python langgraph_soc_containment.py
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.tools import tool
from tulip.control import Action, AuditTrail

from tulip_frameworks.langchain import gate_langchain_tool
from tulip_frameworks.policy_presets import action_gate_policy


@tool
def isolate_host(host: str) -> str:
    """Network-contain a host (EDR/firewall action)."""
    return f"isolated {host}"


async def main() -> None:
    trail = AuditTrail()

    contain = gate_langchain_tool(
        isolate_host,
        action=lambda name, a: Action(
            name=name, asset=a["host"], blast_radius=1,
            kind="containment", environment="production",
        ),
        policy=action_gate_policy(),
        trail=trail,
    )
    # `contain` is a plain LangChain tool — drop it into a LangGraph ToolNode /
    # create_react_agent exactly as you would the original `isolate_host`.

    # Triage decided to isolate a prod host. The gate holds it for a human:
    held = await contain.ainvoke({"host": "prod-db-01"})
    print("prod action:", json.loads(held)["status"])   # held_for_approval

    # A dev host clears policy and runs:
    dev = gate_langchain_tool(
        isolate_host,
        action=lambda name, a: Action(name=name, asset=a["host"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    print("dev action: ", await dev.ainvoke({"host": "dev-7"}))  # isolated dev-7

    print("\naudit chain intact:", trail.verify())
    print(trail.export_jsonl())


if __name__ == "__main__":
    asyncio.run(main())
