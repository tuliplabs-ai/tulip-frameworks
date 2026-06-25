#!/usr/bin/env python
"""Gate a LangChain tool's action behind Tulip's admission gate.

Your LangChain agent already has a ``refund`` tool. Wrap it once and a production
refund is held for a human — and the decision lands on a tamper-evident trail.

Run: pip install "tulip-frameworks[langchain]" && python langchain_refund_gate.py
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.tools import tool
from tulip.security import Action, AuditTrail

from tulip_frameworks.langchain import gate_langchain_tool
from tulip_frameworks.policy_presets import action_gate_policy


@tool
def refund(order_id: str, amount_usd: float) -> str:
    """Issue a customer refund."""
    # In real life this moves money. Here we just prove the gate ran first.
    return f"refunded {order_id} (${amount_usd})"


async def main() -> None:
    trail = AuditTrail()
    policy = action_gate_policy()  # production -> human

    safe_refund = gate_langchain_tool(
        refund,
        action=lambda name, a: Action(
            name=name, asset=a["order_id"], blast_radius=1,
            kind="payment", environment="production",
        ),
        policy=policy,
        trail=trail,
    )

    # The agent (any LangChain/LangGraph runtime) would call this. We call it
    # directly to show the gate decision without spending an LLM token.
    result = await safe_refund.ainvoke({"order_id": "ord-4821", "amount_usd": 250.0})
    print("tool result:", json.loads(result))   # -> status: held_for_approval

    print("\naudit chain intact:", trail.verify())
    print(trail.export_jsonl())


if __name__ == "__main__":
    asyncio.run(main())
