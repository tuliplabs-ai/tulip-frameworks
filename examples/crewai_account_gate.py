#!/usr/bin/env python
"""Gate a CrewAI tool's action behind Tulip's admission gate.

Your CrewAI crew has a tool that disables accounts. Wrap it once and a production
disable is held for a human — and the decision lands on a tamper-evident trail.

Run: pip install "tulip-frameworks[crewai]" && python crewai_account_gate.py
(No LLM/API key needed — we invoke the gated tool directly to show the gate.)
"""

from __future__ import annotations

import json

from crewai.tools import BaseTool
from tulip.security import Action, AuditTrail

from tulip_frameworks.crewai import gate_crewai_tool
from tulip_frameworks.policy_presets import action_gate_policy


class DisableAccount(BaseTool):
    name: str = "disable_account"
    description: str = "Disable a user account by email."

    def _run(self, email: str) -> str:
        # In real life this calls your IdP. Here we prove the gate ran first.
        return f"disabled {email}"


def main() -> None:
    trail = AuditTrail()
    gated = gate_crewai_tool(
        DisableAccount(),
        action=lambda name, a: Action(
            name=name, asset=a["email"], kind="identity", environment="production"
        ),
        policy=action_gate_policy(),  # production -> human
        trail=trail,
    )
    # `gated` is a CrewAI BaseTool — drop it into agent(tools=[gated]) unchanged.

    held = json.loads(gated._run(email="mallory@corp.com"))
    print("prod disable:", held["status"], "-", held["reason"])  # held_for_approval

    # A dev-environment action clears policy and runs:
    dev = gate_crewai_tool(
        DisableAccount(),
        action=lambda name, a: Action(name=name, asset=a["email"], environment="dev"),
        policy=action_gate_policy(),
        trail=trail,
    )
    print("dev disable :", dev._run(email="test@corp.com"))

    print("\naudit chain intact:", trail.verify())
    print(trail.export_jsonl())


if __name__ == "__main__":
    main()
