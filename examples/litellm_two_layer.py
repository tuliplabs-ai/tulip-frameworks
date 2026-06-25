#!/usr/bin/env python
"""LiteLLM + Tulip — two layers, not a gate.

LiteLLM is NOT a framework whose tools you gate. It's a *model-call* gateway
(routing, virtual keys, budgets, cost, fallback) — the same layer Portkey lives
on. Tulip lives on a different layer: it gates the *action* the model decides to
take. They COMPOSE — they don't compete:

    model call  ──>  LiteLLM gateway   (which model? whose key? within budget?)
    the action  ──>  Tulip admit()     (allowed? needs a human? on the record?)

So you wire LiteLLM as your model and Tulip as your action gate. This script
runs the action layer for real (offline, no key); the model layer is shown as
config and runs only if a gateway URL is set.

Run: python litellm_two_layer.py     (needs only tulip-agents)
"""

from __future__ import annotations

import asyncio
import os

from tulip.control import Action, AdmissionError, AuditTrail

from tulip_frameworks.policy_presets import action_gate_policy


def build_model() -> object | None:
    """Model layer: route every model call through the LiteLLM gateway.

    The gateway holds the upstream provider keys; the agent only holds a
    gateway-issued virtual key. Nothing here is Tulip-specific — it's just an
    OpenAI-shaped base_url, because the LiteLLM gateway is OpenAI-shaped.
    """
    url = os.environ.get("LITELLM_GATEWAY_URL")
    if not url:
        print("model layer  : (set LITELLM_GATEWAY_URL + LITELLM_KEY to route live)")
        print("               OpenAIModel(base_url=<litellm>, api_key=<virtual key>)")
        return None
    from tulip.models.native.openai import OpenAIModel

    print(f"model layer  : routing model calls through LiteLLM at {url}")
    return OpenAIModel(model="gpt-4o", base_url=url, api_key=os.environ.get("LITELLM_KEY", "sk-"))


async def main() -> None:
    build_model()  # the model layer (LiteLLM) — orthogonal to the gate below

    # Action layer: the model decided to refund. LiteLLM has no opinion on that —
    # Tulip does. Production payment -> held for a human, on a forge-proof trail.
    trail = AuditTrail()
    policy = action_gate_policy()
    refund = Action(
        name="refund", asset="cust:4821", blast_radius=1,
        kind="payment", environment="production",
    )

    async def do_refund() -> str:
        return "refunded cust:4821"

    print("action layer : admit() gating the refund...")
    try:
        from tulip.control import admit

        await admit(refund, do_refund, policy=policy, trail=trail)
        print("               -> executed")
    except AdmissionError as e:
        print(f"               -> {e.decision.outcome}: refund NOT run")

    print("\naudit chain intact:", trail.verify())
    print(trail.export_jsonl())


if __name__ == "__main__":
    asyncio.run(main())
