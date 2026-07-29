# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Drop Tulip's control runtime into the agent framework you already use.

``import tulip_frameworks`` exposes only the framework-agnostic core — it pulls in
**no** third-party framework package. Import a bridge explicitly to use it::

    from tulip_frameworks.langchain import gate_langchain_tool   # needs [langchain]
    from tulip_frameworks.openai_agents import gate_openai_tool  # needs [openai-agents]

Each bridge wraps the same primitive, :func:`tulip_frameworks.core.gate_callable`, so
the action your agent already takes runs only after it clears the admission gate.
"""

from __future__ import annotations

from tulip_frameworks.actions import (
    ActionSpec,
    asset_from_args,
    default_action,
    resolve_action,
)
from tulip_frameworks.approval import ApprovalBridge, gateway_bridge
from tulip_frameworks.core import (
    DENIED,
    HELD,
    GatePolicy,
    Mode,
    VerificationResult,
    gate_callable,
    held_result,
    is_held,
)
from tulip_frameworks.gateway import RemotePolicy
from tulip_frameworks.policy_presets import action_gate_policy

__version__ = "0.1.0"

__all__ = [
    "DENIED",
    "HELD",
    "ActionSpec",
    "ApprovalBridge",
    "GatePolicy",
    "Mode",
    "RemotePolicy",
    "VerificationResult",
    "__version__",
    "action_gate_policy",
    "asset_from_args",
    "default_action",
    "gate_callable",
    "gateway_bridge",
    "held_result",
    "is_held",
    "resolve_action",
]
