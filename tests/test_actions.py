# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Action derivation — offline."""

from __future__ import annotations

from tulip.control import Action

from tulip_frameworks.actions import asset_from_args, default_action, resolve_action


def test_asset_from_args_priority() -> None:
    assert asset_from_args({"host": "h1", "email": "e@x"}) == "h1"
    assert asset_from_args({"email": "e@x"}) == "e@x"
    assert asset_from_args({"unknown": "z"}) == ""


def test_default_action_is_failsafe() -> None:
    act = default_action("isolate", {"host": "prod-1"})
    assert act.name == "isolate"
    assert act.asset == "prod-1"
    assert act.environment == "unknown"  # fail-safe: unknown env, conservative blast radius
    assert act.blast_radius == 1


def test_resolve_action_constant_and_callable() -> None:
    constant = Action(name="x", asset="a")
    assert resolve_action(constant, "x", {}) is constant

    derived = resolve_action(
        lambda name, a: Action(name=name, asset=a["host"]), "scan", {"host": "h9"}
    )
    assert derived.asset == "h9"

    assert resolve_action(None, "fallback", {"target": "t1"}).asset == "t1"
