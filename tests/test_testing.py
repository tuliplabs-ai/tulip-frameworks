# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The offline test helpers themselves — `Spy`, `assert_allowed`, `assert_held`.

These are what downstream users import to test their own gated tools, so they get
their own coverage."""

from __future__ import annotations

import pytest
from tulip.control import Action

from tulip_frameworks import action_gate_policy, gate_callable
from tulip_frameworks.testing import Spy, assert_allowed, assert_held


def test_spy_records_calls_and_ran() -> None:
    spy = Spy(result="r")
    assert spy.ran is False
    out = spy(a=1, b=2)
    assert out == "r"
    assert spy.ran is True
    assert spy.calls == [{"a": 1, "b": 2}]


async def test_assert_allowed_on_run() -> None:
    spy = Spy(result="ok")
    gated = gate_callable(
        spy,
        name="read",
        action=Action(name="read", asset="x", environment="dev"),
        policy=action_gate_policy(),
    )
    result = await gated(asset="x")
    assert_allowed(result, spy)  # ran, and result == spy.result


async def test_assert_held_dict_form() -> None:
    spy = Spy()
    gated = gate_callable(
        spy,
        name="refund",
        action=Action(name="refund", asset="x", environment="production"),
        policy=action_gate_policy(),
    )
    result = await gated(asset="x")
    assert_held(spy=spy, result=result)


async def test_assert_held_json_string_form() -> None:
    spy = Spy()
    gated = gate_callable(
        spy,
        name="refund",
        action=Action(name="refund", asset="x", environment="production"),
        policy=action_gate_policy(),
        serialize=True,
    )
    result = await gated(asset="x")
    assert isinstance(result, str)
    assert_held(result, spy)  # decodes the JSON string before asserting


async def test_assert_held_denied_outcome() -> None:
    spy = Spy()
    gated = gate_callable(
        spy,
        name="wipe",
        action=Action(name="wipe", tags=frozenset({"irreversible"})),
        policy=action_gate_policy(),
    )
    result = await gated()
    assert_held(result, spy, outcome="denied")


async def test_assert_allowed_fails_when_held() -> None:
    spy = Spy()
    gated = gate_callable(
        spy,
        name="refund",
        action=Action(name="refund", environment="production"),
        policy=action_gate_policy(),
    )
    result = await gated()
    with pytest.raises(AssertionError):
        assert_allowed(result, spy)  # it was held, not allowed
