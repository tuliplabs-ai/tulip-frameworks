# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""`run_sync` — drive an async gated callable from a sync framework hook (CrewAI).

Both branches are covered: no running loop (asyncio.run) and inside a running loop
(offload to a worker thread)."""

from __future__ import annotations

from tulip_frameworks._sync import run_sync


def test_run_sync_with_no_running_loop() -> None:
    async def compute() -> int:
        return 42

    # A plain sync test: no event loop is running, so run_sync uses asyncio.run.
    assert run_sync(compute()) == 42


async def test_run_sync_inside_running_loop() -> None:
    async def compute() -> str:
        return "threaded"

    # This test itself runs inside pytest-asyncio's event loop, so run_sync must
    # offload to a worker thread rather than nesting loops.
    assert run_sync(compute()) == "threaded"


def test_run_sync_propagates_exception() -> None:
    async def boom() -> None:
        raise ValueError("kaboom")

    try:
        run_sync(boom())
    except ValueError as exc:
        assert str(exc) == "kaboom"
    else:  # pragma: no cover
        raise AssertionError("run_sync should have propagated the exception")
