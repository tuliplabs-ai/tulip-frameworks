# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Run an async gated callable from a sync framework hook (CrewAI's ``_run``)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_sync(awaitable: Awaitable[T]) -> T:
    """Run ``awaitable`` to completion from sync code, loop-running or not.

    CrewAI executes tools synchronously. If no event loop is running we use
    :func:`asyncio.run`; if one is (rare for a sync tool body), we offload to a
    worker thread so we don't try to nest loops.
    """

    async def _drive() -> T:
        return await awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_drive())

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _drive()).result()


__all__ = ["run_sync"]
