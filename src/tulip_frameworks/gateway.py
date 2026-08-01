# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The gate as a plain network call — ``POST /v1/admit`` on a Tulip gateway.

Every bridge in this package gates a tool by calling the SDK's
:func:`~tulip.control.admit` **in process**. That is the zero-infra default, but it
means the policy and the audit chain live inside the agent's own process. A
:class:`RemotePolicy` moves the *decision* to a gateway instead: the wrapper marshals
the :class:`~tulip.control.Action`, POSTs it, and runs the side effect locally only if
the gateway says ALLOW (see ``DESIGN-admit-rpc.md`` — one policy engine, thin clients).

It is a drop-in for the ``policy=`` argument every bridge already takes::

    from tulip_frameworks.gateway import RemotePolicy
    from tulip_frameworks.langchain import gate_langchain_tool

    safe = gate_langchain_tool(refund, action=..., policy=RemotePolicy("http://gw:8420"))

On ``require_human`` the gateway opens an approval; the held result carries its
``approval_id``, and :meth:`RemotePolicy.approval_state` / :meth:`RemotePolicy.consume`
drive the out-of-band lifecycle. Transport is stdlib ``urllib`` — no new dependency,
and no policy logic on this side of the wire.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import quote

from tulip.control import Action, AdmissionError, ApprovalDecision, AuditTrail

T = TypeVar("T")

#: A pluggable HTTP transport: ``(method, url, body, headers, timeout) -> (status, bytes)``.
#: Injectable so the client can be tested without a server.
Transport = Callable[[str, str, bytes | None, Mapping[str, str], float], tuple[int, bytes]]


class GatewayError(RuntimeError):
    """The gateway could not be reached, or answered with a non-2xx status."""


class RemoteAdmissionError(AdmissionError):
    """A remote gate refused the action; carries the gateway's ``approval_id``."""

    def __init__(self, decision: ApprovalDecision, approval_id: str | None) -> None:
        super().__init__(decision)
        self.approval_id = approval_id


@dataclass(frozen=True)
class RemoteDecision:
    """What ``POST /v1/admit`` answered, in SDK shape plus the gateway's ids."""

    decision: ApprovalDecision
    approval_id: str | None
    audit_id: str

    @property
    def allowed(self) -> bool:
        return self.decision.allowed


def _urllib_transport(
    method: str, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
) -> tuple[int, bytes]:
    """The default transport — stdlib only, so this package stays dependency-free."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as exc:  # a real HTTP answer, just not 2xx
        return int(exc.code), bytes(exc.read())
    except OSError as exc:  # unreachable, DNS, TLS, timeout…
        raise GatewayError(f"gateway at {url} is unreachable: {exc}") from exc


class RemotePolicy:
    """A gate that lives on a Tulip gateway, reached over HTTP.

    Args:
        base_url: The gateway's root, e.g. ``http://127.0.0.1:8420``.
        policy_ref: Names the **server-side** policy to decide against. No policy
            logic is shipped from this side.
        principal: Who is acting — recorded, and what per-principal rules key on.
        tenant: Whose approvals queue a hold lands in.
        token: Optional bearer token.
        timeout: Per-request timeout in seconds.
        transport: Override the HTTP transport (tests, custom auth, proxies).
    """

    def __init__(
        self,
        base_url: str,
        *,
        policy_ref: str = "default",
        principal: str = "agent",
        tenant: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.policy_ref = policy_ref
        self.principal = principal
        self.tenant = tenant
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self._headers: dict[str, str] = {"content-type": "application/json"}
        if token:
            self._headers["authorization"] = f"Bearer {token}"
        # Tenancy travels as the gateway's own header. The gateway resolves the
        # caller's tenant from authentication (the verified token, or this
        # header in local/dev mode) on EVERY route — submitting a hold under a
        # body-supplied tenant and reading it back under the caller's used to
        # split-brain the lifecycle: the hold existed, and approval_state()
        # could never see it.
        if tenant:
            self._headers["x-tulip-tenant"] = tenant

    # -- transport ---------------------------------------------------------

    async def _call(self, method: str, path: str, payload: Any = None) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = await asyncio.to_thread(
            self._transport,
            method,
            f"{self.base_url}{path}",
            body,
            self._headers,
            self.timeout,
        )
        if status // 100 != 2:
            raise GatewayError(f"{method} {path} -> HTTP {status}: {raw.decode(errors='replace')}")
        return json.loads(raw or b"null")

    # -- the gate ----------------------------------------------------------

    async def decide(self, action: Action, *, idempotency_key: str | None = None) -> RemoteDecision:
        """Ask the gateway to weigh ``action``. Returns the decision; runs nothing."""
        payload: dict[str, Any] = {
            "principal": self.principal,
            "policy_ref": self.policy_ref,
            "action": {
                "name": action.name,
                "asset": action.asset,
                "blast_radius": action.blast_radius,
                "environment": action.environment,
                "kind": action.kind,
                "tags": sorted(action.tags),
            },
        }
        if self.tenant is not None:
            payload["tenant"] = self.tenant
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        data = await self._call("POST", "/v1/admit", payload)
        return RemoteDecision(
            decision=ApprovalDecision(
                outcome=str(data["outcome"]),
                reason=str(data["reason"]),
                action=action,
                checks=list(data.get("checks") or []),
            ),
            approval_id=data.get("approval_id"),
            audit_id=str(data.get("audit_id", "")),
        )

    async def admit(
        self,
        action: Action,
        perform: Callable[[], Awaitable[T]],
        *,
        trail: AuditTrail | None = None,
    ) -> T:
        """:func:`~tulip.control.admit`, split across the wire.

        The decision is the gateway's; ``perform`` stays local and runs only on ALLOW.
        Mirrors the in-process gate's trail record exactly, so an agent that moves from
        the local gate to the remote one keeps the same audit shape.
        """
        remote = await self.decide(action)
        if trail is not None:
            trail.record(
                "action-admission",
                {
                    "action": action.name,
                    "asset": action.asset,
                    "outcome": remote.decision.outcome,
                    "reason": remote.decision.reason,
                },
            )
        if not remote.allowed:
            raise RemoteAdmissionError(remote.decision, remote.approval_id)
        return await perform()

    # -- out-of-band approval lifecycle ------------------------------------

    async def approval_state(self, approval_id: str) -> str | None:
        """Poll a held action: ``pending`` / ``approved`` / ``denied`` / ``consumed``."""
        try:
            data = await self._call("GET", f"/v1/admit/approval/{quote(approval_id)}")
        except GatewayError:
            return None
        state: str | None = data.get("state")
        return state

    async def consume(self, approval_id: str) -> bool:
        """Claim an approved hold. Single-use: a second claim returns ``False``."""
        try:
            await self._call("POST", f"/v1/admit/approval/{quote(approval_id)}/consume", {})
        except GatewayError:
            return False
        return True


__all__ = [
    "GatewayError",
    "RemoteAdmissionError",
    "RemoteDecision",
    "RemotePolicy",
    "Transport",
]
