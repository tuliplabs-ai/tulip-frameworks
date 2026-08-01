# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The cross-process gate — ``RemotePolicy`` against a stub, and over real HTTP.

No live gateway is needed here: the wire contract is pinned against a stub transport,
and the default stdlib transport is exercised against a throwaway ``http.server`` so
the code that actually talks to a gateway is covered too. The live-gateway proof is
``tests/integration/test_admit_rpc_live.py``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from tulip.control import Action, AuditTrail

from tulip_frameworks.core import gate_callable
from tulip_frameworks.gateway import (
    GatewayError,
    RemoteAdmissionError,
    RemotePolicy,
    _urllib_transport,
)

ALLOW = {
    "outcome": "allow",
    "allowed": True,
    "reason": "all policy checks passed",
    "checks": [],
    "approval_id": None,
    "audit_id": "aud-1",
}
HOLD = {
    "outcome": "require_human",
    "allowed": False,
    "reason": "labels ['production'] require human approval",
    "checks": ["labels ['production'] require human approval"],
    "approval_id": "appr-1",
    "audit_id": "aud-2",
}


class StubTransport:
    """Records every request and replies from a canned map of ``METHOD path``."""

    def __init__(self, replies: Mapping[str, tuple[int, Any]]) -> None:
        self.replies = dict(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes]:
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body) if body else None,
            }
        )
        key = f"{method} /{path.split('?')[0]}"
        status, payload = self.replies[key]
        return status, json.dumps(payload).encode()


def _refund() -> Action:
    return Action(
        name="refund",
        asset="ord-9",
        blast_radius=2,
        environment="production",
        kind="payment",
        tags=frozenset({"payment-rail"}),
    )


async def test_decide_marshals_the_action_and_names_a_server_side_policy() -> None:
    stub = StubTransport({"POST /v1/admit": (200, ALLOW)})
    policy = RemotePolicy(
        "http://gw:8420/",
        policy_ref="strict",
        principal="agent:test",
        tenant="acme",
        token="secret",
        transport=stub,
    )

    remote = await policy.decide(_refund(), idempotency_key="key-1")

    assert remote.allowed is True
    assert remote.audit_id == "aud-1"
    sent = stub.calls[0]
    assert sent["url"] == "http://gw:8420/v1/admit"
    assert sent["headers"]["authorization"] == "Bearer secret"
    assert sent["body"] == {
        "principal": "agent:test",
        "policy_ref": "strict",
        "tenant": "acme",
        "idempotency_key": "key-1",
        # Every label the policy reasons over crosses the wire — this is the field set
        # whose loss elsewhere let a production payment through.
        "action": {
            "name": "refund",
            "asset": "ord-9",
            "blast_radius": 2,
            "environment": "production",
            "kind": "payment",
            "tags": ["payment-rail"],
        },
    }


async def test_admit_runs_the_side_effect_only_on_allow() -> None:
    stub = StubTransport({"POST /v1/admit": (200, ALLOW)})
    trail = AuditTrail()
    ran: list[str] = []

    async def perform() -> str:
        ran.append("yes")
        return "done"

    out = await RemotePolicy("http://gw", transport=stub).admit(_refund(), perform, trail=trail)

    assert out == "done"
    assert ran == ["yes"]
    assert trail.verify() is True
    assert trail.records()[0].payload["outcome"] == "allow"


async def test_admit_holds_and_records_without_running() -> None:
    stub = StubTransport({"POST /v1/admit": (200, HOLD)})
    trail = AuditTrail()
    ran: list[str] = []

    async def perform() -> str:  # pragma: no cover - must never run
        ran.append("yes")
        return "done"

    with pytest.raises(RemoteAdmissionError) as excinfo:
        await RemotePolicy("http://gw", transport=stub).admit(_refund(), perform, trail=trail)

    assert ran == []
    assert excinfo.value.approval_id == "appr-1"
    record = trail.records()[0].payload
    assert record == {
        "action": "refund",
        "asset": "ord-9",
        "outcome": "require_human",
        "reason": HOLD["reason"],
    }


async def test_gated_tool_uses_the_remote_gate_and_surfaces_the_approval_id() -> None:
    stub = StubTransport({"POST /v1/admit": (200, HOLD)})
    ran: list[str] = []

    def refund(order_id: str) -> str:  # pragma: no cover - must never run
        ran.append(order_id)
        return "refunded"

    gated = gate_callable(
        refund,
        name="refund",
        action=lambda n, a: _refund(),
        policy=RemotePolicy("http://gw", transport=stub),
    )
    out = await gated(order_id="ord-9")

    assert ran == []
    assert out["status"] == "held_for_approval"
    assert out["approval_id"] == "appr-1"
    assert out["next"].startswith("call approval_status")


async def test_gated_tool_raise_mode_propagates_the_remote_error() -> None:
    stub = StubTransport({"POST /v1/admit": (200, HOLD)})
    gated = gate_callable(
        lambda order_id: "refunded",
        name="refund",
        action=lambda n, a: _refund(),
        policy=RemotePolicy("http://gw", transport=stub),
        mode="raise",
    )
    with pytest.raises(RemoteAdmissionError):
        await gated(order_id="ord-9")


async def test_non_2xx_raises_gateway_error() -> None:
    stub = StubTransport({"POST /v1/admit": (404, {"detail": "unknown policy_ref"})})
    with pytest.raises(GatewayError, match="404"):
        await RemotePolicy("http://gw", transport=stub).decide(_refund())


async def test_approval_lifecycle_is_pollable_and_single_use() -> None:
    stub = StubTransport(
        {
            "GET /v1/admit/approval/appr-1": (200, {"state": "approved"}),
            "POST /v1/admit/approval/appr-1/consume": (200, {"state": "consumed"}),
        }
    )
    policy = RemotePolicy("http://gw", tenant="acme", transport=stub)

    assert await policy.approval_state("appr-1") == "approved"
    assert await policy.consume("appr-1") is True
    # Tenancy travels as the gateway's own header on every call — the old
    # ?tenant= query param is dead server-side and no longer sent.
    assert stub.calls[0]["url"].endswith("/v1/admit/approval/appr-1")
    assert stub.calls[0]["headers"]["x-tulip-tenant"] == "acme"

    stub.replies["POST /v1/admit/approval/appr-1/consume"] = (409, {"detail": "already consumed"})
    stub.replies["GET /v1/admit/approval/appr-1"] = (404, {"detail": "unknown approval_id"})
    assert await policy.consume("appr-1") is False
    assert await policy.approval_state("appr-1") is None


# --- the real stdlib transport ------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = HOLD if body["action"]["environment"] == "production" else ALLOW
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: Any) -> None:  # keep the test output quiet
        return


@pytest.fixture
def http_gateway() -> Any:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


async def test_default_transport_speaks_real_http(http_gateway: str) -> None:
    """The stdlib transport — the one that talks to a real gateway — over real HTTP."""
    policy = RemotePolicy(http_gateway)

    held = await policy.decide(_refund())
    allowed = await policy.decide(Action(name="read_docs", asset="doc-1", environment="dev"))

    assert held.allowed is False
    assert held.approval_id == "appr-1"
    assert allowed.allowed is True


async def test_default_transport_reports_an_unreachable_gateway() -> None:
    # Port 1 on loopback: nothing listens, so the connection is refused immediately.
    with pytest.raises(GatewayError, match="unreachable"):
        await RemotePolicy("http://127.0.0.1:1", timeout=2.0).decide(_refund())


def test_transport_returns_the_body_of_a_non_2xx_answer(http_gateway: str) -> None:
    """A non-2xx is an answer, not a transport failure — the client must see its body."""
    status, body = _urllib_transport("GET", f"{http_gateway}/nope", None, {}, 5.0)
    assert status == 501  # the stub server implements POST only
    assert body  # the server's error page, passed through for the error message
