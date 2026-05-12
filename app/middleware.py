"""Raw ASGI middleware for request identity, access logging and latency.

Written against the ASGI interface rather than Starlette's ``BaseHTTPMiddleware``
because that base class buffers the response through a memory stream, which breaks the
two things this gateway cares about most: streaming a token the instant it arrives, and
noticing that the client hung up.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_setup import owner_var, request_id_var
from app.metrics import request_duration_seconds
from app.tracing import current_trace_id, get_tracer

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "x-request-id"
# Handlers publish their routing labels here once they are known; the middleware picks
# them up when the response body is finished, which for a stream is the only moment the
# end-to-end duration is actually known.
SCOPE_SLOT = "gateway"


def scope_slot(scope: Scope) -> dict[str, Any]:
    slot = scope.get(SCOPE_SLOT)
    if not isinstance(slot, dict):
        slot = {}
        scope[SCOPE_SLOT] = slot
    return slot


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        slot = scope_slot(scope)
        slot["request_id"] = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        tracer = get_tracer()
        path = scope.get("path", "")
        method = scope.get("method", "")
        try:
            with tracer.start_as_current_span(f"{method} {path}") as span:
                span.set_attribute("http.request.method", method)
                span.set_attribute("url.path", path)
                await self._app(scope, receive, send_wrapper)
                span.set_attribute("http.response.status_code", status_code)
        finally:
            duration = time.perf_counter() - started
            _observe(slot, duration)
            logger.info(
                "request completed",
                extra={
                    "method": method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "trace_id": current_trace_id(),
                    **{k: v for k, v in slot.items() if k != "request_id"},
                },
            )
            request_id_var.reset(token)
            owner_var.set("-")


def _observe(slot: dict[str, Any], duration: float) -> None:
    route = slot.get("route")
    provider = slot.get("provider")
    model = slot.get("model")
    if route and provider and model:
        request_duration_seconds.labels(str(route), str(provider), str(model)).observe(duration)


def _incoming_request_id(scope: Scope) -> str | None:
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.decode("latin-1").lower() == REQUEST_ID_HEADER:
            value = raw_value.decode("latin-1").strip()
            # An upstream proxy may forward a header the client controls; cap it so a
            # log line cannot be stuffed with a kilobyte of caller-supplied text.
            return value[:64] or None
    return None
