"""Errors that reach the client, rendered in the OpenAI error envelope.

Clients written against the OpenAI SDK inspect ``error.type`` and ``error.code``.
Inventing a different envelope would break exactly the drop-in compatibility that
this gateway exists to provide.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """Base class for failures with a defined HTTP rendering."""

    status_code = 500
    error_type = "server_error"
    code: str | None = None

    def __init__(self, message: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.headers = headers or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
                "param": None,
            }
        }


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class ModelNotFoundError(GatewayError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class NoRouteError(GatewayError):
    """The model is known but no target satisfies the caller's cost/latency budget."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "no_eligible_target"


class RateLimitedError(GatewayError):
    status_code = 429
    error_type = "rate_limit_error"
    code = "rate_limit_exceeded"

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message, headers={"Retry-After": str(retry_after_seconds)})
        self.retry_after_seconds = retry_after_seconds


class DependencyUnavailableError(GatewayError):
    """A backing service the request cannot proceed without is unreachable.

    Used for Redis on the rate-limiting path only. The gateway fails closed there: if it
    cannot enforce a caller's quota it must not forward the request to a paid provider.
    Everywhere else a Redis failure degrades instead — see ``ChatService``.
    """

    status_code = 503
    error_type = "api_error"
    code = "dependency_unavailable"


class UpstreamError(GatewayError):
    status_code = 502
    error_type = "api_error"
    code = "upstream_error"


class AllTargetsFailedError(GatewayError):
    """Every target in the fallback chain was rejected, open or failing."""

    status_code = 503
    error_type = "api_error"
    code = "no_healthy_target"

    def __init__(self, message: str, *, attempted: list[str]) -> None:
        super().__init__(message)
        self.attempted = attempted

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["error"]["attempted_targets"] = self.attempted
        return payload
