"""Shared HTTP plumbing for provider adapters."""

from __future__ import annotations

import httpx

from app.providers.base import (
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def build_client(
    base_url: str,
    *,
    connect_timeout: float,
    request_timeout: float,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    timeout = httpx.Timeout(request_timeout, connect=connect_timeout)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
        headers=headers or {},
    )


def translate_transport_error(provider: str, exc: httpx.HTTPError) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(provider, f"request timed out: {exc}")
    return ProviderUnavailableError(provider, f"transport error: {exc}")


def error_from_response(provider: str, response: httpx.Response, body: str) -> ProviderError:
    """Map an upstream status code onto the gateway's error vocabulary.

    408 and 409 join the 5xx family because both mean "try again", while every other
    4xx is the caller's problem and must not be retried or counted against the breaker.
    """
    status = response.status_code
    snippet = body.strip()[:400]
    if status == httpx.codes.TOO_MANY_REQUESTS:
        return ProviderRateLimitedError(
            provider,
            f"upstream rate limited: {snippet}",
            status_code=status,
            retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
        )
    if status in (httpx.codes.REQUEST_TIMEOUT, httpx.codes.CONFLICT) or status >= 500:
        return ProviderUnavailableError(
            provider,
            f"upstream returned {status}: {snippet}",
            status_code=status,
        )
    return ProviderBadRequestError(
        provider,
        f"upstream rejected the request with {status}: {snippet}",
        status_code=status,
    )


def parse_retry_after(value: str | None) -> float | None:
    """Read the delta-seconds form of ``Retry-After``.

    The HTTP-date form is ignored on purpose: honouring an absolute timestamp from an
    upstream whose clock we do not control is worse than falling back to our own
    backoff.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
