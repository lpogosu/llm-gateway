"""Request-scoped dependencies: authentication, budgets, rate limiting."""

from __future__ import annotations

import logging
from typing import Annotated, cast

from fastapi import Depends, Request
from redis.exceptions import RedisError

from app.container import Container
from app.domain import RequestContext
from app.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    InvalidRequestError,
    RateLimitedError,
)
from app.logging_setup import owner_var, request_id_var
from app.metrics import dependency_errors_total, rate_limit_rejections_total
from app.middleware import scope_slot

logger = logging.getLogger(__name__)

CACHE_BYPASS_HEADER = "X-Gateway-Cache-Bypass"
MAX_COST_HEADER = "X-Gateway-Max-Cost-Per-1k"
LATENCY_BUDGET_HEADER = "X-Gateway-Latency-Budget-Ms"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("missing bearer token in the Authorization header")
    return token.strip()


def _float_header(request: Request, name: str) -> float | None:
    raw = request.headers.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise InvalidRequestError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise InvalidRequestError(f"{name} must not be negative")
    return value


def _int_header(request: Request, name: str) -> int | None:
    value = _float_header(request, name)
    return int(value) if value is not None else None


async def authorized_context(
    request: Request,
    container: Annotated[Container, Depends(get_container)],
) -> RequestContext:
    """Authenticate the caller and charge one token to their bucket.

    Rate limiting sits in the dependency rather than in a middleware so that it only
    guards the endpoints that cost money: ``/metrics`` and the health probes are
    scraped constantly and must never be throttled.
    """
    token = _bearer_token(request)
    key_config = container.settings.resolve_api_key(token)
    if key_config is None:
        logger.warning("rejected an unknown api key")
        raise AuthenticationError("the provided api key is not recognised")

    owner_var.set(key_config.owner)
    scope_slot(request.scope)["owner"] = key_config.owner

    if container.limiter is not None:
        try:
            decision = await container.limiter.consume(
                token,
                capacity=key_config.burst,
                refill_per_second=key_config.rate_per_second,
            )
        except RedisError as exc:
            # Fail closed. An unenforced quota in front of a paid provider is a worse
            # outcome than a 503, and the caller can retry.
            dependency_errors_total.labels("ratelimit").inc()
            logger.error("rate limiting is unavailable", extra={"detail": str(exc)})
            raise DependencyUnavailableError(
                "rate limiting is temporarily unavailable; the request was not forwarded"
            ) from exc
        if not decision.allowed:
            rate_limit_rejections_total.labels(key_config.owner).inc()
            raise RateLimitedError(
                f"rate limit exceeded for {key_config.owner}: "
                f"{key_config.rate_per_second}/s with a burst of {key_config.burst}",
                retry_after_seconds=decision.retry_after_header,
            )

    return RequestContext(
        request_id=request_id_var.get(),
        api_key=token,
        owner=key_config.owner,
        bypass_cache=request.headers.get(CACHE_BYPASS_HEADER, "").lower() in _TRUE_VALUES,
        max_cost_per_1k_usd=_float_header(request, MAX_COST_HEADER),
        latency_budget_ms=_int_header(request, LATENCY_BUDGET_HEADER),
    )


ContainerDep = Annotated[Container, Depends(get_container)]
ContextDep = Annotated[RequestContext, Depends(authorized_context)]
