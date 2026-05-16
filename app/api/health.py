"""Probes and the Prometheus scrape endpoint.

None of these require an API key: a kubelet does not carry credentials, and a metrics
endpoint that fails authentication is a metrics endpoint nobody scrapes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.exceptions import RedisError

from app.api.deps import ContainerDep
from app.metrics import REGISTRY
from app.redis_support import resolve
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """The process is up. Deliberately checks nothing else: a liveness probe that
    depends on Redis restarts the gateway every time Redis hiccups."""
    return HealthResponse(status="ok", checks={"process": "ok"})


@router.get("/health/ready")
async def readiness(container: ContainerDep, response: Response) -> HealthResponse:
    """Ready means "can serve a request end to end", which requires Redis.

    An open circuit is reported but does not make the pod unready: the fallback chain
    exists precisely so that one broken provider does not take the gateway down.
    """
    checks: dict[str, str] = {}
    healthy = True
    try:
        await resolve(container.redis.ping())
        checks["redis"] = "ok"
    except RedisError as exc:
        healthy = False
        checks["redis"] = f"error: {exc}"
        logger.error("readiness probe failed on redis", extra={"detail": str(exc)})

    for name, state in container.breakers.snapshot().items():
        checks[f"circuit:{name}"] = state.value

    if not healthy:
        response.status_code = 503
    return HealthResponse(status="ok" if healthy else "degraded", checks=checks)


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
