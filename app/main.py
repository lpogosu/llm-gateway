"""Application factory and process entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import chat, health, models, usage
from app.config import Settings
from app.container import Container, build_container, close_container
from app.errors import GatewayError
from app.logging_setup import configure_logging
from app.middleware import RequestContextMiddleware
from app.schemas import error_payload
from app.tracing import configure_tracing, shutdown_tracing

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "OpenAI-compatible gateway: rule-based routing with fallback, semantic cache, "
    "per-key token buckets and cost accounting."
)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the ASGI application.

    ``container`` is an injection point for tests, which wire fake providers and a fake
    Redis without going anywhere near the network.
    """
    resolved = settings or Settings()
    configure_logging(resolved.log_level, resolved.service_name)
    configure_tracing(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = container if container is not None else build_container(resolved)
        app.state.container = built
        logger.info(
            "gateway started",
            extra={
                "providers": sorted(built.providers),
                "routes": [route.name for route in built.router.routes],
                "cache_enabled": built.cache is not None,
                "rate_limit_enabled": built.limiter is not None,
            },
        )
        try:
            yield
        finally:
            if container is None:
                await close_container(built)
            shutdown_tracing()

    app = FastAPI(
        title="llm-gateway",
        description=DESCRIPTION,
        version="0.4.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(usage.router)
    app.include_router(health.router)

    @app.exception_handler(GatewayError)
    async def _gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_payload(), headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI answers 422 with its own envelope; OpenAI clients expect 400 with the
        # error object, and drop-in compatibility is the entire point of this service.
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
            for error in exc.errors()
        )
        return JSONResponse(
            status_code=400,
            content=error_payload(
                detail or "the request body is invalid",
                "invalid_request_error",
                "invalid_request",
            ),
        )

    return app
