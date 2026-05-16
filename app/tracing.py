"""OpenTelemetry setup.

Tracing is opt-in. With ``GATEWAY_OTEL_ENABLED=false`` the OpenTelemetry API hands
out no-op spans, so the instrumentation sprinkled through the pipeline costs an
attribute dictionary that is never built and nothing else.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import Settings

logger = logging.getLogger(__name__)

TRACER_NAME = "llm-gateway"


def configure_tracing(settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    resource = Resource.create({"service.name": settings.service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    logger.info("tracing enabled", extra={"otel_endpoint": settings.otel_endpoint})


def shutdown_tracing() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")
