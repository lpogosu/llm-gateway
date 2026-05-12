"""Prometheus collectors.

Label discipline: never the API key (unbounded and secret), only its ``owner``
label; never the free-form model string from the client, only the model the router
actually resolved. Both rules exist because a gateway sits in front of arbitrary
callers and is the easiest place in a platform to blow up cardinality.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# LLM latency spans four orders of magnitude: a cache hit is sub-millisecond, a long
# local generation is a minute. The default buckets top out at 10s and would put every
# real generation in +Inf.
_LATENCY_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0, 90.0)
_TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

requests_total = Counter(
    "llm_gateway_requests_total",
    "Chat completion requests served, by outcome.",
    ("route", "provider", "model", "outcome"),
    registry=REGISTRY,
)

request_duration_seconds = Histogram(
    "llm_gateway_request_duration_seconds",
    "End-to-end duration of a chat completion, gateway ingress to last byte.",
    ("route", "provider", "model"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

time_to_first_token_seconds = Histogram(
    "llm_gateway_time_to_first_token_seconds",
    "Delay between accepting a streaming request and emitting the first delta.",
    ("provider", "model"),
    buckets=_TTFT_BUCKETS,
    registry=REGISTRY,
)

tokens_total = Counter(
    "llm_gateway_tokens_total",
    "Tokens accounted, split into prompt and completion.",
    ("provider", "model", "kind"),
    registry=REGISTRY,
)

cost_usd_total = Counter(
    "llm_gateway_cost_usd_total",
    "Estimated spend derived from the pricing table in routing.yaml.",
    ("owner", "provider", "model"),
    registry=REGISTRY,
)

cache_lookups_total = Counter(
    "llm_gateway_cache_lookups_total",
    "Semantic cache lookups by result (hit, miss, bypass, skipped, error).",
    ("result",),
    registry=REGISTRY,
)

cache_similarity = Histogram(
    "llm_gateway_cache_similarity",
    "Cosine similarity of the best candidate found for a lookup, hit or not.",
    buckets=(0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0),
    registry=REGISTRY,
)

provider_errors_total = Counter(
    "llm_gateway_provider_errors_total",
    "Upstream failures by provider and error kind.",
    ("provider", "kind"),
    registry=REGISTRY,
)

provider_retries_total = Counter(
    "llm_gateway_provider_retries_total",
    "Retries issued against a provider after a retryable failure.",
    ("provider",),
    registry=REGISTRY,
)

circuit_state = Gauge(
    "llm_gateway_circuit_state",
    "Circuit breaker state per provider: 0 closed, 1 half-open, 2 open.",
    ("provider",),
    registry=REGISTRY,
)

circuit_transitions_total = Counter(
    "llm_gateway_circuit_transitions_total",
    "Circuit breaker state transitions.",
    ("provider", "to_state"),
    registry=REGISTRY,
)

rate_limit_rejections_total = Counter(
    "llm_gateway_rate_limit_rejections_total",
    "Requests rejected with 429 by the token bucket.",
    ("owner",),
    registry=REGISTRY,
)

dependency_errors_total = Counter(
    "llm_gateway_dependency_errors_total",
    "Failures talking to a backing service, by the component that hit them.",
    ("component",),
    registry=REGISTRY,
)

upstream_inflight = Gauge(
    "llm_gateway_upstream_inflight",
    "Provider calls currently in flight.",
    ("provider",),
    registry=REGISTRY,
)
