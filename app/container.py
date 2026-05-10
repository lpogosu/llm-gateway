"""Composition root.

Everything is constructed once at startup and handed to the request layer through
``app.state``. No module reaches for a global client, which is what makes the whole
pipeline constructible in a test with fakes in place of Redis and the providers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.asyncio import Redis

from app.accounting.usage import UsageRecorder
from app.cache.embeddings import Embedder, OllamaEmbedder
from app.cache.semantic import SemanticCache
from app.config import Settings
from app.metrics import circuit_state, circuit_transitions_total
from app.providers.base import Provider
from app.providers.ollama import PROVIDER_NAME as OLLAMA
from app.providers.ollama import OllamaProvider
from app.providers.openrouter import PROVIDER_NAME as OPENROUTER
from app.providers.openrouter import OpenRouterProvider
from app.ratelimit.bucket import TokenBucketLimiter
from app.resilience.circuit import STATE_CODE, BreakerRegistry, CircuitState
from app.resilience.retry import RetryPolicy
from app.routing.config import load_routing_config
from app.routing.rules import RoutingTable
from app.service import ChatService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    settings: Settings
    redis: Redis
    router: RoutingTable
    providers: dict[str, Provider]
    breakers: BreakerRegistry
    limiter: TokenBucketLimiter | None
    recorder: UsageRecorder
    cache: SemanticCache | None
    embedder: Embedder | None
    service: ChatService


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Instantiate the backends that are actually configured.

    A provider without credentials is left out rather than registered and left to fail:
    the router logs and skips an unconfigured target, which is a clearer failure than a
    stream of upstream 401s opening a circuit.
    """
    providers: dict[str, Provider] = {
        OLLAMA: OllamaProvider(
            settings.ollama_base_url,
            connect_timeout=settings.connect_timeout_seconds,
            request_timeout=settings.request_timeout_seconds,
        )
    }
    if settings.openrouter_api_key is not None:
        providers[OPENROUTER] = OpenRouterProvider(
            settings.openrouter_base_url,
            settings.openrouter_api_key.get_secret_value(),
            connect_timeout=settings.connect_timeout_seconds,
            request_timeout=settings.request_timeout_seconds,
            referer=settings.openrouter_referer,
        )
    else:
        logger.warning("openrouter is not configured; routes pointing at it will be skipped")
    return providers


def _on_circuit_transition(provider: str, state: CircuitState) -> None:
    circuit_state.labels(provider).set(STATE_CODE[state])
    circuit_transitions_total.labels(provider, state.value).inc()
    logger.warning("circuit breaker changed state", extra={"provider": provider, "state": state})


def build_container(settings: Settings) -> Container:
    routing_config = load_routing_config(settings.routing_config_path)
    router = RoutingTable(routing_config)
    providers = build_providers(settings)

    # Binary values (packed float32 vectors) live in Redis, so responses stay as bytes
    # and every string is decoded explicitly at the point of use.
    redis = Redis.from_url(settings.redis_url, decode_responses=False)

    breakers = BreakerRegistry(
        failure_threshold=settings.breaker_failure_threshold,
        success_threshold=settings.breaker_success_threshold,
        recovery_seconds=settings.breaker_recovery_seconds,
        half_open_max_calls=settings.breaker_half_open_max_calls,
        on_transition=_on_circuit_transition,
    )
    for name in providers:
        # Publish a closed circuit up front; a gauge that only appears after the first
        # failure is a gauge nobody can alert on.
        circuit_state.labels(name).set(STATE_CODE[CircuitState.CLOSED])
        breakers.get(name)

    embedder: Embedder | None = None
    cache: SemanticCache | None = None
    if settings.cache_enabled:
        embedder = OllamaEmbedder(
            settings.embedding_base_url,
            settings.embedding_model,
            timeout=settings.embedding_timeout_seconds,
        )
        cache = SemanticCache(
            redis,
            ttl_seconds=settings.cache_ttl_seconds,
            threshold=settings.cache_similarity_threshold,
            max_candidates=settings.cache_max_candidates,
            index_size=settings.cache_index_size,
        )

    recorder = UsageRecorder(redis, retention_days=settings.usage_retention_days)
    limiter = TokenBucketLimiter(redis) if settings.rate_limit_enabled else None

    service = ChatService(
        router=router,
        providers=providers,
        breakers=breakers,
        retry_policy=RetryPolicy(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay_seconds,
            max_delay=settings.retry_max_delay_seconds,
            jitter=settings.retry_jitter,
        ),
        cache=cache,
        embedder=embedder,
        cache_max_temperature=settings.cache_max_temperature,
        embedding_model=settings.embedding_model,
        recorder=recorder,
    )
    return Container(
        settings=settings,
        redis=redis,
        router=router,
        providers=providers,
        breakers=breakers,
        limiter=limiter,
        recorder=recorder,
        cache=cache,
        embedder=embedder,
        service=service,
    )


async def close_container(container: Container) -> None:
    for provider in container.providers.values():
        await provider.aclose()
    if container.embedder is not None:
        await container.embedder.aclose()
    await container.redis.aclose()
