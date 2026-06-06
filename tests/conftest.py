"""Shared fixtures.

The whole gateway is assembled here from fakes: an in-process Redis, scripted providers
and a scripted embedder. Nothing in the suite touches the network, so the tests run the
same on a laptop and in CI with no services started.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import fakeredis.aioredis
import pytest
from redis.asyncio import Redis

from app.accounting.usage import UsageRecorder
from app.cache.semantic import SemanticCache
from app.config import ApiKeyConfig, Settings
from app.container import Container
from app.domain import RequestContext
from app.ratelimit.bucket import TokenBucketLimiter
from app.resilience.circuit import BreakerRegistry
from app.resilience.retry import RetryPolicy
from app.routing.config import parse_routing_config
from app.routing.rules import RoutingTable
from app.service import ChatService
from tests.fakes import FakeEmbedder, FakeProvider

ROUTING_FIXTURE: dict[str, Any] = {
    "version": 1,
    "catalog": [
        {
            "provider": "alpha",
            "model": "alpha-small",
            "input_cost_per_1k_usd": 0.0,
            "output_cost_per_1k_usd": 0.0,
            "p95_latency_ms": 800,
        },
        {
            "provider": "alpha",
            "model": "alpha-large",
            "input_cost_per_1k_usd": 0.001,
            "output_cost_per_1k_usd": 0.002,
            "p95_latency_ms": 5000,
        },
        {
            "provider": "beta",
            "model": "beta-mid",
            "input_cost_per_1k_usd": 0.0005,
            "output_cost_per_1k_usd": 0.0015,
            "p95_latency_ms": 2000,
        },
    ],
    "routes": [
        {
            "name": "fast",
            "match": {"models": ["fast", "tiny-*"]},
            "targets": [{"provider": "alpha", "model": "alpha-small"}],
        },
        {
            "name": "general",
            "match": {"models": ["demo-model", "gpt-3.5-turbo"]},
            "targets": [
                {"provider": "alpha", "model": "alpha-large"},
                {"provider": "beta", "model": "beta-mid"},
            ],
        },
    ],
}

API_KEY = "sk-test-key"
OTHER_KEY = "sk-other-key"


@pytest.fixture
def routing_table() -> RoutingTable:
    return RoutingTable(parse_routing_config(ROUTING_FIXTURE))


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    await client.flushall()
    yield client
    await client.aclose()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        api_keys={
            API_KEY: ApiKeyConfig(owner="dev", rate_per_second=100.0, burst=100),
            OTHER_KEY: ApiKeyConfig(owner="ci", rate_per_second=1.0, burst=2),
        },
        cache_enabled=True,
        rate_limit_enabled=True,
    )


@pytest.fixture
def providers() -> dict[str, FakeProvider]:
    return {
        "alpha": FakeProvider(name_="alpha", models=["alpha-small", "alpha-large"]),
        "beta": FakeProvider(name_="beta", models=["beta-mid"]),
    }


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(default=[1.0, 0.0, 0.0])


@pytest.fixture
def breakers() -> BreakerRegistry:
    return BreakerRegistry(
        failure_threshold=2,
        success_threshold=1,
        recovery_seconds=30.0,
        half_open_max_calls=1,
    )


@pytest.fixture
def semantic_cache(redis_client: Redis) -> SemanticCache:
    return SemanticCache(
        redis_client,
        ttl_seconds=60,
        threshold=0.94,
        max_candidates=16,
        index_size=100,
    )


@pytest.fixture
def container(
    settings: Settings,
    redis_client: Redis,
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
) -> Container:
    for name in providers:
        # Mirrors build_container: a breaker exists for every provider from the start,
        # so /health/ready and the circuit gauge are populated before the first failure.
        breakers.get(name)
    recorder = UsageRecorder(redis_client, retention_days=7)
    service = ChatService(
        router=routing_table,
        providers=dict(providers),
        breakers=breakers,
        retry_policy=RetryPolicy(max_attempts=1, base_delay=0.0, jitter=0.0),
        cache=semantic_cache,
        embedder=embedder,
        cache_max_temperature=0.2,
        embedding_model="fake-embed",
        recorder=recorder,
    )
    return Container(
        settings=settings,
        redis=redis_client,
        router=routing_table,
        providers=dict(providers),
        breakers=breakers,
        limiter=TokenBucketLimiter(redis_client),
        recorder=recorder,
        cache=semantic_cache,
        embedder=embedder,
        service=service,
    )


@pytest.fixture
def context() -> RequestContext:
    return RequestContext(request_id="req-1", api_key=API_KEY, owner="dev")
