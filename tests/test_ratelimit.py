"""Token bucket behaviour, exercised through the real Lua script on an in-process Redis."""

from __future__ import annotations

import pytest
from redis.asyncio import Redis

from app.ratelimit.bucket import RateLimitDecision, TokenBucketLimiter
from tests.fakes import FakeClock

KEY = "sk-bucket-test"


def limiter(redis: Redis, clock: FakeClock) -> TokenBucketLimiter:
    return TokenBucketLimiter(redis, clock=clock)


async def test_burst_is_allowed_up_to_capacity_then_rejected(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)

    allowed = [
        (await bucket.consume(KEY, capacity=5, refill_per_second=1.0)).allowed for _ in range(5)
    ]
    assert allowed == [True] * 5

    rejected = await bucket.consume(KEY, capacity=5, refill_per_second=1.0)
    assert rejected.allowed is False
    assert rejected.tokens_left == pytest.approx(0.0)


async def test_tokens_refill_over_time(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    for _ in range(5):
        await bucket.consume(KEY, capacity=5, refill_per_second=2.0)

    clock.advance(1.0)  # two tokens back
    first = await bucket.consume(KEY, capacity=5, refill_per_second=2.0)
    second = await bucket.consume(KEY, capacity=5, refill_per_second=2.0)
    third = await bucket.consume(KEY, capacity=5, refill_per_second=2.0)

    assert [first.allowed, second.allowed, third.allowed] == [True, True, False]


async def test_refill_never_exceeds_capacity(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    await bucket.consume(KEY, capacity=3, refill_per_second=1.0)

    clock.advance(3600.0)
    decision = await bucket.consume(KEY, capacity=3, refill_per_second=1.0)
    # An idle hour must not buy an hour's worth of burst.
    assert decision.tokens_left == pytest.approx(2.0)


async def test_retry_after_reflects_the_wait_for_one_token(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    for _ in range(2):
        await bucket.consume(KEY, capacity=2, refill_per_second=0.5)

    decision = await bucket.consume(KEY, capacity=2, refill_per_second=0.5)
    assert decision.retry_after_seconds == pytest.approx(2.0)
    assert decision.retry_after_header == 2


async def test_retry_after_header_is_never_zero() -> None:
    decision = RateLimitDecision(allowed=False, tokens_left=0.99, retry_after_seconds=0.01)
    # A Retry-After of 0 invites an immediate retry and a hot loop.
    assert decision.retry_after_header == 1


async def test_buckets_are_isolated_per_key(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    for _ in range(2):
        await bucket.consume("key-a", capacity=2, refill_per_second=1.0)

    assert (await bucket.consume("key-a", capacity=2, refill_per_second=1.0)).allowed is False
    assert (await bucket.consume("key-b", capacity=2, refill_per_second=1.0)).allowed is True


async def test_the_raw_key_never_appears_in_redis(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    await bucket.consume(KEY, capacity=2, refill_per_second=1.0)

    keys = [key.decode() for key in await redis_client.keys("*")]
    assert keys
    assert all(KEY not in key for key in keys)


async def test_bucket_expires_so_idle_keys_do_not_leak(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    await bucket.consume(KEY, capacity=10, refill_per_second=1.0)

    ttl = await redis_client.ttl(bucket.key_for(KEY))
    # Two full refills of head-room: long enough that state is never lost while the
    # caller is active, short enough that a one-off key is collected.
    assert ttl == 20


async def test_a_cost_larger_than_the_bucket_is_rejected_up_front(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        await bucket.consume(KEY, capacity=2, refill_per_second=1.0, cost=3)


async def test_fractional_tokens_survive_a_round_trip(redis_client: Redis) -> None:
    clock = FakeClock()
    bucket = limiter(redis_client, clock)
    await bucket.consume(KEY, capacity=5, refill_per_second=1.0)
    clock.advance(0.25)
    decision = await bucket.consume(KEY, capacity=5, refill_per_second=1.0)
    # 5 - 1 + 0.25 - 1: the fractional part must not be truncated by the Lua boundary.
    assert decision.tokens_left == pytest.approx(3.25)
