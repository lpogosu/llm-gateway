"""Token bucket enforced in Redis.

The whole refill-and-consume step runs as one Lua script so that N gateway replicas
share one bucket per API key without a read-modify-write race. Doing it in Python
would need WATCH/MULTI/EXEC plus a retry loop, which is two extra round-trips on the
hot path for the same guarantee.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from redis.asyncio import Redis

Clock = Callable[[], float]

# KEYS[1] bucket hash. ARGV: capacity, refill/s, now (epoch seconds), cost, ttl seconds.
# Returns {allowed, tokens_left, retry_after_seconds} with floats as strings, because
# Lua -> RESP integer conversion truncates and we would lose the fractional token state.
_CONSUME_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local state = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(state[1])
local updated = tonumber(state[2])

if tokens == nil or updated == nil then
  tokens = capacity
  updated = now
end

local elapsed = now - updated
if elapsed < 0 then
  elapsed = 0
end

tokens = math.min(capacity, tokens + elapsed * refill)

local allowed = 0
local retry_after = 0.0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
else
  retry_after = (cost - tokens) / refill
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'updated', tostring(now))
redis.call('EXPIRE', key, ttl)

return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    tokens_left: float
    retry_after_seconds: float

    @property
    def retry_after_header(self) -> int:
        """``Retry-After`` is an integer number of seconds and 0 would invite a hot loop."""
        return max(1, math.ceil(self.retry_after_seconds))


class TokenBucketLimiter:
    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str = "rl",
        clock: Clock = time.time,
    ) -> None:
        self._redis = redis
        self._namespace = namespace
        self._clock = clock
        self._script = redis.register_script(_CONSUME_SCRIPT)

    def key_for(self, api_key: str) -> str:
        # The raw API key never becomes a Redis key: a Redis dump would leak every
        # credential. A digest is enough to keep buckets separate.
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]
        return f"{self._namespace}:{digest}"

    async def consume(
        self,
        api_key: str,
        *,
        capacity: int,
        refill_per_second: float,
        cost: float = 1.0,
    ) -> RateLimitDecision:
        if cost > capacity:
            # A request that can never fit would otherwise report a Retry-After that
            # never comes true.
            raise ValueError(f"cost {cost} exceeds bucket capacity {capacity}")

        ttl = max(1, math.ceil(capacity / refill_per_second) * 2)
        raw = await self._script(
            keys=[self.key_for(api_key)],
            args=[
                str(capacity),
                str(refill_per_second),
                str(self._clock()),
                str(cost),
                str(ttl),
            ],
        )
        return _decode(raw)


def _decode(raw: Any) -> RateLimitDecision:
    values = cast(list[Any], raw)
    allowed = int(values[0]) == 1
    return RateLimitDecision(
        allowed=allowed,
        tokens_left=float(_to_str(values[1])),
        retry_after_seconds=float(_to_str(values[2])),
    )


def _to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
