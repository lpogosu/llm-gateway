"""Per-key, per-model token and cost accounting.

Counters live in Redis hashes bucketed by UTC day:

    usage:<key digest>:<provider>:<model>:<YYYY-MM-DD>   hash of counters
    usage:models:<key digest>                            set of "provider/model" seen

Daily buckets rather than one running total, because "how much did this team spend last
week" is the question people actually ask, and because a bucket can be given a TTL
while a running total can only grow. Costs are derived from the pricing table in
``routing.yaml``, so the number is an estimate the operator controls, not an invoice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

from redis.asyncio import Redis

from app.domain import TokenUsage
from app.redis_support import resolve

Clock = Callable[[], datetime]

FIELD_REQUESTS = "requests"
FIELD_CACHE_HITS = "cache_hits"
FIELD_PROMPT = "prompt_tokens"
FIELD_COMPLETION = "completion_tokens"
FIELD_COST = "cost_usd"


@dataclass(frozen=True, slots=True)
class UsageRow:
    provider: str
    model: str
    requests: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class UsageReport:
    owner: str
    start: date
    end: date
    rows: tuple[UsageRow, ...]

    @property
    def total_cost_usd(self) -> float:
        return round(sum(row.cost_usd for row in self.rows), 6)

    @property
    def total_tokens(self) -> int:
        return sum(row.total_tokens for row in self.rows)

    @property
    def total_requests(self) -> int:
        return sum(row.requests for row in self.rows)


class UsageRecorder:
    def __init__(
        self,
        redis: Redis,
        *,
        retention_days: int = 90,
        namespace: str = "usage",
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self._redis = redis
        self._retention = retention_days
        self._ns = namespace
        self._clock = clock

    @staticmethod
    def digest(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]

    def _bucket_key(self, key_digest: str, provider: str, model: str, day: date) -> str:
        return f"{self._ns}:{key_digest}:{provider}:{model}:{day.isoformat()}"

    def _models_key(self, key_digest: str) -> str:
        return f"{self._ns}:models:{key_digest}"

    async def record(
        self,
        api_key: str,
        *,
        provider: str,
        model: str,
        usage: TokenUsage,
        cost_usd: float,
        cache_hit: bool = False,
    ) -> None:
        key_digest = self.digest(api_key)
        day = self._clock().date()
        bucket = self._bucket_key(key_digest, provider, model, day)
        ttl = self._retention * 86400

        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.hincrby(bucket, FIELD_REQUESTS, 1)
            if cache_hit:
                pipe.hincrby(bucket, FIELD_CACHE_HITS, 1)
            pipe.hincrby(bucket, FIELD_PROMPT, usage.prompt_tokens)
            pipe.hincrby(bucket, FIELD_COMPLETION, usage.completion_tokens)
            pipe.hincrbyfloat(bucket, FIELD_COST, cost_usd)
            pipe.expire(bucket, ttl)
            pipe.sadd(self._models_key(key_digest), f"{provider}/{model}")
            pipe.expire(self._models_key(key_digest), ttl)
            await pipe.execute()

    async def report(
        self,
        api_key: str,
        owner: str,
        *,
        start: date,
        end: date,
    ) -> UsageReport:
        if end < start:
            raise ValueError("end must not be earlier than start")
        key_digest = self.digest(api_key)
        raw_pairs = await resolve(self._redis.smembers(self._models_key(key_digest)))
        pairs = sorted(_decode(item) for item in cast(Iterable[Any], raw_pairs))
        days = list(_days_between(start, end))
        if not pairs or not days:
            return UsageReport(owner=owner, start=start, end=end, rows=())

        async with self._redis.pipeline(transaction=False) as pipe:
            for pair in pairs:
                provider, _, model = pair.partition("/")
                for day in days:
                    pipe.hgetall(self._bucket_key(key_digest, provider, model, day))
            buckets = await pipe.execute()

        rows: list[UsageRow] = []
        stride = len(days)
        for position, pair in enumerate(pairs):
            provider, _, model = pair.partition("/")
            totals = _accumulate(buckets[position * stride : (position + 1) * stride])
            if totals.requests == 0:
                continue
            rows.append(
                UsageRow(
                    provider=provider,
                    model=model,
                    requests=totals.requests,
                    cache_hits=totals.cache_hits,
                    prompt_tokens=totals.prompt_tokens,
                    completion_tokens=totals.completion_tokens,
                    cost_usd=round(totals.cost_usd, 6),
                )
            )
        return UsageReport(owner=owner, start=start, end=end, rows=tuple(rows))


@dataclass(slots=True)
class _Totals:
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def _accumulate(buckets: Iterable[Any]) -> _Totals:
    totals = _Totals()
    for bucket in buckets:
        if not bucket:
            continue
        fields = {_decode(k): _decode(v) for k, v in dict(bucket).items()}
        totals.requests += int(fields.get(FIELD_REQUESTS, 0))
        totals.cache_hits += int(fields.get(FIELD_CACHE_HITS, 0))
        totals.prompt_tokens += int(fields.get(FIELD_PROMPT, 0))
        totals.completion_tokens += int(fields.get(FIELD_COMPLETION, 0))
        totals.cost_usd += float(fields.get(FIELD_COST, 0.0))
    return totals


def _days_between(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
