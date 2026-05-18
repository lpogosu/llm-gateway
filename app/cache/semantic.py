"""Semantic response cache on Redis.

Layout, per cache scope (a scope is one requested model plus one set of sampling
parameters):

    sc:idx:<scope>          sorted set, member = entry id, score = expiry epoch
    sc:e:<scope>:<entry id>  hash {vec: float32 bytes, payload: json, dim: int}

Lookup pulls the newest ``max_candidates`` ids from the index, loads their vectors and
scores them in one matrix-vector product. There is no vector index: a linear scan over
a bounded, per-scope candidate set is exact, needs no Redis module, and stays under a
millisecond for the couple of thousand entries a single scope is allowed to hold. The
moment a deployment needs more than that, this class is the seam where RediSearch or
pgvector goes in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from redis.asyncio import Redis

from app.cache.similarity import (
    DegenerateVectorError,
    Vector,
    best_match,
    pack,
    to_unit_vector,
    unpack,
)
from app.domain import CompletionRequest, CompletionResult, TokenUsage

logger = logging.getLogger(__name__)

Clock = Callable[[], float]


def cache_scope(request: CompletionRequest, embedding_model: str) -> str:
    """Partition key for cache entries.

    Two prompts may only share an entry when they were asked of the same model with the
    same sampling parameters and embedded by the same model. Mixing sampling parameters
    would let ``max_tokens=16`` return an answer generated for ``max_tokens=2000``;
    mixing embedding models would compare vectors from two different spaces.
    """
    material = "|".join(
        [
            request.model,
            embedding_model,
            f"t={request.temperature}",
            f"p={request.top_p}",
            f"m={request.max_tokens}",
            f"s={','.join(request.stop)}",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class CacheHit:
    result: CompletionResult
    similarity: float
    entry_id: str


@dataclass(frozen=True, slots=True)
class LookupOutcome:
    """Result of a lookup plus the score that produced it.

    ``best_similarity`` is reported even when it fell short of the threshold: watching
    the distribution of near-misses is how an operator decides whether the threshold is
    set too high or dangerously low.
    """

    hit: CacheHit | None
    best_similarity: float | None
    candidates: int


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        *,
        ttl_seconds: int,
        threshold: float,
        max_candidates: int,
        index_size: int,
        namespace: str = "sc",
        clock: Clock = time.time,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._threshold = threshold
        self._max_candidates = max_candidates
        self._index_size = index_size
        self._ns = namespace
        self._clock = clock

    @property
    def threshold(self) -> float:
        return self._threshold

    def _index_key(self, scope: str) -> str:
        return f"{self._ns}:idx:{scope}"

    def _entry_key(self, scope: str, entry_id: str) -> str:
        return f"{self._ns}:e:{scope}:{entry_id}"

    async def lookup(self, scope: str, vector: Sequence[float]) -> LookupOutcome:
        """Find a stored answer whose prompt embedding is close enough to ``vector``.

        Misses on a cold index, on a best score below the threshold, and on candidates
        that vanished between the index read and the hash read. A cache is never
        allowed to turn into an error path.
        """
        query = to_unit_vector(vector)
        now = self._clock()
        index_key = self._index_key(scope)

        # Drop members whose entries have already expired; without this the index grows
        # into a graveyard of ids whose hashes Redis has long since collected.
        await self._redis.zremrangebyscore(index_key, "-inf", now)
        raw_ids = await self._redis.zrevrange(index_key, 0, self._max_candidates - 1)
        entry_ids = [_decode(item) for item in cast(list[Any], raw_ids)]
        if not entry_ids:
            return LookupOutcome(hit=None, best_similarity=None, candidates=0)

        async with self._redis.pipeline(transaction=False) as pipe:
            for entry_id in entry_ids:
                pipe.hmget(self._entry_key(scope, entry_id), ["vec", "payload"])
            rows = await pipe.execute()

        candidates: list[Vector] = []
        payloads: list[str] = []
        kept_ids: list[str] = []
        missing: list[str] = []
        for entry_id, row in zip(entry_ids, rows, strict=True):
            vec_blob, payload_blob = row[0], row[1]
            if vec_blob is None or payload_blob is None:
                missing.append(entry_id)
                continue
            try:
                stored = unpack(_as_bytes(vec_blob), dimensions=int(query.size))
            except DegenerateVectorError:
                # A different embedding model was configured after these entries were
                # written; they are unusable, not corrupt.
                missing.append(entry_id)
                continue
            candidates.append(stored)
            payloads.append(_decode(payload_blob))
            kept_ids.append(entry_id)

        if missing:
            await self._redis.zrem(index_key, *missing)
        if not candidates:
            return LookupOutcome(hit=None, best_similarity=None, candidates=0)

        index, score = best_match(query, candidates)
        if score < self._threshold:
            logger.debug(
                "semantic cache miss below threshold",
                extra={"similarity": score, "threshold": self._threshold},
            )
            return LookupOutcome(hit=None, best_similarity=score, candidates=len(candidates))
        hit = CacheHit(
            result=_decode_result(payloads[index]),
            similarity=score,
            entry_id=kept_ids[index],
        )
        return LookupOutcome(hit=hit, best_similarity=score, candidates=len(candidates))

    async def store(
        self,
        scope: str,
        vector: Sequence[float],
        result: CompletionResult,
    ) -> str | None:
        """Persist a completion. Returns the entry id, or ``None`` if it was not cached."""
        if not result.content.strip():
            # An empty answer is never worth serving again.
            return None
        try:
            unit = to_unit_vector(vector)
        except DegenerateVectorError:
            logger.warning("refusing to cache a degenerate embedding", extra={"scope": scope})
            return None

        entry_id = uuid.uuid4().hex
        expires_at = self._clock() + self._ttl
        entry_key = self._entry_key(scope, entry_id)
        index_key = self._index_key(scope)

        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.hset(
                entry_key,
                mapping={
                    "vec": pack(unit),
                    "payload": _encode_result(result),
                    "dim": str(unit.size),
                },
            )
            pipe.expire(entry_key, self._ttl)
            pipe.zadd(index_key, {entry_id: expires_at})
            # Keep the index bounded so a scan stays predictable; evicted hashes fall
            # away on their own TTL.
            pipe.zremrangebyrank(index_key, 0, -(self._index_size + 1))
            pipe.expire(index_key, self._ttl)
            await pipe.execute()
        return entry_id


def _as_bytes(value: Any) -> bytes:
    if not isinstance(value, bytes):
        raise DegenerateVectorError("stored vector is not binary")
    return value


def _encode_result(result: CompletionResult) -> str:
    return json.dumps(
        {
            "content": result.content,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "provider": result.provider,
            "upstream_model": result.upstream_model,
        },
        ensure_ascii=False,
    )


def _decode_result(payload: str) -> CompletionResult:
    data = json.loads(payload)
    return CompletionResult(
        content=str(data["content"]),
        finish_reason=str(data["finish_reason"]),
        usage=TokenUsage(
            prompt_tokens=int(data["prompt_tokens"]),
            completion_tokens=int(data["completion_tokens"]),
        ),
        provider=str(data["provider"]),
        upstream_model=str(data["upstream_model"]),
    )


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
