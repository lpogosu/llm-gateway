"""Semantic cache: similarity maths, threshold behaviour, TTL, eviction, cold start."""

from __future__ import annotations

import math

import pytest
from redis.asyncio import Redis

from app.cache.semantic import SemanticCache, cache_scope
from app.cache.similarity import DegenerateVectorError, best_match, to_unit_vector
from app.domain import CompletionRequest, Message
from tests.fakes import FakeClock, completion

SCOPE = "scope-1"


def unit(*values: float) -> list[float]:
    return list(values)


def rotated(angle_degrees: float) -> list[float]:
    """A 2-D vector at a known angle, so the expected cosine is exact."""
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians), 0.0]


# --- similarity primitives ---------------------------------------------------------


def test_unit_vector_has_length_one() -> None:
    vector = to_unit_vector([3.0, 4.0])
    assert float((vector**2).sum()) == pytest.approx(1.0)


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(DegenerateVectorError):
        to_unit_vector([0.0, 0.0, 0.0])


def test_empty_vector_is_rejected() -> None:
    with pytest.raises(DegenerateVectorError):
        to_unit_vector([])


def test_best_match_scores_a_known_angle() -> None:
    query = to_unit_vector(rotated(0))
    candidates = [to_unit_vector(rotated(60)), to_unit_vector(rotated(20))]
    index, score = best_match(query, candidates)
    assert index == 1
    assert score == pytest.approx(math.cos(math.radians(20)), abs=1e-6)


def test_best_match_on_an_empty_candidate_set_is_not_an_error() -> None:
    assert best_match(to_unit_vector([1.0, 0.0]), []) == (-1, -1.0)


# --- scope -------------------------------------------------------------------------


def request(model: str = "demo-model", **kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=(Message(role="user", content="hi"),),
        **kwargs,
    )


def test_scope_separates_models() -> None:
    assert cache_scope(request("a"), "embed") != cache_scope(request("b"), "embed")


def test_scope_separates_sampling_parameters() -> None:
    assert cache_scope(request(max_tokens=16), "embed") != cache_scope(
        request(max_tokens=2000), "embed"
    )


def test_scope_separates_embedding_models() -> None:
    assert cache_scope(request(), "embed-a") != cache_scope(request(), "embed-b")


def test_scope_is_stable_for_identical_requests() -> None:
    assert cache_scope(request(), "embed") == cache_scope(request(), "embed")


# --- store and lookup --------------------------------------------------------------


async def test_cold_index_misses_without_raising(semantic_cache: SemanticCache) -> None:
    outcome = await semantic_cache.lookup(SCOPE, unit(1.0, 0.0, 0.0))
    assert outcome.hit is None
    assert outcome.candidates == 0
    assert outcome.best_similarity is None


async def test_an_identical_prompt_hits(semantic_cache: SemanticCache) -> None:
    await semantic_cache.store(SCOPE, unit(1.0, 0.0, 0.0), completion("cached answer"))
    outcome = await semantic_cache.lookup(SCOPE, unit(1.0, 0.0, 0.0))

    assert outcome.hit is not None
    assert outcome.hit.result.content == "cached answer"
    assert outcome.hit.similarity == pytest.approx(1.0, abs=1e-6)


async def test_a_rescaled_vector_still_hits(semantic_cache: SemanticCache) -> None:
    await semantic_cache.store(SCOPE, unit(1.0, 0.0, 0.0), completion("cached answer"))
    # Cosine ignores magnitude; an embedder that returns unnormalised vectors must not
    # produce a miss.
    outcome = await semantic_cache.lookup(SCOPE, unit(7.5, 0.0, 0.0))
    assert outcome.hit is not None


async def test_a_vector_just_above_the_threshold_hits(semantic_cache: SemanticCache) -> None:
    await semantic_cache.store(SCOPE, rotated(0), completion())
    # cos(19 deg) = 0.9455, above the 0.94 threshold.
    outcome = await semantic_cache.lookup(SCOPE, rotated(19))
    assert outcome.hit is not None
    assert outcome.best_similarity == pytest.approx(math.cos(math.radians(19)), abs=1e-6)


async def test_a_vector_just_below_the_threshold_misses(semantic_cache: SemanticCache) -> None:
    await semantic_cache.store(SCOPE, rotated(0), completion())
    # cos(21 deg) = 0.9336, below 0.94. The near-miss score is still reported so the
    # threshold can be tuned from data rather than from taste.
    outcome = await semantic_cache.lookup(SCOPE, rotated(21))
    assert outcome.hit is None
    assert outcome.best_similarity == pytest.approx(math.cos(math.radians(21)), abs=1e-6)
    assert outcome.candidates == 1


async def test_the_closest_of_several_candidates_wins(semantic_cache: SemanticCache) -> None:
    await semantic_cache.store(SCOPE, rotated(0), completion("far"))
    await semantic_cache.store(SCOPE, rotated(15), completion("near"))

    outcome = await semantic_cache.lookup(SCOPE, rotated(14))
    assert outcome.hit is not None
    assert outcome.hit.result.content == "near"


async def test_scopes_do_not_leak_into_each_other(semantic_cache: SemanticCache) -> None:
    await semantic_cache.store("scope-a", unit(1.0, 0.0, 0.0), completion("a"))
    outcome = await semantic_cache.lookup("scope-b", unit(1.0, 0.0, 0.0))
    assert outcome.hit is None


async def test_usage_and_provider_survive_the_round_trip(semantic_cache: SemanticCache) -> None:
    stored = completion("answer", provider="beta", model="beta-mid", prompt_tokens=41)
    await semantic_cache.store(SCOPE, unit(1.0, 0.0, 0.0), stored)

    outcome = await semantic_cache.lookup(SCOPE, unit(1.0, 0.0, 0.0))
    assert outcome.hit is not None
    assert outcome.hit.result.provider == "beta"
    assert outcome.hit.result.usage.prompt_tokens == 41


async def test_an_empty_answer_is_not_cached(semantic_cache: SemanticCache) -> None:
    assert await semantic_cache.store(SCOPE, unit(1.0, 0.0, 0.0), completion("   ")) is None
    assert (await semantic_cache.lookup(SCOPE, unit(1.0, 0.0, 0.0))).hit is None


async def test_a_degenerate_embedding_is_not_cached(semantic_cache: SemanticCache) -> None:
    assert await semantic_cache.store(SCOPE, [0.0, 0.0, 0.0], completion()) is None


async def test_entries_expire_and_the_index_is_pruned(redis_client: Redis) -> None:
    clock = FakeClock()
    cache = SemanticCache(
        redis_client,
        ttl_seconds=60,
        threshold=0.9,
        max_candidates=16,
        index_size=100,
        clock=clock,
    )
    await cache.store(SCOPE, unit(1.0, 0.0, 0.0), completion())
    clock.advance(61)

    outcome = await cache.lookup(SCOPE, unit(1.0, 0.0, 0.0))
    assert outcome.hit is None
    assert await redis_client.zcard(f"sc:idx:{SCOPE}") == 0


async def test_a_vanished_entry_is_dropped_from_the_index(
    semantic_cache: SemanticCache, redis_client: Redis
) -> None:
    await semantic_cache.store(SCOPE, unit(1.0, 0.0, 0.0), completion())
    entry_keys = await redis_client.keys("sc:e:*")
    await redis_client.delete(*entry_keys)

    outcome = await semantic_cache.lookup(SCOPE, unit(1.0, 0.0, 0.0))
    assert outcome.hit is None
    # The dangling id is cleaned up rather than rescanned on every later lookup.
    assert await redis_client.zcard(f"sc:idx:{SCOPE}") == 0


async def test_the_index_is_bounded(redis_client: Redis) -> None:
    cache = SemanticCache(
        redis_client,
        ttl_seconds=60,
        threshold=0.9,
        max_candidates=16,
        index_size=3,
        clock=FakeClock(),
    )
    for index in range(6):
        await cache.store(SCOPE, rotated(index * 10), completion(f"answer-{index}"))

    assert await redis_client.zcard(f"sc:idx:{SCOPE}") == 3


async def test_a_vector_of_the_wrong_width_is_skipped(
    semantic_cache: SemanticCache, redis_client: Redis
) -> None:
    await semantic_cache.store(SCOPE, unit(1.0, 0.0, 0.0), completion())
    # Simulates the embedding model being swapped for one with a different width.
    outcome = await semantic_cache.lookup(SCOPE, unit(1.0, 0.0, 0.0, 0.0))

    assert outcome.hit is None
    assert await redis_client.zcard(f"sc:idx:{SCOPE}") == 0
