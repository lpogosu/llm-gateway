"""The pipeline itself: fallback, breaker interaction, cache paths, streaming."""

from __future__ import annotations

import pytest

from app.cache.semantic import SemanticCache
from app.domain import CompletionRequest, Message, RequestContext, StreamEvent, TokenUsage
from app.errors import AllTargetsFailedError, UpstreamError
from app.providers.base import (
    ProviderBadRequestError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from app.resilience.circuit import BreakerRegistry, CircuitState
from app.resilience.retry import RetryPolicy
from app.routing.rules import RoutingTable
from app.service import (
    CACHE_BYPASS,
    CACHE_ERROR,
    CACHE_HIT,
    CACHE_MISS,
    CACHE_SKIPPED,
    ChatService,
)
from tests.fakes import FakeEmbedder, FakeProvider, RecordingSink, completion


def build_service(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    *,
    cache: SemanticCache | None = None,
    embedder: FakeEmbedder | None = None,
    recorder: RecordingSink | None = None,
    retry: RetryPolicy | None = None,
) -> ChatService:
    return ChatService(
        router=routing_table,
        providers=dict(providers),
        breakers=breakers,
        retry_policy=retry or RetryPolicy(max_attempts=1, base_delay=0.0, jitter=0.0),
        cache=cache,
        embedder=embedder,
        cache_max_temperature=0.2,
        embedding_model="fake-embed",
        recorder=recorder or RecordingSink(),
    )


def chat(model: str = "demo-model", **kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=(Message(role="user", content="what is the capital of France?"),),
        **kwargs,
    )


# --- fallback ----------------------------------------------------------------------


async def test_the_primary_target_is_used_when_it_works(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("from alpha")]
    service = build_service(routing_table, providers, breakers)

    outcome = await service.complete(chat(), context)

    assert outcome.result.content == "from alpha"
    assert providers["beta"].complete_calls == []
    assert outcome.attempted == ("alpha:alpha-large",)


async def test_a_failing_primary_falls_through_to_the_next_target(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [ProviderUnavailableError("alpha", "down")]
    providers["beta"].completions = [completion("from beta", provider="beta", model="beta-mid")]
    service = build_service(routing_table, providers, breakers)

    outcome = await service.complete(chat(), context)

    assert outcome.result.content == "from beta"
    assert outcome.attempted == ("alpha:alpha-large", "beta:beta-mid")


async def test_the_upstream_model_replaces_the_requested_one(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion()]
    service = build_service(routing_table, providers, breakers)

    await service.complete(chat("demo-model"), context)

    assert providers["alpha"].complete_calls[0].model == "alpha-large"


async def test_a_bad_request_is_returned_immediately_without_fallback(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [ProviderBadRequestError("alpha", "context too long")]
    providers["beta"].completions = [completion("from beta")]
    service = build_service(routing_table, providers, breakers)

    with pytest.raises(UpstreamError, match="context too long"):
        await service.complete(chat(), context)
    # The second provider would reject the same payload; trying it only wastes time.
    assert providers["beta"].complete_calls == []


async def test_an_exhausted_chain_reports_what_it_tried(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [ProviderUnavailableError("alpha", "down")]
    providers["beta"].completions = [ProviderUnavailableError("beta", "down too")]
    service = build_service(routing_table, providers, breakers)

    with pytest.raises(AllTargetsFailedError) as excinfo:
        await service.complete(chat(), context)

    assert excinfo.value.status_code == 503
    assert excinfo.value.attempted == ["alpha:alpha-large", "beta:beta-mid"]
    assert "down too" in excinfo.value.message


async def test_an_open_circuit_skips_the_target_without_calling_it(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    breakers.get("alpha").record_failure()
    breakers.get("alpha").record_failure()
    assert breakers.get("alpha").state is CircuitState.OPEN

    providers["beta"].completions = [completion("from beta")]
    service = build_service(routing_table, providers, breakers)

    outcome = await service.complete(chat(), context)

    assert outcome.result.content == "from beta"
    assert providers["alpha"].complete_calls == []


async def test_repeated_failures_open_the_circuit(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [ProviderUnavailableError("alpha", "down")]
    providers["beta"].completions = [completion()]
    service = build_service(routing_table, providers, breakers)

    for _ in range(2):
        await service.complete(chat(), context)

    assert breakers.get("alpha").state is CircuitState.OPEN


async def test_retries_are_attempted_before_falling_back(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [
        ProviderRateLimitedError("alpha", "429"),
        completion("second attempt"),
    ]
    service = build_service(
        routing_table,
        providers,
        breakers,
        retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=0.0),
    )

    outcome = await service.complete(chat(), context)

    assert outcome.result.content == "second attempt"
    assert len(providers["alpha"].complete_calls) == 2
    assert providers["beta"].complete_calls == []


# --- accounting --------------------------------------------------------------------


async def test_cost_is_billed_from_the_catalog_price(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion(prompt_tokens=1000, completion_tokens=1000)]
    sink = RecordingSink()
    service = build_service(routing_table, providers, breakers, recorder=sink)

    await service.complete(chat(), context)

    # alpha-large: 0.001 per 1k in, 0.002 per 1k out.
    assert sink.entries[0]["cost_usd"] == pytest.approx(0.003)
    assert sink.entries[0]["api_key"] == context.api_key


# --- cache -------------------------------------------------------------------------


async def test_a_miss_calls_the_provider_and_a_repeat_does_not(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    sink = RecordingSink()
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder, recorder=sink
    )

    first = await service.complete(chat(), context)
    second = await service.complete(chat(), context)

    assert first.cache_status == CACHE_MISS
    assert second.cache_status == CACHE_HIT
    assert second.result.content == "Paris"
    assert len(providers["alpha"].complete_calls) == 1
    assert sink.entries[1]["cache_hit"] is True
    assert sink.entries[1]["cost_usd"] == 0.0


async def test_a_cache_hit_still_reports_the_original_provider(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("Paris", provider="alpha", model="alpha-large")]
    sink = RecordingSink()
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder, recorder=sink
    )

    await service.complete(chat(), context)
    await service.complete(chat(), context)

    assert sink.entries[1]["provider"] == "alpha"
    assert sink.entries[1]["model"] == "alpha-large"


async def test_the_bypass_header_skips_the_lookup_entirely(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder
    )
    await service.complete(chat(), context)

    bypass = RequestContext(
        request_id="r2", api_key=context.api_key, owner="dev", bypass_cache=True
    )
    outcome = await service.complete(chat(), bypass)

    assert outcome.cache_status == CACHE_BYPASS
    assert len(providers["alpha"].complete_calls) == 2
    # A bypass must not even pay for an embedding.
    assert len(embedder.calls) == 1


async def test_an_explicit_high_temperature_is_not_cached(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder
    )

    outcome = await service.complete(chat(temperature=0.9), context)

    assert outcome.cache_status == CACHE_SKIPPED
    assert embedder.calls == []


async def test_a_failing_embedder_degrades_to_no_cache_instead_of_a_5xx(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    service = build_service(
        routing_table,
        providers,
        breakers,
        cache=semantic_cache,
        embedder=FakeEmbedder(fail=True),
    )

    outcome = await service.complete(chat(), context)

    assert outcome.cache_status == CACHE_ERROR
    assert outcome.result.content == "Paris"


async def test_different_prompts_do_not_share_an_answer(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    context: RequestContext,
) -> None:
    providers["alpha"].completions = [completion("Paris"), completion("Berlin")]
    embedder = FakeEmbedder(
        vectors={
            "user: what is the capital of France?": [1.0, 0.0, 0.0],
            "user: what is the capital of Germany?": [0.0, 1.0, 0.0],
        }
    )
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder
    )

    await service.complete(chat(), context)
    other = CompletionRequest(
        model="demo-model",
        messages=(Message(role="user", content="what is the capital of Germany?"),),
    )
    outcome = await service.complete(other, context)

    assert outcome.cache_status == CACHE_MISS
    assert outcome.result.content == "Berlin"


# --- streaming ---------------------------------------------------------------------


async def test_a_stream_forwards_every_delta_and_accounts_at_the_end(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].streams = [
        [
            StreamEvent(delta="Pa"),
            StreamEvent(delta="ris"),
            StreamEvent(
                finish_reason="stop", usage=TokenUsage(prompt_tokens=7, completion_tokens=2)
            ),
        ]
    ]
    sink = RecordingSink()
    service = build_service(routing_table, providers, breakers, recorder=sink)

    session = await service.open_stream(chat(stream=True), context)
    events = [event async for event in session.events]

    assert "".join(event.delta for event in events) == "Paris"
    assert sink.entries[0]["completion_tokens"] == 2
    assert sink.entries[0]["provider"] == "alpha"


async def test_a_stream_that_fails_before_the_first_delta_falls_back(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].streams = [ProviderUnavailableError("alpha", "down")]
    providers["beta"].streams = [[StreamEvent(delta="hi"), StreamEvent(finish_reason="stop")]]
    service = build_service(routing_table, providers, breakers)

    session = await service.open_stream(chat(stream=True), context)
    events = [event async for event in session.events]

    assert session.provider == "beta"
    assert "".join(event.delta for event in events) == "hi"


async def test_an_empty_stream_is_treated_as_a_protocol_failure(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].streams = [[]]
    providers["beta"].streams = [[StreamEvent(delta="hi"), StreamEvent(finish_reason="stop")]]
    service = build_service(routing_table, providers, breakers)

    session = await service.open_stream(chat(stream=True), context)

    assert session.provider == "beta"


async def test_a_client_disconnect_closes_the_upstream_and_still_accounts(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    context: RequestContext,
) -> None:
    providers["alpha"].streams = [
        [
            StreamEvent(delta="one"),
            StreamEvent(delta="two", usage=TokenUsage(prompt_tokens=3, completion_tokens=1)),
            StreamEvent(delta="three"),
            StreamEvent(finish_reason="stop"),
        ]
    ]
    sink = RecordingSink()
    service = build_service(routing_table, providers, breakers, recorder=sink)

    session = await service.open_stream(chat(stream=True), context)
    seen = []
    async for event in session.events:
        seen.append(event.delta)
        if len(seen) == 2:
            break
    await session.events.aclose()

    assert providers["alpha"].closed_streams == 1
    assert sink.entries[0]["completion_tokens"] == 1
    assert len(sink.entries) == 1


async def test_a_completed_stream_is_cached_and_replayed(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
    context: RequestContext,
) -> None:
    providers["alpha"].streams = [
        [
            StreamEvent(delta="Pa"),
            StreamEvent(delta="ris"),
            StreamEvent(
                finish_reason="stop", usage=TokenUsage(prompt_tokens=7, completion_tokens=2)
            ),
        ]
    ]
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder
    )

    first = await service.open_stream(chat(stream=True), context)
    [event async for event in first.events]

    second = await service.open_stream(chat(stream=True), context)
    replayed = [event async for event in second.events]

    assert second.cache_status == CACHE_HIT
    assert "".join(event.delta for event in replayed) == "Paris"
    assert replayed[-1].usage is not None
    assert replayed[-1].usage.completion_tokens == 2
    assert len(providers["alpha"].stream_calls) == 1


async def test_a_cancelled_stream_is_not_cached(
    routing_table: RoutingTable,
    providers: dict[str, FakeProvider],
    breakers: BreakerRegistry,
    semantic_cache: SemanticCache,
    embedder: FakeEmbedder,
    context: RequestContext,
) -> None:
    providers["alpha"].streams = [
        [StreamEvent(delta="Pa"), StreamEvent(delta="ris"), StreamEvent(finish_reason="stop")],
        [StreamEvent(delta="second run"), StreamEvent(finish_reason="stop")],
    ]
    service = build_service(
        routing_table, providers, breakers, cache=semantic_cache, embedder=embedder
    )

    session = await service.open_stream(chat(stream=True), context)
    async for _ in session.events:
        break
    await session.events.aclose()

    # Half an answer must never be served to the next caller as a complete one.
    again = await service.open_stream(chat(stream=True), context)
    assert again.cache_status == CACHE_MISS
