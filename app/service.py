"""The request pipeline: authorise, limit, cache, route, call, account.

Every stage here is deliberately ordered. Rate limiting runs before the cache so that
a caller cannot burn through a shared cache for free; the cache runs before routing so
that a hit costs no provider capacity at all; accounting runs on every outcome,
including cache hits and client disconnects, because usage a dashboard cannot see is
usage nobody controls.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from redis.exceptions import RedisError

from app.cache.embeddings import Embedder, EmbeddingError
from app.cache.semantic import SemanticCache, cache_scope
from app.domain import (
    CompletionRequest,
    CompletionResult,
    RequestContext,
    StreamEvent,
    TokenUsage,
)
from app.errors import AllTargetsFailedError, UpstreamError
from app.metrics import (
    cache_lookups_total,
    cache_similarity,
    cost_usd_total,
    dependency_errors_total,
    provider_errors_total,
    provider_retries_total,
    requests_total,
    tokens_total,
    upstream_inflight,
)
from app.providers.base import (
    Provider,
    ProviderBadRequestError,
    ProviderError,
    ProviderProtocolError,
)
from app.resilience.circuit import BreakerRegistry
from app.resilience.retry import RetryPolicy, run_with_retry
from app.routing.config import ResolvedTarget
from app.routing.rules import RoutingDecision, RoutingTable
from app.tracing import get_tracer

logger = logging.getLogger(__name__)

CACHE_HIT = "hit"
CACHE_MISS = "miss"
CACHE_BYPASS = "bypass"
CACHE_DISABLED = "disabled"
CACHE_SKIPPED = "skipped"
CACHE_ERROR = "error"

_CACHED_PROVIDER_LABEL = "cache"


class UsageSink(Protocol):
    """The slice of :class:`~app.accounting.usage.UsageRecorder` the service depends on."""

    async def record(
        self,
        api_key: str,
        *,
        provider: str,
        model: str,
        usage: TokenUsage,
        cost_usd: float,
        cache_hit: bool = False,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CacheOutcome:
    result: CompletionResult
    similarity: float


@dataclass(frozen=True, slots=True)
class ChatOutcome:
    result: CompletionResult
    route_name: str
    cache_status: str
    cache_similarity: float | None
    attempted: tuple[str, ...]


@dataclass(slots=True)
class StreamSession:
    """A stream that has already produced its first event.

    Opening eagerly is what makes fallback possible: once a single byte has reached the
    client there is no honest way to switch providers, so the choice has to be settled
    before the response headers go out.
    """

    route_name: str
    provider: str
    model: str
    cache_status: str
    cache_similarity: float | None
    events: AsyncGenerator[StreamEvent, None]


class ChatService:
    def __init__(
        self,
        *,
        router: RoutingTable,
        providers: Mapping[str, Provider],
        breakers: BreakerRegistry,
        retry_policy: RetryPolicy,
        cache: SemanticCache | None,
        embedder: Embedder | None,
        cache_max_temperature: float,
        embedding_model: str,
        recorder: UsageSink,
    ) -> None:
        self._router = router
        self._providers = providers
        self._breakers = breakers
        self._retry = retry_policy
        self._cache = cache
        self._embedder = embedder
        self._cache_max_temperature = cache_max_temperature
        self._embedding_model = embedding_model
        self._recorder = recorder

    # ------------------------------------------------------------------ non-streaming

    async def complete(self, request: CompletionRequest, ctx: RequestContext) -> ChatOutcome:
        decision = self._router.select(
            request.model,
            max_cost_per_1k_usd=ctx.max_cost_per_1k_usd,
            latency_budget_ms=ctx.latency_budget_ms,
        )
        cached, vector, cache_status = await self._try_cache(request, ctx)
        if cached is not None:
            await self._account(ctx, decision, cached.result, cost_usd=0.0, cache_hit=True)
            requests_total.labels(
                decision.route_name, _CACHED_PROVIDER_LABEL, request.model, "cache_hit"
            ).inc()
            return ChatOutcome(
                result=cached.result,
                route_name=decision.route_name,
                cache_status=CACHE_HIT,
                cache_similarity=cached.similarity,
                attempted=(),
            )

        attempted: list[str] = []
        last_error: ProviderError | None = None
        for target in decision.chain:
            if not self._admit(target, attempted):
                continue
            try:
                result = await self._call_provider(request, target)
            except ProviderBadRequestError as exc:
                # The payload is wrong, not the provider. Every target would reject it,
                # so the caller gets the answer immediately instead of after N attempts.
                self._note_failure(target, exc)
                raise UpstreamError(exc.message) from exc
            except ProviderError as exc:
                self._note_failure(target, exc)
                last_error = exc
                continue

            cost = target.cost_usd(result.usage.prompt_tokens, result.usage.completion_tokens)
            await self._account(ctx, decision, result, cost_usd=cost, cache_hit=False)
            self._observe_tokens(target, result.usage, ctx.owner, cost)
            requests_total.labels(
                decision.route_name, target.provider, target.model, "success"
            ).inc()
            if vector is not None:
                await self._store_cache(request, vector, result)
            return ChatOutcome(
                result=result,
                route_name=decision.route_name,
                cache_status=cache_status,
                cache_similarity=None,
                attempted=tuple(attempted),
            )

        requests_total.labels(decision.route_name, "none", request.model, "exhausted").inc()
        detail = f" last error: {last_error.message}" if last_error is not None else ""
        raise AllTargetsFailedError(
            f"no target in route {decision.route_name!r} could serve the request.{detail}",
            attempted=attempted,
        )

    # ---------------------------------------------------------------------- streaming

    async def open_stream(self, request: CompletionRequest, ctx: RequestContext) -> StreamSession:
        decision = self._router.select(
            request.model,
            max_cost_per_1k_usd=ctx.max_cost_per_1k_usd,
            latency_budget_ms=ctx.latency_budget_ms,
        )
        cached, vector, cache_status = await self._try_cache(request, ctx)
        if cached is not None:
            await self._account(ctx, decision, cached.result, cost_usd=0.0, cache_hit=True)
            requests_total.labels(
                decision.route_name, _CACHED_PROVIDER_LABEL, request.model, "cache_hit"
            ).inc()
            return StreamSession(
                route_name=decision.route_name,
                provider=_CACHED_PROVIDER_LABEL,
                model=cached.result.upstream_model,
                cache_status=CACHE_HIT,
                cache_similarity=cached.similarity,
                events=_replay(cached.result),
            )

        attempted: list[str] = []
        last_error: ProviderError | None = None
        for target in decision.chain:
            if not self._admit(target, attempted):
                continue
            try:
                events, first = await self._open_upstream_stream(request, target)
            except ProviderBadRequestError as exc:
                self._note_failure(target, exc)
                raise UpstreamError(exc.message) from exc
            except ProviderError as exc:
                self._note_failure(target, exc)
                last_error = exc
                continue

            return StreamSession(
                route_name=decision.route_name,
                provider=target.provider,
                model=target.model,
                cache_status=cache_status,
                cache_similarity=None,
                events=self._drain(events, first, request, ctx, decision, target, vector),
            )

        requests_total.labels(decision.route_name, "none", request.model, "exhausted").inc()
        detail = f" last error: {last_error.message}" if last_error is not None else ""
        raise AllTargetsFailedError(
            f"no target in route {decision.route_name!r} could start a stream.{detail}",
            attempted=attempted,
        )

    async def _open_upstream_stream(
        self, request: CompletionRequest, target: ResolvedTarget
    ) -> tuple[AsyncIterator[StreamEvent], StreamEvent]:
        """Start the upstream stream and pull its first event.

        Retries live here and nowhere else in the streaming path: an attempt that has
        not yet produced a delta can be thrown away, one that has cannot.
        """
        provider = self._providers[target.provider]
        upstream_request = replace(request, model=target.model, stream=True)

        async def attempt(_: int) -> tuple[AsyncIterator[StreamEvent], StreamEvent]:
            iterator = provider.stream(upstream_request)
            try:
                first = await anext(iterator)
            except StopAsyncIteration as exc:
                await _aclose(iterator)
                raise ProviderProtocolError(
                    target.provider, "upstream closed the stream without sending anything"
                ) from exc
            except BaseException:
                await _aclose(iterator)
                raise
            return iterator, first

        return await run_with_retry(
            attempt,
            self._retry,
            on_retry=lambda *_: provider_retries_total.labels(target.provider).inc(),
        )

    async def _drain(
        self,
        events: AsyncIterator[StreamEvent],
        first: StreamEvent,
        request: CompletionRequest,
        ctx: RequestContext,
        decision: RoutingDecision,
        target: ResolvedTarget,
        vector: list[float] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Forward upstream events, then account for whatever actually got through.

        The ``finally`` block runs on a clean end, on an upstream error and on a client
        disconnect alike — FastAPI closes the generator, which raises ``GeneratorExit``
        here — so a caller that hangs up mid-generation is still billed for the tokens
        the provider produced.
        """
        chunks: list[str] = [first.delta]
        usage = first.usage or TokenUsage()
        completed = False
        upstream_inflight.labels(target.provider).inc()
        try:
            yield first
            async for event in events:
                chunks.append(event.delta)
                if event.usage is not None:
                    usage = event.usage
                yield event
            completed = True
            self._breakers.get(target.provider).record_success()
        except ProviderError as exc:
            self._note_failure(target, exc)
            raise
        finally:
            upstream_inflight.labels(target.provider).dec()
            # A stream torn down mid-flight often fails to close cleanly; that must not
            # replace the original reason the stream ended.
            with contextlib.suppress(Exception):
                await _aclose(events)

            text = "".join(chunks)
            if usage.total_tokens == 0 and text:
                logger.warning(
                    "provider reported no token usage for a streamed response",
                    extra={"provider": target.provider, "model": target.model},
                )
            result = CompletionResult(
                content=text,
                finish_reason="stop" if completed else "cancelled",
                usage=usage,
                provider=target.provider,
                upstream_model=target.model,
            )
            cost = target.cost_usd(usage.prompt_tokens, usage.completion_tokens)
            self._observe_tokens(target, usage, ctx.owner, cost)
            requests_total.labels(
                decision.route_name,
                target.provider,
                target.model,
                "success" if completed else "aborted",
            ).inc()
            # On a client disconnect this coroutine already runs inside a cancelled
            # scope, so the Redis writes are shielded: losing the accounting for a
            # cancelled stream is exactly how usage goes unnoticed.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(
                    self._finalise_stream(
                        ctx, decision, result, request, vector, cost_usd=cost, store=completed
                    )
                )

    async def _finalise_stream(
        self,
        ctx: RequestContext,
        decision: RoutingDecision,
        result: CompletionResult,
        request: CompletionRequest,
        vector: list[float] | None,
        *,
        cost_usd: float,
        store: bool,
    ) -> None:
        await self._account(ctx, decision, result, cost_usd=cost_usd, cache_hit=False)
        if store and vector is not None:
            await self._store_cache(request, vector, result)

    # ------------------------------------------------------------------------ helpers

    def _admit(self, target: ResolvedTarget, attempted: list[str]) -> bool:
        if target.provider not in self._providers:
            logger.error("route points at an unconfigured provider", extra={"target": target.key})
            return False
        attempted.append(target.key)
        breaker = self._breakers.get(target.provider)
        if not breaker.allow():
            provider_errors_total.labels(target.provider, "circuit_open").inc()
            logger.warning("skipping target, circuit is open", extra={"target": target.key})
            return False
        return True

    async def _call_provider(
        self, request: CompletionRequest, target: ResolvedTarget
    ) -> CompletionResult:
        provider = self._providers[target.provider]
        upstream_request = replace(request, model=target.model, stream=False)
        tracer = get_tracer()

        async def attempt(index: int) -> CompletionResult:
            with tracer.start_as_current_span("provider.complete") as span:
                span.set_attribute("llm.provider", target.provider)
                span.set_attribute("llm.model", target.model)
                span.set_attribute("llm.attempt", index)
                upstream_inflight.labels(target.provider).inc()
                try:
                    return await provider.complete(upstream_request)
                finally:
                    upstream_inflight.labels(target.provider).dec()

        result = await run_with_retry(
            attempt,
            self._retry,
            on_retry=lambda *_: provider_retries_total.labels(target.provider).inc(),
        )
        self._breakers.get(target.provider).record_success()
        return result

    def _note_failure(self, target: ResolvedTarget, exc: ProviderError) -> None:
        provider_errors_total.labels(target.provider, exc.kind).inc()
        if exc.counts_against_circuit:
            self._breakers.get(target.provider).record_failure()
        logger.warning(
            "provider call failed",
            extra={"target": target.key, "kind": exc.kind, "detail": exc.message},
        )

    async def _try_cache(
        self, request: CompletionRequest, ctx: RequestContext
    ) -> tuple[CacheOutcome | None, list[float] | None, str]:
        """Look the prompt up. Returns (hit, embedding to reuse on write, status).

        The embedding is handed back so that a miss does not have to pay for a second
        embedding call when the answer is later stored.
        """
        if self._cache is None or self._embedder is None:
            cache_lookups_total.labels(CACHE_DISABLED).inc()
            return None, None, CACHE_DISABLED
        if ctx.bypass_cache:
            cache_lookups_total.labels(CACHE_BYPASS).inc()
            return None, None, CACHE_BYPASS
        if request.temperature is not None and request.temperature > self._cache_max_temperature:
            # An explicit high temperature is a request for variety; returning a stored
            # answer would be the opposite of what was asked for. A request that omits
            # temperature entirely is treated as cacheable, because that is the shape of
            # almost all real traffic and a cache that never fires is not a cache.
            cache_lookups_total.labels(CACHE_SKIPPED).inc()
            return None, None, CACHE_SKIPPED

        scope = cache_scope(request, self._embedding_model)
        try:
            vector = await self._embedder.embed(request.prompt_text())
        except EmbeddingError as exc:
            # The embedder being down must degrade to "no caching", never to a 5xx.
            cache_lookups_total.labels(CACHE_ERROR).inc()
            logger.warning("embedding failed, serving without cache", extra={"detail": str(exc)})
            return None, None, CACHE_ERROR

        try:
            outcome = await self._cache.lookup(scope, vector)
        except RedisError as exc:
            # A cache that cannot be reached is a cache miss, never a failed request.
            dependency_errors_total.labels("cache").inc()
            cache_lookups_total.labels(CACHE_ERROR).inc()
            logger.warning("cache lookup failed", extra={"detail": str(exc)})
            return None, None, CACHE_ERROR
        if outcome.best_similarity is not None:
            cache_similarity.observe(outcome.best_similarity)
        if outcome.hit is None:
            cache_lookups_total.labels(CACHE_MISS).inc()
            return None, vector, CACHE_MISS
        cache_lookups_total.labels(CACHE_HIT).inc()
        return CacheOutcome(outcome.hit.result, outcome.hit.similarity), vector, CACHE_HIT

    async def _store_cache(
        self, request: CompletionRequest, vector: list[float], result: CompletionResult
    ) -> None:
        if self._cache is None:
            return
        scope = cache_scope(request, self._embedding_model)
        try:
            await self._cache.store(scope, vector, result)
        except RedisError as exc:
            dependency_errors_total.labels("cache").inc()
            logger.warning("cache write failed", extra={"detail": str(exc)})

    async def _account(
        self,
        ctx: RequestContext,
        decision: RoutingDecision,
        result: CompletionResult,
        *,
        cost_usd: float,
        cache_hit: bool,
    ) -> None:
        # A cache hit is booked against the provider that originally produced the answer
        # with zero cost, so that "requests" and "cache_hits" for a model stay
        # comparable in the same row.
        try:
            await self._recorder.record(
                ctx.api_key,
                provider=result.provider,
                model=result.upstream_model,
                usage=result.usage,
                cost_usd=cost_usd,
                cache_hit=cache_hit,
            )
        except RedisError as exc:
            # The completion has already been generated and paid for. Losing a counter is
            # bad; throwing away the answer the caller is waiting for is worse. The
            # Prometheus counters below are unaffected, so the loss stays visible.
            dependency_errors_total.labels("accounting").inc()
            logger.error("usage accounting failed", extra={"detail": str(exc)})
        logger.info(
            "chat completion served",
            extra={
                "route": decision.route_name,
                "provider": result.provider,
                "model": result.upstream_model,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cost_usd": round(cost_usd, 6),
                "cache_hit": cache_hit,
            },
        )

    def _observe_tokens(
        self, target: ResolvedTarget, usage: TokenUsage, owner: str, cost_usd: float
    ) -> None:
        tokens_total.labels(target.provider, target.model, "prompt").inc(usage.prompt_tokens)
        tokens_total.labels(target.provider, target.model, "completion").inc(
            usage.completion_tokens
        )
        if cost_usd:
            cost_usd_total.labels(owner, target.provider, target.model).inc(cost_usd)


async def _replay(result: CompletionResult) -> AsyncGenerator[StreamEvent, None]:
    """Turn a cached completion back into a stream.

    Chunk boundaries from the original generation are not stored, so a replay sends the
    answer as one delta followed by the terminating event. Clients that render deltas
    see the text appear at once; clients that concatenate see no difference at all.
    """
    yield StreamEvent(delta=result.content)
    yield StreamEvent(finish_reason=result.finish_reason, usage=result.usage)


async def _aclose(iterator: AsyncIterator[StreamEvent]) -> None:
    closer = getattr(iterator, "aclose", None)
    if closer is not None:
        await closer()
