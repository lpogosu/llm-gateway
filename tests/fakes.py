"""Test doubles.

Every provider in the test suite is scripted: no test opens a socket, starts Ollama or
needs an API key. The fakes are deliberately literal — they replay a queue of outcomes —
so that a test failure points at the gateway and never at clever fake behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from app.cache.embeddings import EmbeddingError
from app.domain import CompletionRequest, CompletionResult, StreamEvent, TokenUsage
from app.providers.base import ProviderError

CompletionOutcome = CompletionResult | ProviderError
# A stream script is either an immediate failure, or a sequence of events in which a
# ProviderError entry means "the upstream broke at this point".
StreamOutcome = list[StreamEvent | ProviderError] | ProviderError

T = TypeVar("T")


def completion(
    content: str = "hello",
    *,
    provider: str = "alpha",
    model: str = "alpha-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> CompletionResult:
    return CompletionResult(
        content=content,
        finish_reason=finish_reason,
        usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        provider=provider,
        upstream_model=model,
    )


@dataclass
class FakeProvider:
    """Replays queued outcomes; the last one repeats once the queue runs dry."""

    name_: str
    completions: list[CompletionOutcome] = field(default_factory=list)
    streams: list[StreamOutcome] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    complete_calls: list[CompletionRequest] = field(default_factory=list)
    stream_calls: list[CompletionRequest] = field(default_factory=list)
    closed_streams: int = 0
    _completion_index: int = 0
    _stream_index: int = 0

    @property
    def name(self) -> str:
        return self.name_

    async def list_models(self) -> Sequence[str]:
        return self.models

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.complete_calls.append(request)
        outcome = self._next(self.completions, self._completion_index)
        self._completion_index += 1
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        outcome = self._next(self.streams, self._stream_index)
        self._stream_index += 1
        if isinstance(outcome, ProviderError):
            raise outcome
        try:
            for event in outcome:
                if isinstance(event, ProviderError):
                    raise event
                yield event
        finally:
            # Lets a test prove that a client disconnect really tears the upstream down.
            self.closed_streams += 1

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _next(queue: Sequence[T], index: int) -> T:
        if not queue:
            raise AssertionError("the fake provider was called with an empty script")
        return queue[min(index, len(queue) - 1)]


@dataclass
class FakeEmbedder:
    """Returns a queued vector per prompt, or raises to exercise the degradation path."""

    vectors: dict[str, list[float]] = field(default_factory=dict)
    default: list[float] | None = None
    fail: bool = False
    calls: list[str] = field(default_factory=list)

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise EmbeddingError("embedding backend is down")
        vector = self.vectors.get(text, self.default)
        if vector is None:
            raise AssertionError(f"no vector scripted for {text!r}")
        return vector

    async def aclose(self) -> None:
        return None


@dataclass
class RecordingSink:
    """Captures accounting calls so a test can assert what was billed."""

    entries: list[dict[str, object]] = field(default_factory=list)

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
        self.entries.append(
            {
                "api_key": api_key,
                "provider": provider,
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": cost_usd,
                "cache_hit": cache_hit,
            }
        )


class FakeClock:
    """A clock the test moves by hand, so nothing sleeps to test a timeout."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
