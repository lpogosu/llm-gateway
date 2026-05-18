"""The provider port and the error vocabulary the pipeline reasons about.

Adapters translate one upstream API into these four methods and classify upstream
failures into these exceptions. Nothing above this line knows that Ollama streams
NDJSON while OpenRouter streams SSE.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from app.domain import CompletionRequest, CompletionResult, StreamEvent


class ProviderError(Exception):
    """An upstream call failed.

    ``retryable`` drives the retry loop, ``kind`` is the Prometheus label and
    ``counts_against_circuit`` separates "the provider is unhealthy" from "the caller
    sent something the provider rejected" — a wall of 400s must not open the breaker.
    """

    kind = "error"
    retryable = False
    counts_against_circuit = True

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class ProviderTimeoutError(ProviderError):
    kind = "timeout"
    retryable = True


class ProviderUnavailableError(ProviderError):
    """Connection refused, DNS failure, 5xx."""

    kind = "unavailable"
    retryable = True


class ProviderRateLimitedError(ProviderError):
    kind = "rate_limited"
    retryable = True


class ProviderBadRequestError(ProviderError):
    """The upstream rejected the payload; retrying it verbatim cannot help."""

    kind = "bad_request"
    retryable = False
    counts_against_circuit = False


class ProviderProtocolError(ProviderError):
    """The upstream answered with something this adapter cannot parse."""

    kind = "protocol"
    retryable = False


@runtime_checkable
class Provider(Protocol):
    """Everything the gateway needs from a backend."""

    @property
    def name(self) -> str: ...

    async def list_models(self) -> Sequence[str]:
        """Upstream model ids, used by the readiness probe and by config linting."""

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run a non-streaming completion. ``request.model`` is the upstream id."""

    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Run a streaming completion.

        Returning the iterator rather than awaiting it keeps the HTTP connection open
        for exactly as long as the caller consumes deltas: closing the iterator closes
        the upstream response.
        """

    async def aclose(self) -> None:
        """Release the underlying connection pool."""
