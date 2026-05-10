"""Provider-neutral request and response shapes.

The OpenAI wire format lives in :mod:`app.schemas`; provider payloads live in the
adapters. Everything between the two speaks these types, which is what keeps
provider quirks from leaking into routing, caching and accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A chat completion as the gateway understands it.

    ``model`` is the *requested* model on the way in and the *upstream* model once
    the router has picked a target.
    """

    model: str
    messages: tuple[Message, ...]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False
    user: str | None = None

    def prompt_text(self) -> str:
        """Flatten the conversation into the string the cache embeds.

        Roles are kept because "you are a terse assistant" followed by a question is
        a different prompt from the question alone.
        """
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class CompletionResult:
    content: str
    finish_reason: str
    usage: TokenUsage
    provider: str
    upstream_model: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One upstream delta.

    Providers report usage at different moments: Ollama attaches counters to the
    final chunk, OpenRouter only sends them when ``stream_options`` asks for them.
    A ``StreamEvent`` therefore carries usage optionally and the caller accumulates.
    """

    delta: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class ModelCard:
    id: str
    provider: str
    upstream_model: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything about the caller that the pipeline needs to make decisions."""

    request_id: str
    api_key: str
    owner: str
    bypass_cache: bool = False
    max_cost_per_1k_usd: float | None = None
    latency_budget_ms: int | None = None
    attempted_targets: list[str] = field(default_factory=list)
