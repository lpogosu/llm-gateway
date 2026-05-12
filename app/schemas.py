"""OpenAI-compatible wire format.

Field names, object discriminators and the shape of the streaming chunk are copied
from the OpenAI Chat Completions API on purpose: the gateway is only useful if an
existing client can be pointed at it by changing ``base_url`` and nothing else.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain import CompletionRequest, Message, TokenUsage

# Fields the gateway cannot honour but that change what the caller gets back. Ignoring
# them silently would return one completion to a client that asked for five, or plain
# text to a client that asked for a tool call.
UNSUPPORTED_FIELDS = ("tools", "functions", "tool_choice", "response_format", "logprobs")


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    # Unknown fields are accepted rather than rejected so that a newer client SDK does
    # not break on a field the gateway simply has no opinion about.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int | None = Field(default=None, ge=1)
    user: str | None = None

    @model_validator(mode="after")
    def _reject_unsupported(self) -> ChatCompletionRequest:
        extras = self.model_extra or {}
        present = [name for name in UNSUPPORTED_FIELDS if extras.get(name) is not None]
        if present:
            raise ValueError(f"unsupported field(s) for this gateway: {', '.join(present)}")
        if self.n is not None and self.n > 1:
            raise ValueError("only n=1 is supported; the gateway returns a single choice")
        return self

    def to_domain(self) -> CompletionRequest:
        return CompletionRequest(
            model=self.model,
            messages=tuple(Message(role=m.role, content=m.content) for m in self.messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=_normalise_stop(self.stop),
            stream=self.stream,
            user=self.user,
        )


class UsageSchema(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_domain(cls, usage: TokenUsage) -> UsageSchema:
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageSchema


class ChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: UsageSchema | None = None


class ModelCardSchema(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCardSchema]


class UsageRowSchema(BaseModel):
    provider: str
    model: str
    requests: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class UsageResponse(BaseModel):
    object: Literal["usage"] = "usage"
    owner: str
    start: str
    end: str
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    data: list[UsageRowSchema]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, str]


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def now_epoch() -> int:
    return int(time.time())


def _normalise_stop(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


def error_payload(message: str, error_type: str, code: str | None) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code, "param": None}}
