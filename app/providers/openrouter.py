"""OpenRouter adapter.

OpenRouter is OpenAI-shaped, which makes the mapping short, with two wrinkles worth
naming: it emits ``: OPENROUTER PROCESSING`` SSE comments as a keep-alive, and it only
returns token counters on a stream when ``stream_options.include_usage`` is set.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.domain import CompletionRequest, CompletionResult, StreamEvent, TokenUsage
from app.providers.base import ProviderProtocolError
from app.providers.http import build_client, error_from_response, translate_transport_error

PROVIDER_NAME = "openrouter"
_SSE_DONE = "[DONE]"


class OpenRouterProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        connect_timeout: float,
        request_timeout: float,
        referer: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if referer:
            # OpenRouter uses these for its public app directory; they are not credentials.
            headers["HTTP-Referer"] = referer
            headers["X-Title"] = "llm-gateway"
        self._client = client or build_client(
            base_url,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            headers=headers,
        )

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    async def list_models(self) -> Sequence[str]:
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError as exc:
            raise translate_transport_error(PROVIDER_NAME, exc) from exc
        if response.status_code >= 400:
            raise error_from_response(PROVIDER_NAME, response, response.text)
        payload = _as_mapping(response.json())
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProviderProtocolError(PROVIDER_NAME, "/models did not return a data array")
        return [str(item["id"]) for item in data if isinstance(item, dict) and "id" in item]

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        body = _build_body(request, stream=False)
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise translate_transport_error(PROVIDER_NAME, exc) from exc
        if response.status_code >= 400:
            raise error_from_response(PROVIDER_NAME, response, response.text)

        payload = _as_mapping(response.json())
        choice = _first_choice(payload)
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderProtocolError(PROVIDER_NAME, "choice has no message object")
        return CompletionResult(
            content=str(message.get("content") or ""),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=_usage(payload.get("usage")),
            provider=PROVIDER_NAME,
            upstream_model=str(payload.get("model", request.model)),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        body = _build_body(request, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    raise error_from_response(
                        PROVIDER_NAME, response, (await response.aread()).decode("utf-8", "replace")
                    )
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is not None:
                        yield event
        except httpx.HTTPError as exc:
            raise translate_transport_error(PROVIDER_NAME, exc) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _build_body(request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": stream,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.stop:
        body["stop"] = list(request.stop)
    if stream:
        # Without this the stream ends without token counters and accounting silently
        # under-reports every streamed request.
        body["stream_options"] = {"include_usage": True}
    return body


def _parse_sse_line(line: str) -> StreamEvent | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        # Blank separators and keep-alive comments carry no data.
        return None
    if not stripped.startswith("data:"):
        return None
    data = stripped[len("data:") :].strip()
    if data == _SSE_DONE:
        return None

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        detail = f"malformed SSE payload: {data[:120]}"
        raise ProviderProtocolError(PROVIDER_NAME, detail) from exc
    if not isinstance(payload, dict):
        raise ProviderProtocolError(PROVIDER_NAME, "SSE payload is not an object")
    if isinstance(payload.get("error"), dict):
        raise ProviderProtocolError(PROVIDER_NAME, str(payload["error"].get("message", "error")))

    usage = _usage(payload.get("usage")) if payload.get("usage") is not None else None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        # The final usage-only chunk has an empty choices array.
        return StreamEvent(usage=usage) if usage is not None else None

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderProtocolError(PROVIDER_NAME, "SSE choice is not an object")
    delta_obj = choice.get("delta")
    delta = ""
    if isinstance(delta_obj, dict):
        delta = str(delta_obj.get("content") or "")
    finish_reason = choice.get("finish_reason")
    return StreamEvent(
        delta=delta,
        finish_reason=str(finish_reason) if finish_reason else None,
        usage=usage,
    )


def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderProtocolError(PROVIDER_NAME, "response has no choices")
    return choices[0]


def _usage(raw: Any) -> TokenUsage:
    if not isinstance(raw, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=_as_int(raw.get("prompt_tokens")),
        completion_tokens=_as_int(raw.get("completion_tokens")),
    )


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _as_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderProtocolError(PROVIDER_NAME, "expected a JSON object")
    return value
