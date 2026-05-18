"""Ollama adapter.

Ollama speaks its own JSON API on ``/api/chat``: NDJSON for streaming, token counters
named ``prompt_eval_count`` / ``eval_count``, and sampling knobs nested under
``options``. All of that is normalised here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.domain import CompletionRequest, CompletionResult, StreamEvent, TokenUsage
from app.providers.base import ProviderProtocolError
from app.providers.http import build_client, error_from_response, translate_transport_error

PROVIDER_NAME = "ollama"


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float,
        request_timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or build_client(
            base_url,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    async def list_models(self) -> Sequence[str]:
        try:
            response = await self._client.get("/api/tags")
        except httpx.HTTPError as exc:
            raise translate_transport_error(PROVIDER_NAME, exc) from exc
        if response.status_code >= 400:
            raise error_from_response(PROVIDER_NAME, response, response.text)
        payload = _as_mapping(response.json())
        models = payload.get("models")
        if not isinstance(models, list):
            raise ProviderProtocolError(PROVIDER_NAME, "/api/tags did not return a model list")
        return [str(item["name"]) for item in models if isinstance(item, dict) and "name" in item]

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        body = _build_body(request, stream=False)
        try:
            response = await self._client.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise translate_transport_error(PROVIDER_NAME, exc) from exc
        if response.status_code >= 400:
            raise error_from_response(PROVIDER_NAME, response, response.text)

        payload = _as_mapping(response.json())
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ProviderProtocolError(PROVIDER_NAME, "response has no message object")
        return CompletionResult(
            content=str(message.get("content", "")),
            finish_reason=_finish_reason(payload),
            usage=_usage(payload),
            provider=PROVIDER_NAME,
            upstream_model=str(payload.get("model", request.model)),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        body = _build_body(request, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code >= 400:
                    raise error_from_response(
                        PROVIDER_NAME, response, (await response.aread()).decode("utf-8", "replace")
                    )
                async for line in response.aiter_lines():
                    event = _parse_stream_line(line)
                    if event is not None:
                        yield event
        except httpx.HTTPError as exc:
            raise translate_transport_error(PROVIDER_NAME, exc) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _build_body(request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.stop:
        options["stop"] = list(request.stop)

    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": stream,
    }
    if options:
        body["options"] = options
    return body


def _parse_stream_line(line: str) -> StreamEvent | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        detail = f"malformed NDJSON chunk: {stripped[:120]}"
        raise ProviderProtocolError(PROVIDER_NAME, detail) from exc
    if not isinstance(payload, dict):
        raise ProviderProtocolError(PROVIDER_NAME, "NDJSON chunk is not an object")
    if "error" in payload:
        raise ProviderProtocolError(PROVIDER_NAME, str(payload["error"]))

    message = payload.get("message")
    delta = ""
    if isinstance(message, dict):
        delta = str(message.get("content", ""))

    if payload.get("done") is True:
        return StreamEvent(
            delta=delta,
            finish_reason=_finish_reason(payload),
            usage=_usage(payload),
        )
    return StreamEvent(delta=delta)


def _finish_reason(payload: dict[str, Any]) -> str:
    reason = payload.get("done_reason")
    if reason == "length":
        return "length"
    return "stop"


def _usage(payload: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=_as_int(payload.get("prompt_eval_count")),
        completion_tokens=_as_int(payload.get("eval_count")),
    )


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _as_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderProtocolError(PROVIDER_NAME, "expected a JSON object")
    return value
