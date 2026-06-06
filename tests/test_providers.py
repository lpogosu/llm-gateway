"""Provider adapters against recorded upstream payloads.

``respx`` intercepts at the httpx transport layer, so the adapters run their real
request building, status handling and stream parsing without a socket.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.cache.embeddings import EmbeddingError, OllamaEmbedder
from app.domain import CompletionRequest, Message
from app.providers.base import (
    ProviderBadRequestError,
    ProviderProtocolError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.http import parse_retry_after
from app.providers.ollama import OllamaProvider
from app.providers.openrouter import OpenRouterProvider

OLLAMA_URL = "http://ollama.test"
OPENROUTER_URL = "http://openrouter.test/api/v1"


def chat_request(stream: bool = False) -> CompletionRequest:
    return CompletionRequest(
        model="upstream-model",
        messages=(Message(role="user", content="ping"),),
        max_tokens=64,
        temperature=0.1,
        stop=("STOP",),
        stream=stream,
    )


def ollama() -> OllamaProvider:
    return OllamaProvider(OLLAMA_URL, connect_timeout=1.0, request_timeout=2.0)


def openrouter() -> OpenRouterProvider:
    return OpenRouterProvider(
        OPENROUTER_URL, "test-key", connect_timeout=1.0, request_timeout=2.0, referer="http://x"
    )


# --- Ollama ------------------------------------------------------------------------


@respx.mock
async def test_ollama_maps_options_and_reads_token_counters() -> None:
    route = respx.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "upstream-model",
                "message": {"role": "assistant", "content": "pong"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 3,
            },
        )
    )
    result = await ollama().complete(chat_request())

    body = route.calls[0].request.read()
    assert b'"num_predict":64' in body  # max_tokens has a different name upstream
    assert b'"stream":false' in body
    assert result.content == "pong"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 3
    assert result.finish_reason == "stop"


@respx.mock
async def test_ollama_reports_a_length_truncation() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"content": "half an ans"},
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 1,
                "eval_count": 64,
            },
        )
    )
    assert (await ollama().complete(chat_request())).finish_reason == "length"


@respx.mock
async def test_ollama_streams_ndjson_and_carries_usage_on_the_final_chunk() -> None:
    lines = (
        b'{"message":{"content":"Hel"},"done":false}\n'
        b'{"message":{"content":"lo"},"done":false}\n'
        b'{"message":{"content":""},"done":true,"done_reason":"stop",'
        b'"prompt_eval_count":5,"eval_count":2}\n'
    )
    respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(200, content=lines))

    events = [event async for event in ollama().stream(chat_request(stream=True))]

    assert "".join(event.delta for event in events) == "Hello"
    assert events[-1].finish_reason == "stop"
    assert events[-1].usage is not None
    assert events[-1].usage.completion_tokens == 2


@respx.mock
async def test_ollama_ignores_blank_ndjson_lines() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            content=b'{"message":{"content":"a"},"done":false}\n\n{"done":true}\n',
        )
    )
    events = [event async for event in ollama().stream(chat_request(stream=True))]
    assert len(events) == 2


@respx.mock
async def test_ollama_rejects_malformed_ndjson() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(200, content=b"{not json}\n")
    )
    with pytest.raises(ProviderProtocolError):
        [event async for event in ollama().stream(chat_request(stream=True))]


@respx.mock
async def test_ollama_lists_models() -> None:
    respx.get(f"{OLLAMA_URL}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
    )
    assert list(await ollama().list_models()) == ["a", "b"]


# --- OpenRouter --------------------------------------------------------------------


@respx.mock
async def test_openrouter_returns_the_first_choice_with_usage() -> None:
    respx.post(f"{OPENROUTER_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "upstream-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "pong"},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            },
        )
    )
    result = await openrouter().complete(chat_request())
    assert result.content == "pong"
    assert result.usage.total_tokens == 10


@respx.mock
async def test_openrouter_asks_for_usage_on_streams() -> None:
    route = respx.post(f"{OPENROUTER_URL}/chat/completions").mock(
        return_value=httpx.Response(200, content=b"data: [DONE]\n\n")
    )
    [event async for event in openrouter().stream(chat_request(stream=True))]

    # Without stream_options the stream ends with no counters and accounting silently
    # under-reports every streamed request.
    assert b'"include_usage":true' in route.calls[0].request.read()


@respx.mock
async def test_openrouter_stream_skips_keepalive_comments() -> None:
    body = (
        b": OPENROUTER PROCESSING\n\n"
        b'data: {"choices":[{"index":0,"delta":{"content":"He"}}]}\n\n'
        b": OPENROUTER PROCESSING\n\n"
        b'data: {"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post(f"{OPENROUTER_URL}/chat/completions").mock(
        return_value=httpx.Response(200, content=body)
    )

    events = [event async for event in openrouter().stream(chat_request(stream=True))]

    assert "".join(event.delta for event in events) == "Hello"
    assert events[-2].finish_reason == "stop"
    assert events[-1].usage is not None
    assert events[-1].usage.prompt_tokens == 4


@respx.mock
async def test_openrouter_surfaces_an_in_band_error_frame() -> None:
    respx.post(f"{OPENROUTER_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200, content=b'data: {"error":{"message":"model is offline"}}\n\n'
        )
    )
    with pytest.raises(ProviderProtocolError, match="model is offline"):
        [event async for event in openrouter().stream(chat_request(stream=True))]


@respx.mock
async def test_openrouter_lists_models() -> None:
    respx.get(f"{OPENROUTER_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "x/y"}]})
    )
    assert list(await openrouter().list_models()) == ["x/y"]


# --- status handling, shared -------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, ProviderBadRequestError),
        (401, ProviderBadRequestError),
        (404, ProviderBadRequestError),
        (408, ProviderUnavailableError),
        (429, ProviderRateLimitedError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
    ],
)
@respx.mock
async def test_status_codes_map_to_the_right_error(status: int, expected: type[Exception]) -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(status, text="nope"))
    with pytest.raises(expected):
        await ollama().complete(chat_request())


@respx.mock
async def test_a_429_carries_the_retry_after_hint() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(429, text="slow down", headers={"Retry-After": "7"})
    )
    with pytest.raises(ProviderRateLimitedError) as excinfo:
        await ollama().complete(chat_request())
    assert excinfo.value.retry_after_seconds == 7.0


@respx.mock
async def test_a_timeout_becomes_a_retryable_error() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(side_effect=httpx.ReadTimeout("too slow"))
    with pytest.raises(ProviderTimeoutError) as excinfo:
        await ollama().complete(chat_request())
    assert excinfo.value.retryable is True


@respx.mock
async def test_a_refused_connection_becomes_a_retryable_error() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderUnavailableError):
        await ollama().complete(chat_request())


@respx.mock
async def test_a_400_does_not_count_against_the_circuit() -> None:
    respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(ProviderBadRequestError) as excinfo:
        await ollama().complete(chat_request())
    # A wall of client mistakes must not take a healthy provider out of rotation.
    assert excinfo.value.counts_against_circuit is False


def test_retry_after_parsing() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("0.5") == 0.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("-1") is None
    # The HTTP-date form is ignored rather than trusted against an unknown clock.
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


# --- embedder ----------------------------------------------------------------------


def embedder() -> OllamaEmbedder:
    return OllamaEmbedder(OLLAMA_URL, "embed-model", timeout=1.0)


@respx.mock
async def test_embedder_returns_the_first_vector() -> None:
    route = respx.post(f"{OLLAMA_URL}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
    )
    vector = await embedder().embed("hello")

    assert vector == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert b'"model":"embed-model"' in route.calls[0].request.read()


@respx.mock
async def test_embedder_reports_an_http_failure_as_an_embedding_error() -> None:
    respx.post(f"{OLLAMA_URL}/api/embed").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(EmbeddingError, match="500"):
        await embedder().embed("hello")


@respx.mock
async def test_embedder_reports_a_timeout_as_an_embedding_error() -> None:
    # The caller must be able to treat every embedding failure the same way: skip the
    # cache and carry on.
    respx.post(f"{OLLAMA_URL}/api/embed").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(EmbeddingError):
        await embedder().embed("hello")


@respx.mock
async def test_embedder_rejects_a_response_without_vectors() -> None:
    respx.post(f"{OLLAMA_URL}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": []})
    )
    with pytest.raises(EmbeddingError, match="no vectors"):
        await embedder().embed("hello")


@respx.mock
async def test_embedder_rejects_non_numeric_values() -> None:
    respx.post(f"{OLLAMA_URL}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [["a", "b"]]})
    )
    with pytest.raises(EmbeddingError, match="non-numeric"):
        await embedder().embed("hello")
