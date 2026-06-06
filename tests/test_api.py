"""HTTP surface: OpenAI compatibility, auth, limits, SSE framing, usage reporting."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.accounting.usage import UsageRecorder
from app.cache.semantic import SemanticCache
from app.container import Container
from app.domain import StreamEvent, TokenUsage
from app.main import create_app
from app.providers.base import ProviderUnavailableError
from app.ratelimit.bucket import TokenBucketLimiter
from tests.conftest import API_KEY, OTHER_KEY
from tests.fakes import FakeProvider, completion

AUTH = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def api(container: Container) -> FastAPI:
    app = create_app(settings=container.settings, container=container)
    # ASGITransport does not run the lifespan, so the container is installed directly.
    app.state.container = container
    return app


@pytest.fixture
async def client(api: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        yield http


def body(model: str = "demo-model", **extra: object) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "what is the capital of France?"}],
        **extra,
    }


# --- authentication ----------------------------------------------------------------


async def test_a_missing_token_is_a_401_in_the_openai_envelope(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/v1/chat/completions", json=body())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_an_unknown_token_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json=body(), headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401


async def test_a_non_bearer_scheme_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json=body(), headers={"Authorization": "Basic abc"}
    )
    assert response.status_code == 401


# --- non-streaming completions -----------------------------------------------------


async def test_a_completion_has_the_openai_shape(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris", prompt_tokens=9, completion_tokens=1)]

    response = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "demo-model"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Paris",
        "name": None,
    }
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}
    assert payload["id"].startswith("chatcmpl-")


async def test_gateway_headers_expose_the_routing_decision(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]

    response = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert response.headers["x-gateway-route"] == "general"
    assert response.headers["x-gateway-provider"] == "alpha"
    assert response.headers["x-gateway-cache"] == "miss"
    assert response.headers["x-request-id"]


async def test_the_request_id_from_the_caller_is_echoed(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]

    response = await client.post(
        "/v1/chat/completions", json=body(), headers=AUTH | {"X-Request-ID": "trace-me"}
    )

    assert response.headers["x-request-id"] == "trace-me"


async def test_a_second_identical_request_is_served_from_the_cache(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]

    await client.post("/v1/chat/completions", json=body(), headers=AUTH)
    second = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert second.headers["x-gateway-cache"] == "hit"
    assert float(second.headers["x-gateway-cache-similarity"]) == pytest.approx(1.0, abs=1e-3)
    assert len(providers["alpha"].complete_calls) == 1


async def test_the_bypass_header_forces_a_fresh_call(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]

    await client.post("/v1/chat/completions", json=body(), headers=AUTH)
    second = await client.post(
        "/v1/chat/completions", json=body(), headers=AUTH | {"X-Gateway-Cache-Bypass": "true"}
    )

    assert second.headers["x-gateway-cache"] == "bypass"
    assert len(providers["alpha"].complete_calls) == 2


async def test_a_cost_ceiling_header_reroutes_to_a_cheaper_target(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["beta"].completions = [completion("cheap", provider="beta", model="beta-mid")]

    response = await client.post(
        "/v1/chat/completions",
        json=body(),
        headers=AUTH | {"X-Gateway-Max-Cost-Per-1k": "0.0016"},
    )

    assert response.headers["x-gateway-provider"] == "beta"
    assert providers["alpha"].complete_calls == []


async def test_a_budget_nothing_satisfies_is_a_400(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=body(),
        headers=AUTH | {"X-Gateway-Latency-Budget-Ms": "1"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_eligible_target"


async def test_a_malformed_budget_header_is_a_400(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json=body(), headers=AUTH | {"X-Gateway-Max-Cost-Per-1k": "cheap"}
    )
    assert response.status_code == 400


async def test_an_unknown_model_is_a_404(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json=body("no-such"), headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_validation_errors_use_the_openai_envelope_not_a_422(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions", json={"model": "demo-model", "messages": []}, headers=AUTH
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_a_field_the_gateway_cannot_honour_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=body(tools=[{"type": "function", "function": {"name": "f"}}]),
        headers=AUTH,
    )
    # Silently dropping `tools` would return prose to a client waiting for a tool call.
    assert response.status_code == 400
    assert "tools" in response.json()["error"]["message"]


async def test_more_than_one_choice_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json=body(n=3), headers=AUTH)
    assert response.status_code == 400


async def test_an_unknown_but_harmless_field_is_accepted(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    response = await client.post(
        "/v1/chat/completions", json=body(seed=42, service_tier="auto"), headers=AUTH
    )
    assert response.status_code == 200


async def test_every_target_failing_is_a_503_that_names_the_attempts(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [ProviderUnavailableError("alpha", "down")]
    providers["beta"].completions = [ProviderUnavailableError("beta", "down")]

    response = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert response.status_code == 503
    assert response.json()["error"]["attempted_targets"] == ["alpha:alpha-large", "beta:beta-mid"]


# --- rate limiting -----------------------------------------------------------------


async def test_a_burst_beyond_capacity_gets_429_with_retry_after(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    other = {"Authorization": f"Bearer {OTHER_KEY}"}  # 1/s with a burst of 2

    statuses = []
    for index in range(4):
        response = await client.post(
            "/v1/chat/completions",
            json=body(**{"messages": [{"role": "user", "content": f"q{index}"}]}),
            headers=other | {"X-Gateway-Cache-Bypass": "true"},
        )
        statuses.append(response.status_code)
        last = response

    assert statuses[:2] == [200, 200]
    assert statuses[2:] == [429, 429]
    assert last.json()["error"]["type"] == "rate_limit_error"
    assert int(last.headers["retry-after"]) >= 1


async def test_the_metrics_endpoint_is_not_rate_limited(client: httpx.AsyncClient) -> None:
    for _ in range(20):
        assert (await client.get("/metrics")).status_code == 200


# --- streaming ---------------------------------------------------------------------


def sse_frames(text: str) -> list[str]:
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


async def test_a_stream_is_framed_as_openai_chunks(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
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

    async with client.stream(
        "POST", "/v1/chat/completions", json=body(stream=True), headers=AUTH
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-gateway-provider"] == "alpha"
        text = "".join([chunk async for chunk in response.aiter_text()])

    frames = sse_frames(text)
    assert frames[-1] == "[DONE]"

    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Paris"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1


async def test_an_upstream_failure_mid_stream_is_delivered_as_an_error_frame(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].streams = [
        [StreamEvent(delta="Pa"), ProviderUnavailableError("alpha", "connection reset")]
    ]

    async with client.stream(
        "POST", "/v1/chat/completions", json=body(stream=True), headers=AUTH
    ) as response:
        # The status line is long gone, so the failure has to travel in the body.
        assert response.status_code == 200
        text = "".join([chunk async for chunk in response.aiter_text()])

    frames = sse_frames(text)
    assert frames[-1] == "[DONE]"
    assert json.loads(frames[-2])["error"]["code"] == "upstream_error"


async def test_a_stream_that_cannot_start_fails_before_the_headers(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].streams = [ProviderUnavailableError("alpha", "down")]
    providers["beta"].streams = [ProviderUnavailableError("beta", "down")]

    response = await client.post("/v1/chat/completions", json=body(stream=True), headers=AUTH)

    # Nothing has been written yet, so the client gets a real status code.
    assert response.status_code == 503


# --- models and usage --------------------------------------------------------------


async def test_models_lists_what_a_client_may_ask_for(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/models", headers=AUTH)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    ids = [card["id"] for card in payload["data"]]
    assert ids == ["fast", "demo-model", "gpt-3.5-turbo"]
    assert payload["data"][1]["owned_by"] == "general"


async def test_usage_reports_tokens_and_cost_for_the_calling_key(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion(prompt_tokens=1000, completion_tokens=1000)]
    await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    response = await client.get("/v1/usage", headers=AUTH)

    payload = response.json()
    assert payload["owner"] == "dev"
    assert payload["total_requests"] == 1
    assert payload["total_tokens"] == 2000
    assert payload["total_cost_usd"] == pytest.approx(0.003)
    assert payload["data"][0]["model"] == "alpha-model"


async def test_usage_is_isolated_between_keys(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion()]
    await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    response = await client.get("/v1/usage", headers={"Authorization": f"Bearer {OTHER_KEY}"})

    assert response.json()["total_requests"] == 0


async def test_usage_counts_cache_hits_separately(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion(prompt_tokens=100, completion_tokens=100)]
    await client.post("/v1/chat/completions", json=body(), headers=AUTH)
    await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    row = (await client.get("/v1/usage", headers=AUTH)).json()["data"][0]

    assert row["requests"] == 2
    assert row["cache_hits"] == 1
    # The cached answer is free, so cost did not double.
    assert row["cost_usd"] == pytest.approx(0.0003)


async def test_a_malformed_usage_window_is_a_400(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/usage?start=yesterday", headers=AUTH)
    assert response.status_code == 400


async def test_an_inverted_usage_window_is_a_400(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/usage?start=2026-05-10&end=2026-05-01", headers=AUTH)
    assert response.status_code == 400


# --- operational endpoints ---------------------------------------------------------


async def test_liveness_needs_no_credentials(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_reports_redis_and_circuits(client: httpx.AsyncClient) -> None:
    payload = (await client.get("/health/ready")).json()
    assert payload["checks"]["redis"] == "ok"
    assert payload["checks"]["circuit:alpha"] == "closed"


async def test_metrics_expose_the_gateway_collectors(
    client: httpx.AsyncClient, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    text = (await client.get("/metrics")).text

    assert "llm_gateway_requests_total" in text
    assert "llm_gateway_cache_lookups_total" in text
    assert "llm_gateway_request_duration_seconds_bucket" in text
    assert "llm_gateway_circuit_state" in text


# --- degradation when Redis is unavailable -----------------------------------------


def _fail(*_: object, **__: object) -> object:
    raise RedisConnectionError("connection refused")


class BrokenRedis:
    """Every command raises, which is what a Redis outage looks like to the gateway."""

    def register_script(self, _: str) -> object:
        # Registering is local; only running the script reaches the server.
        return _fail

    def __getattr__(self, name: str) -> object:
        return _fail


async def test_a_redis_outage_on_the_limiter_fails_closed(
    client: httpx.AsyncClient, container: Container
) -> None:
    container.limiter = TokenBucketLimiter(cast(Redis, BrokenRedis()))

    response = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    # Forwarding an unmetered request to a paid provider is worse than a 503.
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "dependency_unavailable"


async def test_a_redis_outage_in_the_cache_degrades_to_a_normal_answer(
    client: httpx.AsyncClient, container: Container, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    container.service._cache = SemanticCache(
        cast(Redis, BrokenRedis()),
        ttl_seconds=60,
        threshold=0.9,
        max_candidates=8,
        index_size=10,
    )

    response = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert response.status_code == 200
    assert response.headers["x-gateway-cache"] == "error"
    assert response.json()["choices"][0]["message"]["content"] == "Paris"


async def test_a_redis_outage_in_accounting_still_returns_the_completion(
    client: httpx.AsyncClient, container: Container, providers: dict[str, FakeProvider]
) -> None:
    providers["alpha"].completions = [completion("Paris")]
    container.service._recorder = UsageRecorder(
        cast(Redis, BrokenRedis())
    )

    response = await client.post("/v1/chat/completions", json=body(), headers=AUTH)

    # The tokens are already spent; dropping the answer would waste them twice.
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Paris"
