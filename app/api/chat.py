"""``POST /v1/chat/completions`` in both streaming and non-streaming form."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import ContainerDep, ContextDep
from app.errors import GatewayError
from app.metrics import time_to_first_token_seconds
from app.middleware import scope_slot
from app.providers.base import ProviderError
from app.schemas import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    UsageSchema,
    error_payload,
    new_completion_id,
    now_epoch,
)
from app.service import ChatService, StreamSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])

SSE_MEDIA_TYPE = "text/event-stream"
_SSE_DONE = "data: [DONE]\n\n"


@router.post("/chat/completions")
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    container: ContainerDep,
    ctx: ContextDep,
) -> Response:
    service: ChatService = container.service
    domain_request = payload.to_domain()

    if domain_request.stream:
        started = time.perf_counter()
        session = await service.open_stream(domain_request, ctx)
        time_to_first_token_seconds.labels(session.provider, session.model).observe(
            time.perf_counter() - started
        )
        _publish_labels(request, session.route_name, session.provider, session.model)
        return StreamingResponse(
            _sse_body(session, payload.model),
            media_type=SSE_MEDIA_TYPE,
            headers=_gateway_headers(
                session.route_name,
                session.provider,
                session.model,
                session.cache_status,
                session.cache_similarity,
            )
            | {"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    outcome = await service.complete(domain_request, ctx)
    result = outcome.result
    _publish_labels(request, outcome.route_name, result.provider, result.upstream_model)
    body = ChatCompletionResponse(
        id=new_completion_id(),
        created=now_epoch(),
        model=payload.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=UsageSchema.from_domain(result.usage),
    )
    return JSONResponse(
        content=body.model_dump(),
        headers=_gateway_headers(
            outcome.route_name,
            result.provider,
            result.upstream_model,
            outcome.cache_status,
            outcome.cache_similarity,
        ),
    )


async def _sse_body(session: StreamSession, advertised_model: str) -> AsyncIterator[str]:
    """Render domain events as OpenAI streaming chunks.

    The status line is long gone by the time an upstream can fail here, so a mid-stream
    failure is delivered as a JSON error frame followed by ``[DONE]``. That is what the
    OpenAI API does and what the official clients understand; closing the socket without
    a terminator leaves them waiting.
    """
    completion_id = new_completion_id()
    created = now_epoch()

    def frame(choice: ChunkChoice, usage: UsageSchema | None = None) -> str:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=advertised_model,
            choices=[choice],
            usage=usage,
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # OpenAI clients expect the assistant role once, in the opening chunk.
    yield frame(ChunkChoice(index=0, delta=ChunkDelta(role="assistant", content="")))
    try:
        async for event in session.events:
            if event.delta:
                yield frame(ChunkChoice(index=0, delta=ChunkDelta(content=event.delta)))
            if event.finish_reason is not None:
                usage = UsageSchema.from_domain(event.usage) if event.usage else None
                yield frame(
                    ChunkChoice(index=0, delta=ChunkDelta(), finish_reason=event.finish_reason),
                    usage,
                )
    except ProviderError as exc:
        logger.warning("stream aborted by the upstream", extra={"detail": exc.message})
        payload = error_payload(exc.message, "api_error", "upstream_error")
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except GatewayError as exc:
        logger.warning("stream aborted by the gateway", extra={"detail": exc.message})
        yield f"data: {json.dumps(exc.to_payload(), ensure_ascii=False)}\n\n"
    yield _SSE_DONE


def _gateway_headers(
    route: str,
    provider: str,
    model: str,
    cache_status: str,
    similarity: float | None,
) -> dict[str, str]:
    headers = {
        "X-Gateway-Route": route,
        "X-Gateway-Provider": provider,
        "X-Gateway-Model": model,
        "X-Gateway-Cache": cache_status,
    }
    if similarity is not None:
        headers["X-Gateway-Cache-Similarity"] = f"{similarity:.4f}"
    return headers


def _publish_labels(request: Request, route: str, provider: str, model: str) -> None:
    slot = scope_slot(request.scope)
    slot["route"] = route
    slot["provider"] = provider
    slot["model"] = model
