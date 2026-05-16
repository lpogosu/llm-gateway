"""``GET /v1/models`` — the model ids this gateway accepts."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ContainerDep, ContextDep
from app.schemas import ModelCardSchema, ModelListResponse, now_epoch

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models(container: ContainerDep, ctx: ContextDep) -> ModelListResponse:
    """List what a client may put in ``model``.

    These are gateway-level names from the routing rules, not the upstream ids: the
    whole point of the routing table is that ``gpt-3.5-turbo`` can be served by a local
    model today and a hosted one tomorrow without the client noticing.
    """
    created = now_epoch()
    cards = [
        ModelCardSchema(
            id=name,
            created=created,
            owned_by=container.router.route_of(name) or "llm-gateway",
        )
        for name in container.router.advertised_models()
    ]
    return ModelListResponse(data=cards)
