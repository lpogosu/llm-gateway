"""``GET /v1/usage`` — token and cost accounting for the calling key."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep, ContextDep
from app.errors import InvalidRequestError
from app.schemas import UsageResponse, UsageRowSchema

router = APIRouter(prefix="/v1", tags=["usage"])

DEFAULT_WINDOW_DAYS = 7
MAX_WINDOW_DAYS = 92


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    container: ContainerDep,
    ctx: ContextDep,
    start: Annotated[str | None, Query(description="Inclusive UTC date, YYYY-MM-DD")] = None,
    end: Annotated[str | None, Query(description="Inclusive UTC date, YYYY-MM-DD")] = None,
) -> UsageResponse:
    """Report usage for the authenticated key only.

    A key can never read another key's spend: the report is keyed by the digest of the
    token that authenticated the call, so there is no parameter to tamper with.
    """
    today = datetime.now(UTC).date()
    end_date = _parse_date(end, "end") or today
    start_date = _parse_date(start, "start") or end_date - timedelta(days=DEFAULT_WINDOW_DAYS - 1)
    if start_date > end_date:
        raise InvalidRequestError("start must not be later than end")
    if (end_date - start_date).days + 1 > MAX_WINDOW_DAYS:
        # Each day in the window costs one Redis read per model, so the window is
        # bounded rather than left to the caller.
        raise InvalidRequestError(f"the window must not exceed {MAX_WINDOW_DAYS} days")

    report = await container.recorder.report(
        ctx.api_key, ctx.owner, start=start_date, end=end_date
    )
    return UsageResponse(
        owner=report.owner,
        start=report.start.isoformat(),
        end=report.end.isoformat(),
        total_requests=report.total_requests,
        total_tokens=report.total_tokens,
        total_cost_usd=report.total_cost_usd,
        data=[
            UsageRowSchema(
                provider=row.provider,
                model=row.model,
                requests=row.requests,
                cache_hits=row.cache_hits,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                total_tokens=row.total_tokens,
                cost_usd=row.cost_usd,
            )
            for row in report.rows
        ],
    )


def _parse_date(raw: str | None, field: str) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise InvalidRequestError(f"{field} must be an ISO date (YYYY-MM-DD), got {raw!r}") from exc
