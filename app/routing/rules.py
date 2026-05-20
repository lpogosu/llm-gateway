"""Rule evaluation: requested model plus caller budgets to an ordered fallback chain."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

from app.errors import ModelNotFoundError, NoRouteError
from app.routing.config import CatalogEntry, ResolvedTarget, Route, RoutingConfig


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    route_name: str
    requested_model: str
    chain: tuple[ResolvedTarget, ...]

    @property
    def primary(self) -> ResolvedTarget:
        return self.chain[0]


class RoutingTable:
    """Compiled view of ``routing.yaml``.

    Rules are evaluated top to bottom and the first route whose patterns match wins.
    Ordering is explicit in the file rather than derived from pattern specificity,
    because "the most specific rule wins" is the kind of cleverness that makes an
    operator guess why their traffic went somewhere else.
    """

    def __init__(self, config: RoutingConfig) -> None:
        self._config = config
        self._catalog: dict[str, CatalogEntry] = {entry.key: entry for entry in config.catalog}
        self._routes: tuple[Route, ...] = config.routes

    @property
    def routes(self) -> tuple[Route, ...]:
        return self._routes

    @property
    def catalog(self) -> tuple[CatalogEntry, ...]:
        return self._config.catalog

    def advertised_models(self) -> list[str]:
        """Model ids a client may ask for, as reported by ``GET /v1/models``.

        Wildcard patterns are skipped: ``*`` is a catch-all rule, not a model a client
        can name.
        """
        names: list[str] = []
        for route in self._routes:
            for pattern in route.match.models:
                if _is_literal(pattern) and pattern not in names:
                    names.append(pattern)
        return names

    def route_of(self, model: str) -> str | None:
        route = self._match_route(model)
        return route.name if route is not None else None

    def pricing_for(self, provider: str, model: str) -> CatalogEntry | None:
        return self._catalog.get(f"{provider}:{model}")

    def select(
        self,
        requested_model: str,
        *,
        max_cost_per_1k_usd: float | None = None,
        latency_budget_ms: int | None = None,
    ) -> RoutingDecision:
        route = self._match_route(requested_model)
        if route is None:
            raise ModelNotFoundError(
                f"model {requested_model!r} is not served by this gateway; "
                "see GET /v1/models"
            )

        resolved = [self._resolve(target.provider, target.model) for target in route.targets]
        eligible = [
            target
            for target in resolved
            if _within_cost(target, max_cost_per_1k_usd)
            and _within_latency(target, latency_budget_ms)
        ]
        if not eligible:
            raise NoRouteError(
                f"route {route.name!r} has no target within the requested budget "
                f"(max_cost_per_1k_usd={max_cost_per_1k_usd}, "
                f"latency_budget_ms={latency_budget_ms})"
            )
        return RoutingDecision(
            route_name=route.name,
            requested_model=requested_model,
            chain=tuple(eligible),
        )

    def _match_route(self, requested_model: str) -> Route | None:
        for route in self._routes:
            if any(fnmatchcase(requested_model, pattern) for pattern in route.match.models):
                return route
        return None

    def _resolve(self, provider: str, model: str) -> ResolvedTarget:
        # The config validator guarantees the join succeeds; a KeyError here would mean
        # the table was built from something other than a validated RoutingConfig.
        entry = self._catalog[f"{provider}:{model}"]
        return ResolvedTarget(
            provider=entry.provider,
            model=entry.model,
            input_cost_per_1k_usd=entry.input_cost_per_1k_usd,
            output_cost_per_1k_usd=entry.output_cost_per_1k_usd,
            p95_latency_ms=entry.p95_latency_ms,
        )


def _is_literal(pattern: str) -> bool:
    return not any(char in pattern for char in "*?[")


def _within_cost(target: ResolvedTarget, ceiling: float | None) -> bool:
    if ceiling is None:
        return True
    # Worst case, not blended: a caller who says "at most $0.001 per 1k" must not be
    # surprised by an output-heavy answer priced above that.
    worst_case = max(target.input_cost_per_1k_usd, target.output_cost_per_1k_usd)
    return worst_case <= ceiling


def _within_latency(target: ResolvedTarget, budget_ms: int | None) -> bool:
    if budget_ms is None:
        return True
    return target.p95_latency_ms <= budget_ms
