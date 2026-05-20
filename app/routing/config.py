"""Declarative routing table loaded from ``config/routing.yaml``.

The file carries three things that belong together and change together: which
upstream models exist, what they cost, and how requested model names map onto them.
Splitting them into separate files would only guarantee that they drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CatalogEntry(BaseModel):
    """One concrete (provider, upstream model) pair with its price and latency profile."""

    model_config = {"frozen": True}

    provider: str
    model: str
    input_cost_per_1k_usd: float = Field(ge=0.0)
    output_cost_per_1k_usd: float = Field(ge=0.0)
    p95_latency_ms: int = Field(gt=0)
    context_window: int | None = Field(default=None, gt=0)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def worst_case_cost_per_1k_usd(self) -> float:
        return max(self.input_cost_per_1k_usd, self.output_cost_per_1k_usd)


class TargetRef(BaseModel):
    model_config = {"frozen": True}

    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


class RouteMatch(BaseModel):
    model_config = {"frozen": True}

    models: tuple[str, ...] = ("*",)

    @field_validator("models")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("match.models must list at least one pattern")
        return value


class Route(BaseModel):
    """A named rule: which requested models it claims and where they go.

    ``targets`` is an ordered fallback chain, not a load-balancing pool. Order is the
    operator's stated preference and the gateway walks it strictly.
    """

    model_config = {"frozen": True}

    name: str
    match: RouteMatch = RouteMatch()
    targets: tuple[TargetRef, ...]

    @field_validator("targets")
    @classmethod
    def _at_least_one_target(cls, value: tuple[TargetRef, ...]) -> tuple[TargetRef, ...]:
        if not value:
            raise ValueError("a route needs at least one target")
        return value


class RoutingConfig(BaseModel):
    model_config = {"frozen": True}

    version: int
    catalog: tuple[CatalogEntry, ...]
    routes: tuple[Route, ...]

    @model_validator(mode="after")
    def _cross_check(self) -> RoutingConfig:
        if self.version != 1:
            raise ValueError(f"unsupported routing config version: {self.version}")
        if not self.catalog:
            raise ValueError("catalog must not be empty")
        if not self.routes:
            raise ValueError("routes must not be empty")

        seen_catalog: set[str] = set()
        for entry in self.catalog:
            if entry.key in seen_catalog:
                raise ValueError(f"duplicate catalog entry: {entry.key}")
            seen_catalog.add(entry.key)

        seen_routes: set[str] = set()
        for route in self.routes:
            if route.name in seen_routes:
                raise ValueError(f"duplicate route name: {route.name}")
            seen_routes.add(route.name)
            for target in route.targets:
                if target.key not in seen_catalog:
                    raise ValueError(
                        f"route {route.name!r} points at {target.key!r}, "
                        "which is not in the catalog"
                    )
        return self


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """A route target joined with its catalog entry."""

    provider: str
    model: str
    input_cost_per_1k_usd: float
    output_cost_per_1k_usd: float
    p95_latency_ms: int

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_cost_per_1k_usd
            + completion_tokens * self.output_cost_per_1k_usd
        ) / 1000.0


def parse_routing_config(raw: dict[str, Any]) -> RoutingConfig:
    return RoutingConfig.model_validate(raw)


def load_routing_config(path: Path) -> RoutingConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return parse_routing_config(raw)
