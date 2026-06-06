"""Routing rules: matching, ordering, budgets and config validation."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.errors import ModelNotFoundError, NoRouteError
from app.routing.config import parse_routing_config
from app.routing.rules import RoutingTable
from tests.conftest import ROUTING_FIXTURE


def test_exact_model_name_selects_its_route(routing_table: RoutingTable) -> None:
    decision = routing_table.select("demo-model")
    assert decision.route_name == "general"
    assert [target.key for target in decision.chain] == ["alpha:alpha-large", "beta:beta-mid"]


def test_glob_pattern_matches(routing_table: RoutingTable) -> None:
    assert routing_table.select("tiny-7b").route_name == "fast"


def test_first_matching_route_wins_even_if_a_later_one_also_matches() -> None:
    config = copy.deepcopy(ROUTING_FIXTURE)
    config["routes"] = [
        {
            "name": "catch-all",
            "match": {"models": ["*"]},
            "targets": [{"provider": "beta", "model": "beta-mid"}],
        },
        {
            "name": "specific",
            "match": {"models": ["demo-model"]},
            "targets": [{"provider": "alpha", "model": "alpha-large"}],
        },
    ]
    table = RoutingTable(parse_routing_config(config))
    # File order decides, not pattern specificity: an operator reading top to bottom
    # must be able to predict where traffic goes.
    assert table.select("demo-model").route_name == "catch-all"


def test_unknown_model_is_a_404_not_a_silent_default(routing_table: RoutingTable) -> None:
    with pytest.raises(ModelNotFoundError) as excinfo:
        routing_table.select("no-such-model")
    assert excinfo.value.status_code == 404


def test_cost_ceiling_drops_expensive_targets(routing_table: RoutingTable) -> None:
    # alpha-large costs 0.002/1k on output, beta-mid 0.0015.
    decision = routing_table.select("demo-model", max_cost_per_1k_usd=0.0016)
    assert [target.key for target in decision.chain] == ["beta:beta-mid"]


def test_cost_ceiling_uses_the_worst_case_price(routing_table: RoutingTable) -> None:
    # beta-mid is 0.0005 in / 0.0015 out. A ceiling between the two must exclude it,
    # otherwise an output-heavy answer would silently break the caller's budget.
    with pytest.raises(NoRouteError):
        routing_table.select("demo-model", max_cost_per_1k_usd=0.001)


def test_latency_budget_drops_slow_targets(routing_table: RoutingTable) -> None:
    decision = routing_table.select("demo-model", latency_budget_ms=2500)
    assert [target.key for target in decision.chain] == ["beta:beta-mid"]


def test_budget_that_excludes_everything_is_a_400(routing_table: RoutingTable) -> None:
    with pytest.raises(NoRouteError) as excinfo:
        routing_table.select("demo-model", latency_budget_ms=10)
    assert excinfo.value.status_code == 400


def test_budgets_preserve_the_declared_fallback_order(routing_table: RoutingTable) -> None:
    decision = routing_table.select("demo-model", max_cost_per_1k_usd=0.01)
    assert [target.key for target in decision.chain] == ["alpha:alpha-large", "beta:beta-mid"]


def test_advertised_models_exclude_wildcards() -> None:
    config = copy.deepcopy(ROUTING_FIXTURE)
    config["routes"].append(
        {
            "name": "catch-all",
            "match": {"models": ["*"]},
            "targets": [{"provider": "beta", "model": "beta-mid"}],
        }
    )
    table = RoutingTable(parse_routing_config(config))
    advertised = table.advertised_models()
    assert "*" not in advertised
    assert advertised == ["fast", "demo-model", "gpt-3.5-turbo"]


def test_pricing_lookup_returns_the_catalog_entry(routing_table: RoutingTable) -> None:
    entry = routing_table.pricing_for("beta", "beta-mid")
    assert entry is not None
    assert entry.output_cost_per_1k_usd == 0.0015
    assert routing_table.pricing_for("beta", "missing") is None


def test_cost_is_computed_per_thousand_tokens(routing_table: RoutingTable) -> None:
    target = routing_table.select("demo-model").chain[1]
    # 2000 prompt tokens at 0.0005 plus 1000 completion tokens at 0.0015.
    assert target.cost_usd(2000, 1000) == pytest.approx(0.0025)


def test_config_rejects_a_target_outside_the_catalog() -> None:
    config = copy.deepcopy(ROUTING_FIXTURE)
    config["routes"][0]["targets"] = [{"provider": "alpha", "model": "ghost"}]
    with pytest.raises(ValidationError, match="not in the catalog"):
        parse_routing_config(config)


def test_config_rejects_duplicate_route_names() -> None:
    config = copy.deepcopy(ROUTING_FIXTURE)
    config["routes"][1]["name"] = "fast"
    with pytest.raises(ValidationError, match="duplicate route name"):
        parse_routing_config(config)


def test_config_rejects_an_unsupported_version() -> None:
    config = copy.deepcopy(ROUTING_FIXTURE)
    config["version"] = 2
    with pytest.raises(ValidationError, match="unsupported routing config version"):
        parse_routing_config(config)


def test_config_rejects_a_route_without_targets() -> None:
    config = copy.deepcopy(ROUTING_FIXTURE)
    config["routes"][0]["targets"] = []
    with pytest.raises(ValidationError):
        parse_routing_config(config)
