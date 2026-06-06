"""Wiring decisions made at startup."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.container import build_providers
from app.routing.config import load_routing_config
from app.routing.rules import RoutingTable


def settings_with(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_openrouter_is_left_out_when_it_has_no_key() -> None:
    providers = build_providers(settings_with(openrouter_api_key=None))
    # A registered provider with no credentials would answer 401 to everything and open
    # its own circuit; leaving it out makes the router skip it with one clear log line.
    assert sorted(providers) == ["ollama"]


def test_openrouter_is_registered_when_configured() -> None:
    providers = build_providers(settings_with(openrouter_api_key=SecretStr("or-key")))
    assert sorted(providers) == ["ollama", "openrouter"]


def test_the_shipped_routing_table_is_valid_and_self_consistent() -> None:
    table = RoutingTable(load_routing_config(Path("config/routing.yaml")))

    assert table.advertised_models()
    for model in table.advertised_models():
        decision = table.select(model)
        assert decision.chain
        for target in decision.chain:
            assert table.pricing_for(target.provider, target.model) is not None


def test_the_shipped_routing_table_only_names_implemented_providers() -> None:
    table = RoutingTable(load_routing_config(Path("config/routing.yaml")))
    known = {"ollama", "openrouter"}

    assert {entry.provider for entry in table.catalog} <= known


def test_an_owner_label_must_be_short_enough_to_be_a_metric_label() -> None:
    with pytest.raises(ValueError, match="owner"):
        settings_with(api_keys={"sk-x": {"owner": "x" * 65}})
