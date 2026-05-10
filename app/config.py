"""Runtime settings.

Everything that differs between environments comes from the environment, everything
that describes the model landscape (routes, prices, latency budgets) comes from
``config/routing.yaml``. Nothing secret is ever read from a file in the repository.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiKeyConfig(BaseModel):
    """A single caller: its label for metrics and its private token bucket."""

    model_config = {"frozen": True}

    owner: str
    rate_per_second: float = Field(default=5.0, gt=0)
    burst: int = Field(default=20, ge=1)

    @field_validator("owner")
    @classmethod
    def _owner_is_a_low_cardinality_label(cls, value: str) -> str:
        # The owner ends up as a Prometheus label, so it must be a stable identifier
        # rather than free text or, worse, the key itself.
        if not value or len(value) > 64:
            raise ValueError("owner must be a non-empty label of at most 64 characters")
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server -------------------------------------------------------------
    host: str = "0.0.0.0"  # noqa: S104 - the process is meant to be reachable in a container
    port: int = 8080
    log_level: str = "INFO"
    service_name: str = "llm-gateway"

    # --- callers ------------------------------------------------------------
    # JSON object: {"sk-local-dev": {"owner": "dev", "rate_per_second": 5, "burst": 20}}
    api_keys: dict[str, ApiKeyConfig] = Field(default_factory=dict)

    # --- backing services ---------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    routing_config_path: Path = Path("config/routing.yaml")

    # --- providers ----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: SecretStr | None = None
    openrouter_referer: str = "https://github.com/lpogosu/llm-gateway"

    # --- timeouts and resilience -------------------------------------------
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    retry_max_attempts: int = Field(default=3, ge=1)
    retry_base_delay_seconds: float = Field(default=0.2, gt=0)
    retry_max_delay_seconds: float = Field(default=5.0, gt=0)
    retry_jitter: float = Field(default=1.0, ge=0.0, le=1.0)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_success_threshold: int = Field(default=2, ge=1)
    breaker_recovery_seconds: float = Field(default=30.0, gt=0)
    breaker_half_open_max_calls: int = Field(default=2, ge=1)

    # --- semantic cache -----------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    cache_similarity_threshold: float = Field(default=0.94, ge=0.0, le=1.0)
    cache_max_candidates: int = Field(default=64, ge=1)
    cache_index_size: int = Field(default=2000, ge=1)
    # Sampling above this temperature makes a reused answer indefensible.
    cache_max_temperature: float = Field(default=0.2, ge=0.0)
    embedding_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_seconds: float = Field(default=5.0, gt=0)

    # --- rate limiting ------------------------------------------------------
    rate_limit_enabled: bool = True

    # --- accounting ---------------------------------------------------------
    usage_retention_days: int = Field(default=90, ge=1)

    # --- tracing ------------------------------------------------------------
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unknown log level: {value}")
        return level

    def resolve_api_key(self, token: str) -> ApiKeyConfig | None:
        return self.api_keys.get(token)
