"""Prompt embeddings for the semantic cache.

The embedder is a separate port from :class:`~app.providers.base.Provider` because it
has different failure semantics: a chat call that fails is an error the client must
see, while an embedding that fails only means "no cache this time".
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from app.providers.http import build_client


class EmbeddingError(Exception):
    """The embedding backend could not produce a vector."""


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...

    async def aclose(self) -> None: ...


class OllamaEmbedder:
    """Embeddings via Ollama's ``/api/embed``.

    Uses its own client with a short timeout: the cache lookup sits in front of a call
    that may take half a minute, and waiting seconds for the embedding that is supposed
    to *save* that call defeats the purpose.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = client or build_client(
            base_url,
            connect_timeout=min(timeout, 2.0),
            request_timeout=timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.post(
                "/api/embed", json={"model": self._model, "input": text}
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"embedding backend returned {response.status_code}: {response.text[:200]}"
            )
        return _extract_vector(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()


def _extract_vector(payload: Any) -> list[float]:
    if not isinstance(payload, dict):
        raise EmbeddingError("embedding response is not an object")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise EmbeddingError("embedding response carries no vectors")
    vector = embeddings[0]
    if not isinstance(vector, list) or not vector:
        raise EmbeddingError("embedding vector is empty")
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("embedding vector holds non-numeric values") from exc
