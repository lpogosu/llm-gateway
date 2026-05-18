from app.cache.embeddings import Embedder, EmbeddingError, OllamaEmbedder
from app.cache.semantic import CacheHit, LookupOutcome, SemanticCache, cache_scope
from app.cache.similarity import DegenerateVectorError, best_match, to_unit_vector

__all__ = [
    "CacheHit",
    "DegenerateVectorError",
    "Embedder",
    "EmbeddingError",
    "LookupOutcome",
    "OllamaEmbedder",
    "SemanticCache",
    "best_match",
    "cache_scope",
    "to_unit_vector",
]
