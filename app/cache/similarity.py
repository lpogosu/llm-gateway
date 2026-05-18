"""Vector helpers for the semantic cache.

Vectors are L2-normalised once, on write, and stored as float32. Cosine similarity is
then a dot product, and the candidate scan becomes a single matrix-vector multiply
instead of a Python loop over cosine computations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


class DegenerateVectorError(ValueError):
    """A zero-length vector has no direction, so cosine similarity is undefined."""


def to_unit_vector(values: Sequence[float]) -> Vector:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise DegenerateVectorError("expected a non-empty one-dimensional vector")
    norm = float(np.linalg.norm(array))
    if norm == 0.0 or not np.isfinite(norm):
        raise DegenerateVectorError("vector has zero or non-finite norm")
    return (array / norm).astype(np.float32)


def pack(vector: Vector) -> bytes:
    return vector.tobytes()


def unpack(blob: bytes, *, dimensions: int) -> Vector:
    array = np.frombuffer(blob, dtype=np.float32)
    if array.size != dimensions:
        raise DegenerateVectorError(
            f"stored vector has {array.size} dimensions, expected {dimensions}"
        )
    return array


def best_match(query: Vector, candidates: Sequence[Vector]) -> tuple[int, float]:
    """Return the index and cosine similarity of the closest candidate.

    Returns ``(-1, -1.0)`` for an empty candidate set, which is the cold-start case and
    not an error.
    """
    if not candidates:
        return -1, -1.0
    matrix = np.stack(candidates)
    scores = matrix @ query
    index = int(np.argmax(scores))
    return index, float(scores[index])
