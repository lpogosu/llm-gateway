"""One small shim over redis-py's return types.

redis-py declares its command surface once for both the sync and the async client, so a
command is statically ``Awaitable[T] | T``. Awaiting that directly does not type-check
under ``--strict``. Rather than sprinkling casts, every call site that hits the typed
part of the surface goes through :func:`resolve`, which is also correct at runtime.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def resolve(value: Awaitable[T] | T) -> T:
    if isinstance(value, Awaitable):
        return await value
    return value
