"""Retry with exponential backoff and jitter.

Written rather than pulled from a library because the retry loop needs three things a
generic decorator does not give: the upstream ``Retry-After`` must win over our own
curve, the sleep and the RNG must be injectable so the tests assert on exact delays,
and each retry must be observable per provider.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from app.providers.base import ProviderError

T = TypeVar("T")

RandomFn = Callable[[], float]
SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """``jitter`` is the fraction of the delay that is randomised.

    1.0 gives full jitter (``uniform(0, ceiling)``), which spreads a thundering herd
    best; 0.0 gives a deterministic curve, which is what the tests use.
    """

    max_attempts: int = 3
    base_delay: float = 0.2
    factor: float = 2.0
    max_delay: float = 5.0
    jitter: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be between 0 and 1")


def backoff_delay(policy: RetryPolicy, attempt: int, rand: RandomFn = random.random) -> float:
    """Delay before the retry that follows the zero-based ``attempt``."""
    ceiling = min(policy.max_delay, policy.base_delay * policy.factor**attempt)
    fixed = ceiling * (1.0 - policy.jitter)
    return fixed + (ceiling - fixed) * rand()


def next_delay(
    policy: RetryPolicy,
    attempt: int,
    error: ProviderError,
    rand: RandomFn = random.random,
) -> float:
    """Combine our backoff with the upstream's ``Retry-After``.

    A provider that says "wait 2s" knows more than our curve does, so we take the
    larger of the two — but never more than ``max_delay``, otherwise a single upstream
    header could pin a request for minutes.
    """
    delay = backoff_delay(policy, attempt, rand)
    hinted = error.retry_after_seconds
    if hinted is not None:
        delay = max(delay, min(hinted, policy.max_delay))
    return delay


async def run_with_retry(
    operation: Callable[[int], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, float, ProviderError], None] | None = None,
    rand: RandomFn = random.random,
    sleep: SleepFn = asyncio.sleep,
) -> T:
    """Call ``operation(attempt)`` until it succeeds or the policy runs out.

    Only :class:`ProviderError` instances flagged ``retryable`` are retried; anything
    else propagates immediately, because retrying a 400 just wastes the caller's time.
    """
    last_error: ProviderError | None = None
    for attempt in range(policy.max_attempts):
        try:
            return await operation(attempt)
        except ProviderError as exc:
            if not exc.retryable or attempt == policy.max_attempts - 1:
                raise
            last_error = exc
            delay = next_delay(policy, attempt, exc, rand)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            await sleep(delay)

    # Unreachable: the loop either returns or raises, but mypy cannot see that and a
    # bare fallthrough would be a silent None.
    raise AssertionError(f"retry loop exhausted without a result: {last_error}")
