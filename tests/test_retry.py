"""Backoff arithmetic and the retry loop."""

from __future__ import annotations

import pytest

from app.providers.base import (
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from app.resilience.retry import (
    RetryPolicy,
    SleepFn,
    backoff_delay,
    next_delay,
    run_with_retry,
)


def test_backoff_grows_exponentially_without_jitter() -> None:
    policy = RetryPolicy(base_delay=0.2, factor=2.0, max_delay=10.0, jitter=0.0)
    delays = [backoff_delay(policy, attempt, rand=lambda: 0.5) for attempt in range(4)]
    assert delays == [0.2, 0.4, 0.8, 1.6]


def test_backoff_is_capped_by_max_delay() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=10.0, max_delay=3.0, jitter=0.0)
    assert backoff_delay(policy, 5, rand=lambda: 1.0) == 3.0


def test_full_jitter_spans_zero_to_the_ceiling() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=2.0, max_delay=10.0, jitter=1.0)
    assert backoff_delay(policy, 1, rand=lambda: 0.0) == 0.0
    assert backoff_delay(policy, 1, rand=lambda: 1.0) == 2.0


def test_partial_jitter_keeps_a_fixed_floor() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=2.0, max_delay=10.0, jitter=0.5)
    # Half of the two-second ceiling is fixed, half is randomised.
    assert backoff_delay(policy, 1, rand=lambda: 0.0) == 1.0
    assert backoff_delay(policy, 1, rand=lambda: 1.0) == 2.0


def test_upstream_retry_after_wins_when_it_is_longer() -> None:
    policy = RetryPolicy(base_delay=0.2, factor=2.0, max_delay=30.0, jitter=0.0)
    error = ProviderRateLimitedError("alpha", "slow down", retry_after_seconds=5.0)
    assert next_delay(policy, 0, error, rand=lambda: 0.0) == 5.0


def test_upstream_retry_after_is_capped_by_max_delay() -> None:
    policy = RetryPolicy(base_delay=0.2, factor=2.0, max_delay=3.0, jitter=0.0)
    error = ProviderRateLimitedError("alpha", "slow down", retry_after_seconds=600.0)
    # A single header must not be able to pin a request for ten minutes.
    assert next_delay(policy, 0, error, rand=lambda: 0.0) == 3.0


def test_our_backoff_wins_when_it_is_longer_than_the_hint() -> None:
    policy = RetryPolicy(base_delay=4.0, factor=2.0, max_delay=30.0, jitter=0.0)
    error = ProviderRateLimitedError("alpha", "slow down", retry_after_seconds=1.0)
    assert next_delay(policy, 0, error, rand=lambda: 0.0) == 4.0


def test_rejects_an_impossible_policy() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="jitter"):
        RetryPolicy(jitter=1.5)


async def test_succeeds_without_sleeping_when_the_first_attempt_works() -> None:
    slept: list[float] = []

    async def op(attempt: int) -> str:
        return f"ok-{attempt}"

    result = await run_with_retry(
        op, RetryPolicy(max_attempts=3), sleep=_recorder(slept), rand=lambda: 0.0
    )
    assert result == "ok-0"
    assert slept == []


async def test_retries_a_retryable_failure_then_succeeds() -> None:
    slept: list[float] = []
    attempts: list[int] = []

    async def op(attempt: int) -> str:
        attempts.append(attempt)
        if attempt < 2:
            raise ProviderUnavailableError("alpha", "boom")
        return "recovered"

    policy = RetryPolicy(max_attempts=3, base_delay=0.5, factor=2.0, jitter=0.0)
    result = await run_with_retry(op, policy, sleep=_recorder(slept), rand=lambda: 0.0)

    assert result == "recovered"
    assert attempts == [0, 1, 2]
    assert slept == [0.5, 1.0]


async def test_a_non_retryable_failure_is_not_retried() -> None:
    attempts: list[int] = []

    async def op(attempt: int) -> str:
        attempts.append(attempt)
        raise ProviderBadRequestError("alpha", "malformed")

    with pytest.raises(ProviderBadRequestError):
        await run_with_retry(op, RetryPolicy(max_attempts=5), sleep=_recorder([]))
    assert attempts == [0]


async def test_the_last_failure_propagates_when_attempts_run_out() -> None:
    async def op(attempt: int) -> str:
        raise ProviderUnavailableError("alpha", f"boom {attempt}")

    with pytest.raises(ProviderError, match="boom 2"):
        await run_with_retry(
            op,
            RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0),
            sleep=_recorder([]),
        )


async def test_no_sleep_happens_after_the_final_attempt() -> None:
    slept: list[float] = []

    async def op(attempt: int) -> str:
        raise ProviderUnavailableError("alpha", "boom")

    with pytest.raises(ProviderError):
        await run_with_retry(
            op,
            RetryPolicy(max_attempts=2, base_delay=0.1, jitter=0.0),
            sleep=_recorder(slept),
            rand=lambda: 0.0,
        )
    assert slept == [0.1]


async def test_on_retry_is_notified_for_every_retry() -> None:
    notified: list[tuple[int, float, str]] = []

    async def op(attempt: int) -> str:
        if attempt < 2:
            raise ProviderUnavailableError("alpha", "boom")
        return "ok"

    await run_with_retry(
        op,
        RetryPolicy(max_attempts=3, base_delay=0.1, factor=2.0, jitter=0.0),
        on_retry=lambda attempt, delay, exc: notified.append((attempt, delay, exc.kind)),
        sleep=_recorder([]),
        rand=lambda: 0.0,
    )
    assert notified == [(0, 0.1, "unavailable"), (1, 0.2, "unavailable")]


def _recorder(sink: list[float]) -> SleepFn:
    """A sleep that records the delay instead of spending it."""

    async def sleep(delay: float) -> None:
        sink.append(delay)

    return sleep
