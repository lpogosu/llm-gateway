"""Circuit breaker state machine."""

from __future__ import annotations

import pytest

from app.resilience.circuit import BreakerRegistry, CircuitBreaker, CircuitState
from tests.fakes import FakeClock


def breaker(
    clock: FakeClock,
    *,
    failure_threshold: int = 3,
    success_threshold: int = 2,
    recovery_seconds: float = 10.0,
    half_open_max_calls: int = 1,
) -> CircuitBreaker:
    return CircuitBreaker(
        "alpha",
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        recovery_seconds=recovery_seconds,
        half_open_max_calls=half_open_max_calls,
        clock=clock,
    )


def test_starts_closed_and_admits_calls() -> None:
    assert breaker(FakeClock()).state is CircuitState.CLOSED


def test_opens_only_on_consecutive_failures() -> None:
    cb = breaker(FakeClock())
    cb.record_failure()
    cb.record_failure()
    cb.record_success()  # a provider that answers is not half broken
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_open_circuit_refuses_calls() -> None:
    cb = breaker(FakeClock(), failure_threshold=1)
    cb.record_failure()
    assert cb.allow() is False


def test_recovery_window_moves_the_circuit_to_half_open() -> None:
    clock = FakeClock()
    cb = breaker(clock, failure_threshold=1)
    cb.record_failure()

    clock.advance(9.9)
    assert cb.state is CircuitState.OPEN
    clock.advance(0.2)
    assert cb.state is CircuitState.HALF_OPEN


def test_half_open_admits_only_the_configured_number_of_probes() -> None:
    clock = FakeClock()
    cb = breaker(clock, failure_threshold=1, half_open_max_calls=2)
    cb.record_failure()
    clock.advance(11)

    assert cb.allow() is True
    assert cb.allow() is True
    assert cb.allow() is False  # the two probes are still in flight


def test_half_open_closes_after_enough_successful_probes() -> None:
    clock = FakeClock()
    cb = breaker(clock, failure_threshold=1, success_threshold=2, half_open_max_calls=1)
    cb.record_failure()
    clock.advance(11)

    assert cb.allow() is True
    cb.record_success()
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.allow() is True
    cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_a_failed_probe_reopens_the_circuit_with_a_fresh_window() -> None:
    clock = FakeClock()
    cb = breaker(clock, failure_threshold=1)
    cb.record_failure()
    clock.advance(11)
    assert cb.state is CircuitState.HALF_OPEN

    cb.allow()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN

    # The window restarts from the moment of the failed probe, not from the first
    # failure, otherwise a broken provider would be probed again immediately.
    clock.advance(9.0)
    assert cb.state is CircuitState.OPEN
    clock.advance(2.0)
    assert cb.state is CircuitState.HALF_OPEN


def test_closing_resets_the_failure_counter() -> None:
    clock = FakeClock()
    cb = breaker(clock, failure_threshold=2, success_threshold=1)
    cb.record_failure()
    cb.record_failure()
    clock.advance(11)
    cb.allow()
    cb.record_success()
    assert cb.state is CircuitState.CLOSED

    cb.record_failure()
    assert cb.state is CircuitState.CLOSED


def test_transitions_are_reported_once_each() -> None:
    clock = FakeClock()
    seen: list[tuple[str, CircuitState]] = []
    cb = CircuitBreaker(
        "alpha",
        failure_threshold=1,
        success_threshold=1,
        recovery_seconds=5.0,
        half_open_max_calls=1,
        clock=clock,
        on_transition=lambda name, state: seen.append((name, state)),
    )
    cb.record_failure()
    clock.advance(6)
    cb.allow()
    cb.record_success()

    assert seen == [
        ("alpha", CircuitState.OPEN),
        ("alpha", CircuitState.HALF_OPEN),
        ("alpha", CircuitState.CLOSED),
    ]


def test_rejects_nonsense_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds must be positive"):
        CircuitBreaker("alpha", failure_threshold=0)


def test_registry_hands_out_one_breaker_per_provider() -> None:
    registry = BreakerRegistry(
        failure_threshold=1,
        success_threshold=1,
        recovery_seconds=1.0,
        half_open_max_calls=1,
    )
    assert registry.get("alpha") is registry.get("alpha")
    assert registry.get("alpha") is not registry.get("beta")

    registry.get("alpha").record_failure()
    assert registry.snapshot() == {"alpha": CircuitState.OPEN, "beta": CircuitState.CLOSED}
