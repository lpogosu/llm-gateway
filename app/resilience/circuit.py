"""Per-provider circuit breaker.

State lives in the process, not in Redis. A breaker exists to stop *this* replica from
hammering a dead upstream and to fail fast for *this* replica's callers; sharing the
state would add a Redis round-trip to the hot path and let one poisoned replica take
the fleet's traffic away from a healthy provider.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

Clock = Callable[[], float]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


STATE_CODE = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}


class CircuitBreaker:
    """Classic three-state breaker.

    Closed counts consecutive failures; a single success resets the counter, because a
    provider that answers is not "half broken". Open refuses calls until the recovery
    window elapses, then admits a bounded number of probes. Half-open needs
    ``success_threshold`` clean probes to close, and one failure sends it straight back
    to open with a fresh window.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_seconds: float = 30.0,
        half_open_max_calls: int = 2,
        clock: Clock = time.monotonic,
        on_transition: Callable[[str, CircuitState], None] | None = None,
    ) -> None:
        if failure_threshold < 1 or success_threshold < 1 or half_open_max_calls < 1:
            raise ValueError("breaker thresholds must be positive")
        self.name = name
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._recovery_seconds = recovery_seconds
        self._half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._on_transition = on_transition

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._probes_in_flight = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        self._maybe_half_open()
        return self._state

    def allow(self) -> bool:
        """Ask for permission to call the provider.

        In half-open this reserves one of the probe slots, so the caller must always
        report the outcome with :meth:`record_success` or :meth:`record_failure`.
        """
        self._maybe_half_open()
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            return False
        if self._probes_in_flight >= self._half_open_max_calls:
            return False
        self._probes_in_flight += 1
        return True

    def record_success(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._probes_in_flight = max(0, self._probes_in_flight - 1)
            self._successes += 1
            if self._successes >= self._success_threshold:
                self._transition(CircuitState.CLOSED)
            return
        self._failures = 0

    def record_failure(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._probes_in_flight = max(0, self._probes_in_flight - 1)
            self._transition(CircuitState.OPEN)
            return
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._transition(CircuitState.OPEN)

    def _maybe_half_open(self) -> None:
        if self._state is not CircuitState.OPEN:
            return
        if self._clock() - self._opened_at >= self._recovery_seconds:
            self._transition(CircuitState.HALF_OPEN)

    def _transition(self, target: CircuitState) -> None:
        if target is self._state:
            if target is CircuitState.OPEN:
                # Re-opening from half-open must restart the recovery window even
                # though the state name did not change.
                self._opened_at = self._clock()
            return
        self._state = target
        self._failures = 0
        self._successes = 0
        self._probes_in_flight = 0
        if target is CircuitState.OPEN:
            self._opened_at = self._clock()
        if self._on_transition is not None:
            self._on_transition(self.name, target)


class BreakerRegistry:
    """One breaker per provider, created on first use."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        success_threshold: int,
        recovery_seconds: float,
        half_open_max_calls: int,
        clock: Clock = time.monotonic,
        on_transition: Callable[[str, CircuitState], None] | None = None,
    ) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._recovery_seconds = recovery_seconds
        self._half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._on_transition = on_transition

    def get(self, provider: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(
                provider,
                failure_threshold=self._failure_threshold,
                success_threshold=self._success_threshold,
                recovery_seconds=self._recovery_seconds,
                half_open_max_calls=self._half_open_max_calls,
                clock=self._clock,
                on_transition=self._on_transition,
            )
            self._breakers[provider] = breaker
        return breaker

    def snapshot(self) -> dict[str, CircuitState]:
        return {name: breaker.state for name, breaker in self._breakers.items()}
