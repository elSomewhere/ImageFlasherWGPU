"""Layer 1 — temperature controllers (pure, optional).

The walk temperature can be a fixed dial or drive itself so the installation evolves
unattended:

- ``static``   : fixed value; the control plane can still set it live.
- ``ou``       : Ornstein-Uhlenbeck mean-reverting random walk — it breathes between
                 focused and wandering on its own.
- ``adaptive`` : reheating. When recent novelty falls (stuck in a cluster) it raises
                 T to wander off; when novelty is high (a rich vein) it lowers T to
                 exploit. Simulated-annealing-with-reheating driven by what it finds.

Each ``update(novelty)`` takes the latest novelty in [0,1] and returns the new
temperature. RNG is injected for reproducibility.
"""
from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class StaticTemperature:
    mode = "static"

    def __init__(self, value: float) -> None:
        self.value = value

    def update(self, novelty: float) -> float:
        return self.value


class OUDriftTemperature:
    mode = "ou"

    def __init__(self, mean: float, low: float, high: float, rng, theta: float = 0.05, sigma: float = 0.15) -> None:
        self.mean = mean
        self.low = low
        self.high = high
        self.rng = rng
        self.theta = theta
        self.sigma = sigma
        self.value = mean

    def update(self, novelty: float) -> float:
        shock = self.sigma * (self.rng.random() * 2.0 - 1.0)
        self.value = _clamp(self.value + self.theta * (self.mean - self.value) + shock, self.low, self.high)
        return self.value


class AdaptiveReheatTemperature:
    mode = "adaptive"

    def __init__(self, low: float, high: float, gain: float = 0.1, start: float | None = None) -> None:
        self.low = low
        self.high = high
        self.gain = gain
        self.value = start if start is not None else (low + high) / 2.0

    def update(self, novelty: float) -> float:
        # Low novelty -> aim hot (wander); high novelty -> aim cool (exploit).
        target = self.low + (self.high - self.low) * (1.0 - _clamp(novelty, 0.0, 1.0))
        self.value = _clamp(self.value + self.gain * (target - self.value), self.low, self.high)
        return self.value


def make_controller(mode: str, *, temperature: float, low: float, high: float, rng):
    """Build a controller. ``static`` returns None so the engine keeps the value
    manually controllable via the control plane."""
    mode = (mode or "static").lower()
    if mode == "ou":
        return OUDriftTemperature(mean=temperature, low=low, high=high, rng=rng)
    if mode == "adaptive":
        return AdaptiveReheatTemperature(low=low, high=high, start=temperature)
    return None
