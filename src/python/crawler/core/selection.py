"""Layer 1 — Boltzmann (softmax) selection with a temperature knob.

This is the heart of the "interesting walk". Instead of always taking the highest
score (greedy argmax), we *sample* from the frontier with

    P(item) ∝ exp(score / T)

- T → 0   : argmax        — focused, tunnels into the best region (old behavior)
- T → ∞   : uniform       — pure wandering, ignores score
- T in between            : a random walk biased by interest

The RNG is injected so runs are reproducible in tests.
"""
from __future__ import annotations

import math
from typing import Sequence

# Below this temperature we treat selection as exact argmax (avoids div-by-zero and
# overflow while giving the "fully focused" end of the dial).
ARGMAX_EPSILON = 1e-9


def boltzmann_weights(scores: Sequence[float], temperature: float) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    if temperature <= ARGMAX_EPSILON:
        weights = [0.0] * n
        weights[max(range(n), key=lambda i: scores[i])] = 1.0
        return weights
    highest = max(scores)
    exps = [math.exp((score - highest) / temperature) for score in scores]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / n] * n
    return [value / total for value in exps]


def boltzmann_choice(scores: Sequence[float], temperature: float, rng) -> int:
    """Return the index of the sampled element."""
    weights = boltzmann_weights(scores, temperature)
    threshold = rng.random()
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return len(weights) - 1
