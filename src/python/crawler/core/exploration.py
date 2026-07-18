"""Homeostatic exploration controller for unattended journeys."""
from __future__ import annotations

import math
import time
from collections import Counter, deque


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _entropy(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    maximum = math.log2(min(total, max(2, len(counts))))
    return entropy / maximum if maximum else 0.0


class ExplorationAutopilot:
    def __init__(
        self,
        base: float = 0.55,
        *,
        enabled: bool = True,
        window: int = 64,
        chapter_seconds: float = 90.0,
        chapter_pages: int = 60,
    ) -> None:
        self.base = _clamp(base)
        self.enabled = enabled
        self.samples: deque[tuple[float, str, str, str, bool]] = deque(maxlen=window)
        self.chapter_seconds = chapter_seconds
        self.chapter_pages = chapter_pages
        self.chapter_started = time.monotonic()
        self.chapter_start_pages = 0
        self.teleport_requested = False

    def set_base(self, value: float) -> float:
        self.base = _clamp(value)
        return self.base

    def record(
        self,
        *,
        novelty: float,
        domain: str,
        language: str = "unknown",
        kind: str = "image",
        success: bool = True,
    ) -> None:
        self.samples.append((_clamp(novelty), domain, language, kind, success))

    def signals(self) -> dict:
        if not self.samples:
            return {
                "mean_novelty": 1.0,
                "domain_entropy": 1.0,
                "language_entropy": 1.0,
                "failure_rate": 0.0,
            }
        samples = list(self.samples)
        return {
            "mean_novelty": sum(sample[0] for sample in samples) / len(samples),
            "domain_entropy": _entropy([sample[1] for sample in samples]),
            "language_entropy": _entropy([sample[2] for sample in samples]),
            "failure_rate": sum(not sample[4] for sample in samples) / len(samples),
        }

    def effective(self, pages_selected: int) -> float:
        signals = self.signals()
        if not self.enabled:
            return self.base
        novelty_pressure = 0.40 - signals["mean_novelty"]
        diversity_pressure = 0.55 - signals["domain_entropy"]
        failure_pressure = signals["failure_rate"] * 0.25
        adjustment = _clamp(
            0.30 * novelty_pressure + 0.25 * diversity_pressure + failure_pressure,
            -0.25,
            0.25,
        )
        chapter_due = (
            time.monotonic() - self.chapter_started >= self.chapter_seconds
            or pages_selected - self.chapter_start_pages >= self.chapter_pages
        )
        if chapter_due:
            self.teleport_requested = (
                signals["mean_novelty"] < 0.30 or signals["domain_entropy"] < 0.35
            )
            self.chapter_started = time.monotonic()
            self.chapter_start_pages = pages_selected
        return _clamp(self.base + adjustment)

    def consume_teleport(self) -> bool:
        requested = self.teleport_requested
        self.teleport_requested = False
        return requested
