"""Layer 1 — the URL frontier (pure).

A priority frontier over the crawl graph. Phase 0 keeps the original score-ordered
heap behavior; later phases add temperature sampling and host stratification.
"""
from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field

from .scoring import host_of
from .selection import boltzmann_choice


@dataclass
class FrontierItem:
    url: str
    score: float = 0.0
    depth: int = 0
    source: str = "seed"
    referrer: str = ""
    context: str = ""
    direct_score: float = 0.0
    inherited_score: float = 0.0


@dataclass(order=True)
class _PrioritizedItem:
    priority: float
    sequence: int
    item: FrontierItem = field(compare=False)


class URLFrontier:
    def __init__(self, max_size: int = 5000, rng: random.Random | None = None) -> None:
        self.max_size = max_size
        self._heap: list[_PrioritizedItem] = []
        self._queued: set[str] = set()
        self._sequence = itertools.count()
        self._rng = rng or random.Random()

    def add(self, item: FrontierItem) -> bool:
        if item.url in self._queued:
            return False
        if len(self._heap) >= self.max_size:
            # Evict the weakest queued item rather than ossifying: an everlasting
            # crawl must keep admitting fresh discoveries once the frontier fills.
            weakest = max(self._heap, key=lambda prioritized: prioritized.priority)
            if -weakest.priority >= item.score:
                return False  # nothing weaker than the newcomer; drop the newcomer
            self._heap.remove(weakest)
            heapq.heapify(self._heap)
            self._queued.discard(weakest.item.url)
        self._queued.add(item.url)
        heapq.heappush(
            self._heap,
            _PrioritizedItem(priority=-item.score, sequence=next(self._sequence), item=item),
        )
        return True

    def rebuild(self, score_fn) -> None:
        rebuilt: list[_PrioritizedItem] = []
        queued: set[str] = set()
        for prioritized in self._heap:
            item = prioritized.item
            item.score = score_fn(item)
            queued.add(item.url)
            rebuilt.append(
                _PrioritizedItem(
                    priority=-item.score,
                    sequence=prioritized.sequence,
                    item=item,
                )
            )
        heapq.heapify(rebuilt)
        self._heap = rebuilt
        self._queued = queued

    def items(self) -> list[FrontierItem]:
        return [prioritized.item for prioritized in self._heap]

    def pop(self) -> FrontierItem | None:
        if not self._heap:
            return None
        prioritized = heapq.heappop(self._heap)
        self._queued.discard(prioritized.item.url)
        return prioritized.item

    def sample(self, temperature: float, window: int = 64) -> FrontierItem | None:
        """Host-stratified, temperature-sampled selection.

        Over the highest-scoring ``window`` items we keep one representative per host
        (its best item), then Boltzmann-sample *across hosts*. This spreads consecutive
        picks over different sites — structural diversity and politeness on top of the
        host-freshness scoring signal. ``temperature == 0`` still reduces to argmax.
        """
        if not self._heap:
            return None
        candidates = heapq.nsmallest(min(window, len(self._heap)), self._heap)
        best_per_host: dict[str, _PrioritizedItem] = {}
        for prioritized in candidates:
            host = host_of(prioritized.item.url)
            incumbent = best_per_host.get(host)
            if incumbent is None or prioritized.priority < incumbent.priority:
                best_per_host[host] = prioritized
        representatives = list(best_per_host.values())
        scores = [-prioritized.priority for prioritized in representatives]
        chosen = representatives[boltzmann_choice(scores, temperature, self._rng)]
        self._heap.remove(chosen)
        heapq.heapify(self._heap)
        self._queued.discard(chosen.item.url)
        return chosen.item

    def __len__(self) -> int:
        return len(self._heap)
