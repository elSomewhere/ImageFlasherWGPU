from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field


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
    def __init__(self, max_size: int = 5000) -> None:
        self.max_size = max_size
        self._heap: list[_PrioritizedItem] = []
        self._queued: set[str] = set()
        self._sequence = itertools.count()

    def add(self, item: FrontierItem) -> bool:
        if item.url in self._queued or len(self._heap) >= self.max_size:
            return False
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

    def __len__(self) -> int:
        return len(self._heap)

