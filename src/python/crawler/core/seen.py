"""Layer 0 — bounded memory primitives for an endless walk.

``RotatingBloomSet`` gives the seen-URL sets a horizon of millions of URLs in a few
megabytes of fixed memory. Two generations rotate on capacity or age: inserts go to
the current generation, membership checks both, and re-touching a value refreshes it
into the current generation so live ground survives rotation (the same semantics the
previous TTL-LRU provided). False positives — rarely skipping an unvisited URL — are
harmless for a wandering art walk; false negatives cannot occur inside the horizon.

``BoundedLRUMap`` is a drop-in dict with insertion-refresh eviction, used to cap
per-host state (visit counters, crawl delays) that would otherwise grow forever.
"""
from __future__ import annotations

import math
import time
from hashlib import blake2b


class _BloomGeneration:
    __slots__ = ("bits", "size_bits", "hash_count", "inserts", "created_at")

    def __init__(self, size_bits: int, hash_count: int) -> None:
        self.bits = bytearray((size_bits + 7) // 8)
        self.size_bits = size_bits
        self.hash_count = hash_count
        self.inserts = 0
        self.created_at = time.monotonic()

    def _indexes(self, value: str):
        digest = blake2b(value.encode("utf-8", errors="replace"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big") | 1  # odd stride
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size_bits

    def __contains__(self, value: str) -> bool:
        return all(self.bits[index >> 3] & (1 << (index & 7)) for index in self._indexes(value))

    def insert(self, value: str) -> None:
        for index in self._indexes(value):
            self.bits[index >> 3] |= 1 << (index & 7)
        self.inserts += 1


class RotatingBloomSet:
    """Two-generation rotating Bloom filter with TTL-LRU-compatible semantics.

    ``add(value)`` returns True when the value was (probably) unseen, False when it
    was (probably) seen — matching ``_TTLLRUSet.add``. Memory is fixed at roughly
    ``2 * capacity * 9.6 bits`` for the default 1% false-positive rate.
    """

    def __init__(self, capacity: int, ttl: float, false_positive_rate: float = 0.01) -> None:
        self.capacity = max(1, int(capacity))
        self.ttl = ttl
        rate = min(max(false_positive_rate, 1e-6), 0.5)
        size_bits = max(64, int(-self.capacity * math.log(rate) / (math.log(2) ** 2)))
        self._size_bits = size_bits
        self._hash_count = max(1, round(size_bits / self.capacity * math.log(2)))
        self._current = _BloomGeneration(size_bits, self._hash_count)
        self._previous: _BloomGeneration | None = None

    def _rotate_if_due(self) -> None:
        generation = self._current
        if generation.inserts >= self.capacity or (
            self.ttl > 0 and time.monotonic() - generation.created_at >= self.ttl
        ):
            self._previous = generation
            self._current = _BloomGeneration(self._size_bits, self._hash_count)

    def add(self, value: str) -> bool:
        self._rotate_if_due()
        in_current = value in self._current
        seen = in_current or (self._previous is not None and value in self._previous)
        if not in_current:
            # Refresh into the current generation so touched values outlive rotation.
            self._current.insert(value)
        return not seen

    def __contains__(self, value: str) -> bool:
        if value in self._current:
            return True
        return self._previous is not None and value in self._previous

    def __len__(self) -> int:
        previous = self._previous.inserts if self._previous is not None else 0
        return self._current.inserts + previous


class BoundedLRUMap(dict):
    """A dict capped at ``capacity`` entries; writes refresh recency, the oldest
    entry is evicted on overflow. Read paths (``[]``, ``get``, ``in``) are plain
    dict operations, so it drops into existing per-host maps unchanged."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = max(1, int(capacity))

    def __setitem__(self, key, value) -> None:
        if key in self:
            super().__delitem__(key)  # move to end (dicts preserve insertion order)
        elif len(self) >= self.capacity:
            oldest = next(iter(self))
            super().__delitem__(oldest)
        super().__setitem__(key, value)
