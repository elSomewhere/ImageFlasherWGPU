"""Layer 1 — optional novelty signal (Novelty Search, Lehman & Stanley).

Instead of chasing a goal, reward *behavioral diversity*: an image is novel if it is
unlike the recently-seen ones. We use a 64-bit average-hash (aHash) as a cheap
perceptual fingerprint; novelty = normalized Hamming distance to the nearest neighbor
in a bounded rolling archive. Doubles as near-duplicate suppression so the wall stays
diverse instead of flashing the same logo repeatedly.

Pure and dependency-free (integers only); the archive is bounded so it runs forever.
"""
from __future__ import annotations

from collections import deque

HASH_BITS = 64


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class NoveltyArchive:
    def __init__(self, capacity: int = 4096) -> None:
        self._hashes: deque[int] = deque(maxlen=capacity)

    def novelty(self, fingerprint: int) -> float:
        """Return normalized distance [0,1] to the nearest archived fingerprint.
        1.0 when the archive is empty (everything is novel at first)."""
        if not self._hashes:
            return 1.0
        nearest = min(_hamming(fingerprint, existing) for existing in self._hashes)
        return nearest / HASH_BITS

    def add(self, fingerprint: int) -> None:
        self._hashes.append(fingerprint)

    def __len__(self) -> int:
        return len(self._hashes)


class VisualNoveltyArchive:
    """Bounded dHash + color novelty used for actual image admission."""

    def __init__(self, capacity: int = 4096) -> None:
        self._features: deque[tuple[int, tuple[float, ...]]] = deque(maxlen=capacity)

    def novelty(self, dhash: int, histogram: tuple[float, ...]) -> float:
        if not self._features:
            return 1.0
        nearest = 1.0
        for existing_hash, existing_histogram in self._features:
            hash_distance = _hamming(dhash, existing_hash) / HASH_BITS
            if histogram and existing_histogram:
                histogram_distance = min(
                    1.0,
                    sum(abs(left - right) for left, right in zip(histogram, existing_histogram)) / 2.0,
                )
            else:
                histogram_distance = 0.0
            distance = 0.75 * hash_distance + 0.25 * histogram_distance
            nearest = min(nearest, distance)
        return nearest

    def add(self, dhash: int, histogram: tuple[float, ...]) -> None:
        self._features.append((dhash, histogram))

    def __len__(self) -> int:
        return len(self._features)
