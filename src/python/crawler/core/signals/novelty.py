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
