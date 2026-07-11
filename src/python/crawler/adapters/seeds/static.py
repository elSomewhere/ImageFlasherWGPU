"""Layer 3 — static seed source.

Yields a fixed list of seed URLs (e.g. from ``--seed`` flags), once. Simplest
SeedSource implementation.
"""
from __future__ import annotations

from ...core.types import SeedNode


class StaticSeedSource:
    name = "static"

    def __init__(self, urls: list[str]) -> None:
        self._pending = [SeedNode(url=url, context=url, source="seed") for url in urls]

    async def poll(self, limit: int = 1) -> list[SeedNode]:
        if not self._pending:
            return []
        taken, self._pending = self._pending[:limit], self._pending[limit:]
        return taken
