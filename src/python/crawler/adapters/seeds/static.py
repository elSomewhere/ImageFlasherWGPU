"""Layer 3 — static seed source.

Yields a fixed list of seed URLs. In one-shot mode (default) each URL is yielded
once — e.g. bootstrap seeds. In cycle mode it round-robins the list forever, which
lets operator seeds serve as the recurring refill/teleport source of a link-only
walk: if the journey ever dies, it restarts from the operator's own entry points.
Frontier dedup and the seen-URL horizon make re-injection harmless while the walk
is alive.
"""
from __future__ import annotations

from ...core.types import SeedNode


class StaticSeedSource:
    name = "static"

    def __init__(self, urls: list[str], *, cycle: bool = False) -> None:
        self._urls = list(dict.fromkeys(urls))
        self._cycle = cycle
        self._cursor = 0
        self._pending = [SeedNode(url=url, context=url, source="seed") for url in self._urls]

    async def poll(self, limit: int = 1) -> list[SeedNode]:
        if not self._cycle:
            if not self._pending:
                return []
            taken, self._pending = self._pending[:limit], self._pending[limit:]
            return taken
        if not self._urls:
            return []
        taken = []
        for _ in range(min(max(1, limit), len(self._urls))):
            url = self._urls[self._cursor % len(self._urls)]
            self._cursor += 1
            taken.append(SeedNode(url=url, context=url, source="seed"))
        return taken
