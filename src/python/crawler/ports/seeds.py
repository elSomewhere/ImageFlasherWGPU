"""Layer 2 — the SeedSource port.

A SeedSource injects fresh entry points into the crawl. Generic sources
(random Wikipedia articles, static seeds, later Common Crawl samples) let the
crawler start and self-restart with zero domain-specific configuration.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import SeedNode


@runtime_checkable
class SeedSource(Protocol):
    name: str

    async def poll(self, limit: int = 1) -> list[SeedNode]:
        """Return up to ``limit`` fresh seed nodes (possibly empty)."""
        ...
