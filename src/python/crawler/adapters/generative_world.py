"""Layer 3 — placeholder for a generative / AI-imagined web.

Not implemented yet. It is documented here to mark the seam: a generative world is
just another ``World`` — implement ``async def fetch(self, url) -> Resource`` returning
synthesized pages/media, assemble it with ``compliance=False`` (no robots/SSRF/rate
limit), and the core engine runs unchanged. See ``InMemoryWorld`` for the shape.
"""
from __future__ import annotations

from ..ports.world import Resource


class GenerativeWorld:
    async def fetch(self, url: str) -> Resource:  # pragma: no cover - not implemented
        raise NotImplementedError(
            "GenerativeWorld is a documented stub. Implement fetch() to synthesize a web."
        )
