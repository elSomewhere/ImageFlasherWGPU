"""Layer 3 — an in-memory World with no network and no compliance.

This exists to prove the seam: the exact same CrawlEngine runs against a world that
is not the internet. It is also the shape a future ``GenerativeWorld`` (an AI-imagined
web) would take — implement ``fetch`` and nothing in the core changes.
"""
from __future__ import annotations

from ..ports.world import FetchError, Resource


class InMemoryWorld:
    """Serves canned Resources from a ``{url: Resource}`` map."""

    def __init__(self, pages: dict[str, Resource]) -> None:
        self.pages = pages
        self.fetched: list[str] = []

    async def fetch(self, url: str) -> Resource:
        self.fetched.append(url)
        if url not in self.pages:
            raise FetchError(f"No such node in world: {url}")
        return self.pages[url]

    @staticmethod
    def html(url: str, body: str, content_type: str = "text/html") -> Resource:
        return Resource(final_url=url, status=200, content_type=content_type, body=body.encode("utf-8"))

    @staticmethod
    def image(url: str, payload: bytes, content_type: str = "image/png") -> Resource:
        return Resource(final_url=url, status=200, content_type=content_type, body=payload)
