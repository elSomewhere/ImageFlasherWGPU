"""Layer 3 — random-Wikipedia seed source.

An infinite, generic, robots-friendly entry into "all of the web": each poll returns
a random article URL. This is what lets the crawler start and self-restart with zero
domain-specific configuration. Multiple language wikis widen the spread.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

from ...core.types import SeedNode

DEFAULT_WIKIS = ("en", "de", "fr", "es", "ja", "ru", "pt", "it")


class WikipediaRandomSeedSource:
    name = "wikipedia_random"

    def __init__(self, user_agent: str, request_timeout: float, wikis: tuple[str, ...] = DEFAULT_WIKIS) -> None:
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.wikis = wikis
        self._cursor = 0

    def _next_wiki(self) -> str:
        wiki = self.wikis[self._cursor % len(self.wikis)]
        self._cursor += 1
        return wiki

    def _fetch_one(self, wiki: str) -> SeedNode | None:
        url = f"https://{wiki}.wikipedia.org/api/rest_v1/page/random/summary"
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        page_url = (payload.get("content_urls", {}).get("desktop", {}) or {}).get("page", "")
        if not page_url:
            return None
        title = payload.get("title", "")
        return SeedNode(url=page_url, context=title, source="wikipedia_random")

    def _poll_blocking(self, limit: int) -> list[SeedNode]:
        seeds: list[SeedNode] = []
        for _ in range(max(1, limit)):
            seed = self._fetch_one(self._next_wiki())
            if seed is not None:
                seeds.append(seed)
        return seeds

    async def poll(self, limit: int = 1) -> list[SeedNode]:
        return await asyncio.to_thread(self._poll_blocking, limit)
