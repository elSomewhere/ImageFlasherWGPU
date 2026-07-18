"""Multilingual Wikipedia seed source with external-link escape routes."""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections import deque
from urllib.parse import urlencode

from ...core.types import SeedNode


DEFAULT_WIKIS = ("en", "de", "fr", "es", "ja", "ru", "pt", "it")


class WikipediaRandomSeedSource:
    name = "wikipedia_random"

    def __init__(self, user_agent: str, request_timeout: float, wikis: tuple[str, ...] = DEFAULT_WIKIS, world=None) -> None:
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.wikis = wikis
        self.world = world
        self._cursor = 0
        self._pending: deque[SeedNode] = deque()

    def _next_wiki(self) -> str:
        wiki = self.wikis[self._cursor % len(self.wikis)]
        self._cursor += 1
        return wiki

    @staticmethod
    def _api_url(wiki: str) -> str:
        query = urlencode(
            {
                "action": "query",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": "1",
                "prop": "info|extlinks",
                "inprop": "url",
                "ellimit": "20",
                "maxlag": "1",
                "format": "json",
            }
        )
        return f"https://{wiki}.wikipedia.org/w/api.php?{query}"

    @staticmethod
    def _parse(payload: dict) -> list[SeedNode]:
        results: list[SeedNode] = []
        for page in payload.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            if page.get("fullurl"):
                results.append(SeedNode(url=page["fullurl"], context=title, source="wikipedia_random"))
            for link in page.get("extlinks", []):
                url = link.get("*") or link.get("url")
                if url:
                    results.append(SeedNode(url=url, context=title, source="wikipedia_external"))
        return results

    async def _fetch(self, wiki: str) -> list[SeedNode]:
        url = self._api_url(wiki)
        if self.world is not None:
            resource = await self.world.fetch(url)
            return self._parse(json.loads(resource.body.decode("utf-8")))
        def blocking() -> list[SeedNode]:
            request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    return self._parse(json.loads(response.read().decode("utf-8")))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                return []
        return await asyncio.to_thread(blocking)

    async def poll(self, limit: int = 1) -> list[SeedNode]:
        while len(self._pending) < max(1, limit):
            found = await self._fetch(self._next_wiki())
            if not found:
                break
            # Alternate between the article and shuffled external exits over later polls.
            self._pending.extend(found)
        taken = [self._pending.popleft() for _ in range(min(limit, len(self._pending)))]
        return taken
