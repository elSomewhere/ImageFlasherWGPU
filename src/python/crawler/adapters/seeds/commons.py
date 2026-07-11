"""Layer 3 — Wikimedia Commons image source.

A keyword-driven media source: it queries the Commons API and returns image
candidates directly (not page seeds). Generic and robots-friendly. One of several
sources the engine can draw on; nothing here is domain-specific.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from urllib.parse import urlencode

from ...core.types import MediaCandidate
from ...ports.world import FetchError


class CommonsImageSource:
    name = "commons"

    def __init__(self, user_agent: str, request_timeout: float, limit: int = 30) -> None:
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.limit = limit

    def fetch_candidates(self, keyword: str) -> list[MediaCandidate]:
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": keyword,
                "gsrnamespace": "6",
                "gsrlimit": str(self.limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "format": "json",
            }
        )
        api_url = f"https://commons.wikimedia.org/w/api.php?{params}"
        request = urllib.request.Request(api_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise FetchError(f"Commons API failed for '{keyword}': {error}") from error

        pages = payload.get("query", {}).get("pages", {})
        candidates: list[MediaCandidate] = []
        for page in pages.values():
            title = page.get("title", "")
            for image_info in page.get("imageinfo", []):
                image_url = image_info.get("url", "")
                mime = image_info.get("mime", "")
                if not image_url or (mime and not mime.startswith("image/")):
                    continue
                candidates.append(
                    MediaCandidate(
                        url=image_url,
                        page_url=api_url,
                        alt=title.replace("File:", ""),
                        context=f"{keyword} {title}",
                    )
                )
        return candidates

    async def fetch_candidates_async(self, keyword: str) -> list[MediaCandidate]:
        return await asyncio.to_thread(self.fetch_candidates, keyword)
