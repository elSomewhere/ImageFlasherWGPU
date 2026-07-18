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

from ...core.types import MediaCandidate, RightsMetadata
from ...ports.world import FetchError


class CommonsImageSource:
    name = "commons"

    def __init__(self, user_agent: str, request_timeout: float, limit: int = 30, world=None) -> None:
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.limit = limit
        self.world = world

    def _api_url(self, keyword: str) -> str:
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": keyword,
                "gsrnamespace": "6",
                "gsrlimit": str(self.limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "maxlag": "1",
                "format": "json",
            }
        )
        return f"https://commons.wikimedia.org/w/api.php?{params}"

    def _parse(self, payload: dict, api_url: str, keyword: str) -> list[MediaCandidate]:
        pages = payload.get("query", {}).get("pages", {})
        candidates: list[MediaCandidate] = []
        for page in pages.values():
            title = page.get("title", "")
            for image_info in page.get("imageinfo", []):
                image_url = image_info.get("url", "")
                mime = image_info.get("mime", "")
                if not image_url or (mime and not mime.startswith("image/")):
                    continue
                metadata = image_info.get("extmetadata", {})
                def value(name: str) -> str:
                    return str((metadata.get(name, {}) or {}).get("value", ""))
                license_name = value("LicenseShortName") or None
                candidates.append(
                    MediaCandidate(
                        url=image_url,
                        page_url=image_info.get("descriptionurl", api_url),
                        alt=title.replace("File:", ""),
                        context=f"{keyword} {title}",
                        mime_hint=mime,
                        rights=RightsMetadata(
                            status="known" if license_name else "unknown",
                            license=license_name,
                            license_url=value("LicenseUrl") or None,
                            creator=value("Artist") or None,
                            attribution_url=image_info.get("descriptionurl") or None,
                            transformation="resized and color-normalized",
                        ),
                    )
                )
        return candidates

    def fetch_candidates(self, keyword: str) -> list[MediaCandidate]:
        api_url = self._api_url(keyword)
        request = urllib.request.Request(api_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise FetchError(f"Commons API failed for '{keyword}': {error}") from error

        return self._parse(payload, api_url, keyword)

    async def fetch_candidates_async(self, keyword: str) -> list[MediaCandidate]:
        if self.world is not None:
            api_url = self._api_url(keyword)
            resource = await self.world.fetch(api_url)
            try:
                payload = json.loads(resource.body.decode("utf-8"))
            except json.JSONDecodeError as error:
                raise FetchError(f"Commons API returned invalid JSON: {error}") from error
            return self._parse(payload, api_url, keyword)
        return await asyncio.to_thread(self.fetch_candidates, keyword)
