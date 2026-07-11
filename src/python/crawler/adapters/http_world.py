"""Layer 3 — raw HTTP transport. NO compliance here on purpose.

This is the innermost real-web adapter: it just fetches bytes. SSRF checks, robots,
rate limiting and byte policy live in wrapping decorators (``adapters/compliance``),
so this class stays a pure transport that a generative world can stand in for.
"""
from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

from ..ports.world import FetchError, Resource


class HttpWorld:
    """Fetch a URL over HTTP(S) with a byte ceiling. Follows redirects like the
    stdlib default; the final URL is surfaced so wrapping guards can re-validate it."""

    def __init__(self, user_agent: str, request_timeout: float, max_bytes: int) -> None:
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.max_bytes = max_bytes

    async def fetch(self, url: str) -> Resource:
        return await asyncio.to_thread(self._fetch_blocking, url)

    def _fetch_blocking(self, url: str) -> Resource:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        chunks: list[bytes] = []
        total = 0
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                final_url = response.geturl()
                status = getattr(response, "status", 200) or 200
                content_type = response.headers.get_content_type().lower()
                headers = {key.lower(): value for key, value in response.headers.items()}
                while True:
                    chunk = response.read(16_384)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise FetchError("Response exceeds byte limit")
                    chunks.append(chunk)
        except urllib.error.HTTPError as error:
            retry_after = None
            raw_retry = error.headers.get("Retry-After") if error.headers else None
            if raw_retry and raw_retry.strip().isdigit():
                retry_after = float(raw_retry.strip())
            raise FetchError(f"HTTP {error.code}", status=error.code, retry_after=retry_after) from error
        except urllib.error.URLError as error:
            raise FetchError(str(error.reason)) from error
        return Resource(
            final_url=final_url,
            status=status,
            content_type=content_type,
            body=b"".join(chunks),
            headers=headers,
        )
