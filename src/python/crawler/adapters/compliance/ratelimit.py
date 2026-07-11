"""Compliance layer — per-host politeness delay.

Wraps a World and enforces a minimum gap between requests to the same host. A
single shared limiter instance should be passed to every wrapper (pages and media)
so timing is global per host, as in the original SafeFetcher.
"""
from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from ...ports.world import Resource, World


class HostRateLimiter:
    """Async, per-host minimum-delay limiter.

    ``crawl_delays`` is an optional shared map a RobotsGuard can populate from each
    host's ``Crawl-delay`` directive; the effective delay is the larger of the two.
    """

    def __init__(self, delay_seconds: float, crawl_delays: dict[str, float] | None = None) -> None:
        self.delay_seconds = delay_seconds
        self.crawl_delays = crawl_delays if crawl_delays is not None else {}
        self._last_request: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, hostname: str) -> None:
        async with self._lock:
            now = time.monotonic()
            last = self._last_request.get(hostname, 0.0)
            effective_delay = max(self.delay_seconds, self.crawl_delays.get(hostname, 0.0))
            wait_for = effective_delay - (now - last)
            # Reserve the slot before sleeping so concurrent callers queue behind us.
            self._last_request[hostname] = now + max(wait_for, 0.0)
        if wait_for > 0:
            await asyncio.sleep(wait_for)


class RateLimited:
    def __init__(self, inner: World, limiter: HostRateLimiter) -> None:
        self.inner = inner
        self.limiter = limiter

    async def fetch(self, url: str) -> Resource:
        host = urlparse(url).hostname or ""
        await self.limiter.wait(host)
        resource = await self.inner.fetch(url)
        final_host = urlparse(resource.final_url).hostname or ""
        if final_host and final_host != host:
            await self.limiter.wait(final_host)
        return resource
