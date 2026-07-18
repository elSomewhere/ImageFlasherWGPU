"""Compliance layer — retry with backoff on rate-limit / unavailable responses.

Wraps a World: when the inner fetch fails with 429 (Too Many Requests) or 503
(Service Unavailable), wait — honoring ``Retry-After`` when the server sent it,
otherwise exponential backoff — and retry a bounded number of times. This is the
polite response to being throttled and directly reduces dropped media on busy CDNs.
"""
from __future__ import annotations

import asyncio
import random

from ...ports.world import FetchError, Resource, World

RETRY_STATUSES = frozenset({429, 502, 503, 504})


class BackoffOnStatus:
    def __init__(self, inner: World, max_retries: int = 2, base_delay: float = 1.0, cap: float = 30.0, rng=None) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.cap = cap
        self.rng = rng or random.Random()

    async def fetch(self, url: str) -> Resource:
        attempt = 0
        while True:
            try:
                return await self.inner.fetch(url)
            except FetchError as error:
                if error.status not in RETRY_STATUSES or attempt >= self.max_retries:
                    raise
                delay = error.retry_after if error.retry_after is not None else self.rng.uniform(0.0, self.base_delay * (2 ** attempt))
                await asyncio.sleep(min(delay, self.cap))
                attempt += 1

    async def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            await close()
