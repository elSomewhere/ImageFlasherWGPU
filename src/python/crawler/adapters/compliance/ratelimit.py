"""Compliance layer — per-host politeness delay.

Wraps a World and enforces a minimum gap between requests to the same host. A
single shared limiter instance should be passed to every wrapper (pages and media)
so timing is global per host, as in the original SafeFetcher.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ...core.seen import BoundedLRUMap
from ...ports.world import Resource, World


class HostRateLimiter:
    """Async, per-host minimum-delay limiter.

    ``crawl_delays`` is an optional shared map a RobotsGuard can populate from each
    host's ``Crawl-delay`` directive; the effective delay is the larger of the two.
    """

    def __init__(self, delay_seconds: float, crawl_delays: dict[str, float] | None = None) -> None:
        self.delay_seconds = delay_seconds
        self.crawl_delays = crawl_delays if crawl_delays is not None else BoundedLRUMap(10_000)
        self._last_request: dict[str, float] = BoundedLRUMap(10_000)
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


@dataclass
class _OriginState:
    semaphore: asyncio.Semaphore
    next_allowed: float = 0.0
    failures: int = 0
    open_until: float = 0.0
    touched_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class OriginScheduler:
    """Global and per-origin concurrency, delay, and circuit breaking."""

    def __init__(
        self,
        *,
        global_concurrency: int = 16,
        per_origin_concurrency: int = 1,
        delay_seconds: float = 1.0,
        failure_threshold: int = 5,
        circuit_seconds: float = 300.0,
        max_origins: int = 20_000,
    ) -> None:
        self.global_semaphore = asyncio.Semaphore(global_concurrency)
        self.per_origin_concurrency = per_origin_concurrency
        self.delay_seconds = delay_seconds
        self.failure_threshold = failure_threshold
        self.circuit_seconds = circuit_seconds
        self.max_origins = max_origins
        # Bounded like _origins: an endless walk must not keep a Crawl-delay entry
        # for every host it ever met.
        self.crawl_delays: dict[str, float] = BoundedLRUMap(10_000)
        self._origins: dict[str, _OriginState] = {}

    @staticmethod
    def origin(url: str) -> str:
        parsed = urlparse(url)
        default_port = 443 if parsed.scheme == "https" else 80
        return f"{parsed.scheme}://{parsed.hostname or ''}:{parsed.port or default_port}"

    def _state(self, origin: str) -> _OriginState:
        state = self._origins.get(origin)
        if state is None:
            if len(self._origins) >= self.max_origins:
                oldest = min(self._origins, key=lambda key: self._origins[key].touched_at)
                self._origins.pop(oldest, None)
            state = _OriginState(asyncio.Semaphore(self.per_origin_concurrency))
            self._origins[origin] = state
        state.touched_at = time.monotonic()
        return state

    @asynccontextmanager
    async def slot(self, url: str):
        origin = self.origin(url)
        state = self._state(origin)
        now = time.monotonic()
        if state.open_until > now:
            from ...ports.world import FetchError

            raise FetchError(
                f"Origin circuit open until {state.open_until:.3f}", status=503
            )
        await self.global_semaphore.acquire()
        try:
            await state.semaphore.acquire()
            try:
                async with state.lock:
                    now = time.monotonic()
                    host = urlparse(url).hostname or ""
                    delay = max(self.delay_seconds, self.crawl_delays.get(host, 0.0))
                    wait_for = state.next_allowed - now
                    state.next_allowed = max(now, state.next_allowed) + delay
                if wait_for > 0:
                    await asyncio.sleep(wait_for)
                yield
            finally:
                state.semaphore.release()
        finally:
            self.global_semaphore.release()

    def record_success(self, url: str) -> None:
        state = self._state(self.origin(url))
        state.failures = 0
        state.open_until = 0.0

    def record_failure(self, url: str) -> None:
        state = self._state(self.origin(url))
        state.failures += 1
        if state.failures >= self.failure_threshold:
            state.open_until = time.monotonic() + self.circuit_seconds

    def state(self) -> dict:
        now = time.monotonic()
        return {
            "tracked_origins": len(self._origins),
            "open_circuits": sum(1 for state in self._origins.values() if state.open_until > now),
        }
