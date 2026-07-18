"""Compliance layer — robots.txt gate.

Wraps a World and refuses URLs disallowed for our user-agent, checking both the
requested URL and the post-redirect final URL. As with every compliance wrapper,
the core engine is oblivious to it — a generative world simply omits it.

We fetch robots.txt with our *own* User-Agent (the stdlib ``RobotFileParser.read()``
uses ``Python-urllib``, which many sites — Wikipedia, Cloudflare-fronted hosts — answer
with 403/429; the stdlib then flips ``disallow_all`` and over-blocks the entire host).
Non-200 / missing robots.txt is treated as allow-all, the conventional crawler policy,
so we honor real rules without letting a blocked robots.txt fetch kill collection.
"""
from __future__ import annotations

import asyncio
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse

from ...ports.world import FetchError, Resource, World

MAX_ROBOTS_BYTES = 512_000


class RobotsGuard:
    def __init__(
        self,
        inner: World,
        user_agent: str,
        request_timeout: float = 8.0,
        cache_ttl: float = 3600.0,
        crawl_delays: dict[str, float] | None = None,
    ) -> None:
        self.inner = inner
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.cache_ttl = cache_ttl
        # Shared with the rate limiter so Crawl-delay directives are honored.
        self.crawl_delays = crawl_delays if crawl_delays is not None else {}
        # host_robots_url -> (parser_or_None, fetched_at_monotonic)
        self._robots: dict[str, tuple[urllib.robotparser.RobotFileParser | None, float]] = {}
        self._lock = asyncio.Lock()

    def _read_parser(self, robots_url: str) -> urllib.robotparser.RobotFileParser | None:
        request = urllib.request.Request(robots_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                status = getattr(response, "status", 200) or 200
                if status != 200:
                    return None  # missing/!=200 -> allow-all (conventional)
                raw = response.read(MAX_ROBOTS_BYTES)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(raw.decode("utf-8", errors="replace").splitlines())
        return parser

    async def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        now = time.monotonic()
        async with self._lock:
            entry = self._robots.get(robots_url)
        fresh = entry is not None and (now - entry[1]) < self.cache_ttl
        if not fresh:
            parser = await asyncio.to_thread(self._read_parser, robots_url)
            async with self._lock:
                self._robots[robots_url] = (parser, now)
            if parser is not None:
                delay = parser.crawl_delay(self.user_agent)
                if delay:
                    self.crawl_delays[host] = float(delay)
        else:
            parser = entry[0]
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    async def fetch(self, url: str) -> Resource:
        if not await self._allowed(url):
            raise FetchError("Disallowed by robots.txt")
        resource = await self.inner.fetch(url)
        if resource.final_url != url and not await self._allowed(resource.final_url):
            raise FetchError("Final URL disallowed by robots.txt")
        return resource


@dataclass
class _PolicyEntry:
    parser: urllib.robotparser.RobotFileParser | None
    fetched_at: float
    ttl: float
    unreachable: bool = False


class RobotsPolicy:
    """RFC 9309 policy used by the production aiohttp transport."""

    def __init__(
        self,
        robots_world: World,
        user_agent: str,
        *,
        scheduler=None,
        default_ttl: float = 3600.0,
        max_ttl: float = 24 * 3600.0,
        max_entries: int = 10_000,
    ) -> None:
        self.robots_world = robots_world
        self.user_agent = user_agent
        self.scheduler = scheduler
        self.default_ttl = default_ttl
        self.max_ttl = max_ttl
        self.max_entries = max_entries
        self._cache: dict[str, _PolicyEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def robots_url(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _ttl(self, headers) -> float:
        cache_control = headers.get("cache-control", "")
        for directive in cache_control.split(","):
            key, separator, value = directive.strip().partition("=")
            if separator and key.lower() == "max-age" and value.strip().isdigit():
                return min(self.max_ttl, max(0.0, float(value.strip())))
        return min(self.default_ttl, self.max_ttl)

    def _store(self, key: str, entry: _PolicyEntry) -> None:
        if len(self._cache) >= self.max_entries and key not in self._cache:
            oldest = min(self._cache, key=lambda url: self._cache[url].fetched_at)
            self._cache.pop(oldest, None)
            self._locks.pop(oldest, None)
        self._cache[key] = entry

    async def _load(self, robots_url: str) -> _PolicyEntry:
        try:
            resource = await self.robots_world.fetch(robots_url)
        except FetchError as error:
            # RFC 9309: 4xx means unavailable (allow); server/network failure is
            # unreachable and therefore a complete temporary disallow.
            unreachable = error.status is None or 500 <= error.status <= 599
            return _PolicyEntry(None, time.monotonic(), self.default_ttl, unreachable)

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(resource.body.decode("utf-8", errors="replace").splitlines())
        if self.scheduler is not None:
            delay = parser.crawl_delay(self.user_agent)
            if delay:
                host = urlparse(robots_url).hostname or ""
                self.scheduler.crawl_delays[host] = float(delay)
        return _PolicyEntry(parser, time.monotonic(), self._ttl(resource.headers))

    async def allowed(self, url: str) -> bool:
        key = self.robots_url(url)
        now = time.monotonic()
        entry = self._cache.get(key)
        if entry is None or now - entry.fetched_at >= entry.ttl:
            lock = self._locks.setdefault(key, asyncio.Lock())
            async with lock:
                entry = self._cache.get(key)
                now = time.monotonic()
                if entry is None or now - entry.fetched_at >= entry.ttl:
                    entry = await self._load(key)
                    self._store(key, entry)
        if entry.unreachable:
            return False
        return entry.parser is None or entry.parser.can_fetch(self.user_agent, url)

    async def require_allowed(self, url: str) -> None:
        if not await self.allowed(url):
            raise FetchError("Disallowed by robots.txt")

    def state(self) -> dict:
        return {
            "robots_cached": len(self._cache),
            "robots_unreachable": sum(1 for entry in self._cache.values() if entry.unreachable),
        }
