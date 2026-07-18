"""Pooled async HTTP transport with safe manual redirect handling."""
from __future__ import annotations

import asyncio
import email.utils
from datetime import datetime, timezone
from typing import Awaitable, Callable
from urllib.parse import urljoin

import aiohttp

from ..core.url_policy import canonicalize_url
from ..ports.world import FetchError, Resource
from .compliance.ratelimit import OriginScheduler
from .compliance.ssrf import SafeResolver, is_public_ip, validate_url_literal


BeforeRequest = Callable[[str], Awaitable[None]]


class AioHttpTransport:
    def __init__(
        self,
        user_agent: str,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 15.0,
    ) -> None:
        self.user_agent = user_agent
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.request_timeout,
                connect=self.connect_timeout,
                sock_connect=self.connect_timeout,
            )
            connector = aiohttp.TCPConnector(
                resolver=SafeResolver(),
                limit=0,
                ttl_dns_cache=60,
                use_dns_cache=True,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                auto_decompress=True,
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return float(stripped)
    try:
        moment = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())


class HttpWorld:
    """A World backed by shared aiohttp transport.

    The legacy positional constructor remains supported, while production assembly
    injects one shared transport and origin scheduler for pages, media, and robots.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        request_timeout: float = 15.0,
        max_bytes: int = 2_000_000,
        *,
        transport: AioHttpTransport | None = None,
        scheduler: OriginScheduler | None = None,
        before_request: BeforeRequest | None = None,
        max_redirects: int = 10,
    ) -> None:
        self.transport = transport or AioHttpTransport(
            user_agent or "ImageFlasherBot/2.0",
            request_timeout=request_timeout,
        )
        self.scheduler = scheduler
        self.before_request = before_request
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    async def _read(self, response: aiohttp.ClientResponse) -> bytes:
        declared = response.content_length
        if declared is not None and declared > self.max_bytes:
            raise FetchError("Response exceeds byte limit")
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.content.iter_chunked(16_384):
            total += len(chunk)
            if total > self.max_bytes:
                raise FetchError("Response exceeds byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _validate_peer(response: aiohttp.ClientResponse) -> None:
        connection = response.connection
        transport = connection.transport if connection is not None else None
        peer = transport.get_extra_info("peername") if transport is not None else None
        if peer and not is_public_ip(str(peer[0])):
            raise FetchError("Connected peer resolved to a non-public address")

    async def fetch(self, url: str) -> Resource:
        current = canonicalize_url(url)
        if not current:
            raise FetchError("Invalid HTTP URL")
        session = await self.transport.session()
        for redirect_count in range(self.max_redirects + 1):
            validate_url_literal(current)
            if self.before_request is not None:
                await self.before_request(current)
            slot = self.scheduler.slot(current) if self.scheduler is not None else _null_slot()
            try:
                async with slot:
                    async with session.get(current, allow_redirects=False) as response:
                        self._validate_peer(response)
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location:
                                raise FetchError("Redirect response has no Location", status=response.status)
                            if redirect_count >= self.max_redirects:
                                raise FetchError("Too many redirects")
                            redirected = canonicalize_url(urljoin(current, location))
                            if not redirected:
                                raise FetchError("Redirect target is invalid")
                            current = redirected
                            continue
                        if response.status >= 400:
                            raise FetchError(
                                f"HTTP {response.status}",
                                status=response.status,
                                retry_after=_retry_after(response.headers.get("Retry-After")),
                            )
                        body = await self._read(response)
                        headers = {key.lower(): value for key, value in response.headers.items()}
                        content_type = response.content_type.lower() if response.content_type else ""
                        if self.scheduler is not None:
                            self.scheduler.record_success(current)
                        return Resource(
                            final_url=str(response.url),
                            status=response.status,
                            content_type=content_type,
                            body=body,
                            headers=headers,
                        )
            except FetchError as error:
                if self.scheduler is not None:
                    if error.status is None or error.status == 429 or error.status >= 500:
                        self.scheduler.record_failure(current)
                    else:
                        # A 4xx is a completed response, not an unhealthy origin.
                        self.scheduler.record_success(current)
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                if self.scheduler is not None:
                    self.scheduler.record_failure(current)
                raise FetchError(str(error)) from error
        raise FetchError("Too many redirects")

    async def close(self) -> None:
        await self.transport.close()


class _null_slot:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False
