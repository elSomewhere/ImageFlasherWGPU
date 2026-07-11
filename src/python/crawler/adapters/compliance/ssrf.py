"""Compliance layer — SSRF guard.

Wraps any World: refuses URLs (and post-redirect final URLs) that resolve to
private / loopback / link-local / reserved addresses. Drop this wrapper and the
core keeps working — that is the whole point of the layering.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from ...ports.world import FetchError, Resource, World


class SsrfGuard:
    def __init__(self, inner: World) -> None:
        self.inner = inner

    def _validate(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise FetchError(f"Unsupported URL scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise FetchError("URL is missing a hostname")
        try:
            addresses = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as error:
            raise FetchError(f"Could not resolve hostname: {parsed.hostname}") from error
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise FetchError(f"Blocked non-public address for {parsed.hostname}")
        return parsed.hostname

    async def fetch(self, url: str) -> Resource:
        await asyncio.to_thread(self._validate, url)
        resource = await self.inner.fetch(url)
        if resource.final_url != url:
            await asyncio.to_thread(self._validate, resource.final_url)
        return resource
