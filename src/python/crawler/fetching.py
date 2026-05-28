from __future__ import annotations

import ipaddress
import socket
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlparse

from .config import CrawlerConfig


@dataclass
class FetchResult:
    url: str
    content: bytes
    content_type: str


class FetchError(Exception):
    pass


class RateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._last_request: dict[str, float] = {}
        self._lock = Lock()

    def wait(self, hostname: str) -> None:
        with self._lock:
            now = time.time()
            last = self._last_request.get(hostname, 0.0)
            wait_for = self.delay_seconds - (now - last)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_request[hostname] = time.time()


class SafeFetcher:
    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.page_delay_seconds)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_lock = Lock()

    def validate_public_url(self, url: str) -> str:
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

    def allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        with self._robots_lock:
            if robots_url not in self._robots:
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                try:
                    parser.read()
                except Exception:
                    self._robots[robots_url] = None
                else:
                    self._robots[robots_url] = parser
            parser = self._robots[robots_url]

        if parser is None:
            return True
        return parser.can_fetch(self.config.user_agent, url)

    def fetch_page(self, url: str) -> FetchResult:
        hostname = self.validate_public_url(url)
        if not self.allowed_by_robots(url):
            raise FetchError("Disallowed by robots.txt")
        self.rate_limiter.wait(hostname)

        response_url, content_type, content = self._fetch(
            url,
            self.config.max_page_bytes,
            hostname,
            final_url_allowed=self.allowed_by_robots,
        )
        if content_type and "html" not in content_type and "xml" not in content_type and "text/plain" not in content_type:
            raise FetchError(f"Not an HTML page: {content_type}")
        return FetchResult(url=response_url, content=content, content_type=content_type)

    def fetch_image(self, url: str) -> FetchResult:
        hostname = self.validate_public_url(url)
        self.rate_limiter.wait(hostname)

        response_url, content_type, content = self._fetch(url, self.config.max_image_bytes, hostname)
        if not content_type.startswith("image/"):
            raise FetchError(f"Not an image: {content_type or 'unknown'}")
        return FetchResult(url=response_url, content=content, content_type=content_type)

    def _fetch(self, url: str, byte_limit: int, original_hostname: str, final_url_allowed=None) -> tuple[str, str, bytes]:
        request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
        chunks: list[bytes] = []
        total = 0
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                final_url = response.geturl()
                final_hostname = self.validate_public_url(final_url)
                if final_hostname != original_hostname:
                    self.rate_limiter.wait(final_hostname)
                if final_url_allowed and not final_url_allowed(final_url):
                    raise FetchError("Final URL disallowed by robots.txt")
                content_type = response.headers.get_content_type().lower()
                while True:
                    chunk = response.read(16_384)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > byte_limit:
                        raise FetchError("Response exceeds byte limit")
                    chunks.append(chunk)
                return final_url, content_type, b"".join(chunks)
        except urllib.error.HTTPError as error:
            raise FetchError(f"HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise FetchError(str(error.reason)) from error

