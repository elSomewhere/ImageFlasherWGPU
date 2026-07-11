"""Layer 3 — HTML extractor.

Turns an HTML Resource into image candidates + links + title. Owns the "is this
actually HTML?" content-type check, keeping that concern out of the transport layer.
"""
from __future__ import annotations

from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urldefrag, urljoin

from ...core.types import Link, MediaCandidate
from ...ports.world import FetchError, Resource


IMAGE_ATTRS = ("src", "data-src", "data-original", "data-lazy-src", "data-url")


def normalize_url(url: str, base_url: str = "") -> str:
    if not url:
        return ""
    absolute = urljoin(base_url, url.strip())
    normalized, _fragment = urldefrag(absolute)
    return normalized


def parse_srcset(srcset: str) -> Iterable[str]:
    for candidate in srcset.split(","):
        parts = candidate.strip().split()
        if parts:
            yield parts[0]


class _DiscoveryParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.images: list[MediaCandidate] = []
        self.links: list[Link] = []
        self.title_parts: list[str] = []
        self.meta_robots = ""
        self._in_title = False
        self._current_anchor: dict[str, str] | None = None

    @property
    def title(self) -> str:
        return " ".join(part.strip() for part in self.title_parts if part.strip())[:300]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()

        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            property_name = (attr_map.get("property") or attr_map.get("name") or "").lower()
            if property_name == "robots":
                self.meta_robots = (attr_map.get("content", "") or "").lower()
            if property_name in {"og:image", "twitter:image", "twitter:image:src"}:
                url = normalize_url(attr_map.get("content", ""), self.page_url)
                if url:
                    self.images.append(MediaCandidate(url=url, page_url=self.page_url, context=self.title))
            return

        if tag in {"img", "source"}:
            urls: list[str] = []
            for attr in IMAGE_ATTRS:
                value = attr_map.get(attr)
                if value:
                    urls.append(value)
            srcset = attr_map.get("srcset") or attr_map.get("data-srcset")
            if srcset:
                urls.extend(parse_srcset(srcset))

            context = " ".join(
                value
                for value in (
                    self.title,
                    attr_map.get("alt", ""),
                    attr_map.get("title", ""),
                    attr_map.get("aria-label", ""),
                )
                if value
            )
            for raw_url in urls:
                url = normalize_url(raw_url, self.page_url)
                if url:
                    self.images.append(
                        MediaCandidate(
                            url=url,
                            page_url=self.page_url,
                            alt=attr_map.get("alt", ""),
                            context=context,
                        )
                    )
            return

        if tag == "a" and attr_map.get("href"):
            url = normalize_url(attr_map["href"], self.page_url)
            self._current_anchor = {"url": url, "text": ""}

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._current_anchor is not None:
            self._current_anchor["text"] += data[:300]

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._current_anchor:
            url = self._current_anchor["url"]
            if url:
                self.links.append(
                    Link(
                        url=url,
                        context=f"{self.title} {self._current_anchor['text'].strip()[:300]}",
                    )
                )
            self._current_anchor = None


def _parse(html: str, page_url: str) -> _DiscoveryParser:
    parser = _DiscoveryParser(page_url)
    parser.feed(html)
    parser.close()
    return parser


def discover(html: str, page_url: str) -> tuple[list[MediaCandidate], list[Link], str]:
    parser = _parse(html, page_url)
    return parser.images, parser.links, parser.title


class HtmlExtractor:
    """Extractor port implementation for HTML pages.

    Honors ``noindex``/``nofollow`` from both the ``<meta name="robots">`` tag and the
    ``X-Robots-Tag`` response header: ``noindex`` drops harvested media, ``nofollow``
    drops outbound links. This keeps the crawler polite to pages that ask to be left
    out, without needing per-site configuration.
    """

    def extract(self, resource: Resource) -> tuple[list[MediaCandidate], list[Link], str]:
        content_type = resource.content_type
        if content_type and not (
            "html" in content_type or "xml" in content_type or "text/plain" in content_type
        ):
            raise FetchError(f"Not an HTML page: {content_type}")
        html = resource.body.decode("utf-8", errors="replace")
        parser = _parse(html, resource.final_url)

        directives = f"{parser.meta_robots} {resource.headers.get('x-robots-tag', '')}".lower()
        images = [] if "noindex" in directives else parser.images
        links = [] if ("nofollow" in directives or "noindex" in directives) else parser.links
        return images, links, parser.title
