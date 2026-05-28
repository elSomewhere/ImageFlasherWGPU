from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urljoin, urldefrag


IMAGE_ATTRS = ("src", "data-src", "data-original", "data-lazy-src", "data-url")


@dataclass
class ImageCandidate:
    url: str
    page_url: str
    alt: str = ""
    context: str = ""
    score: float = 0.0


@dataclass
class LinkCandidate:
    url: str
    context: str = ""
    score: float = 0.0


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
        self.images: list[ImageCandidate] = []
        self.links: list[LinkCandidate] = []
        self.title_parts: list[str] = []
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
            if property_name in {"og:image", "twitter:image", "twitter:image:src"}:
                url = normalize_url(attr_map.get("content", ""), self.page_url)
                if url:
                    self.images.append(ImageCandidate(url=url, page_url=self.page_url, context=self.title))
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
                        ImageCandidate(
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
                    LinkCandidate(
                        url=url,
                        context=f"{self.title} {self._current_anchor['text'].strip()[:300]}",
                    )
                )
            self._current_anchor = None


def discover(html: str, page_url: str) -> tuple[list[ImageCandidate], list[LinkCandidate], str]:
    parser = _DiscoveryParser(page_url)
    parser.feed(html)
    parser.close()
    return parser.images, parser.links, parser.title

