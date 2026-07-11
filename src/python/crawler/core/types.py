"""Layer 0 — pure domain data. No I/O, no network, no compliance concepts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Artifact:
    """A piece of media harvested from the world, ready for the renderer.

    ``kind`` lets the same pipeline carry images today and text/video/audio later.
    ``payload`` is the normalized bytes (a PNG for images) the sink streams out.
    """

    kind: str
    payload: bytes
    width: int = 0
    height: int = 0
    source_url: str = ""
    page_url: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaCandidate:
    """A discovered media reference (an image URL, etc.) not yet fetched."""

    url: str
    page_url: str
    kind: str = "image"
    alt: str = ""
    context: str = ""
    score: float = 0.0


@dataclass
class Link:
    """A discovered outbound link, a candidate edge in the crawl graph."""

    url: str
    context: str = ""
    score: float = 0.0


@dataclass
class SeedNode:
    """A fresh entry point into the world, produced by a SeedSource."""

    url: str
    context: str = ""
    source: str = "seed"
