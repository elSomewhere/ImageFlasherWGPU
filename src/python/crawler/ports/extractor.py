"""Layer 2 — the Extractor port.

An Extractor turns a fetched Resource into the graph edges and media it contains.
HTML is one implementation; a JSON/AI-web extractor can be another. Extractors own
content-type validation (an HTML extractor rejects non-HTML) so the World stays a
dumb transport.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Link, MediaCandidate
from .world import Resource


@runtime_checkable
class Extractor(Protocol):
    def extract(self, resource: Resource) -> tuple[list[MediaCandidate], list[Link], str]:
        """Return (media candidates, links, title). May raise FetchError if the
        resource is not the kind this extractor understands."""
        ...
