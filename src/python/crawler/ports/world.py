"""Layer 2 — the World port.

A ``World`` answers one question: given a URL, hand me back a Resource. It knows
nothing about HTTP, robots.txt, rate limits, or SSRF — those are compliance layers
that *wrap* a World (see ``adapters/compliance``). The core engine only ever sees
this port, so the identical engine can later run against a generative/imagined web
by swapping the innermost adapter and dropping the wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable


class FetchError(Exception):
    """Raised by any World (or wrapping layer) when a URL cannot be delivered.

    Carries the HTTP status and any ``Retry-After`` hint when known, so a backoff
    layer can react to 429/503 without re-parsing messages.
    """

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class Resource:
    """An abstract fetched response. For a generative world, ``status`` is just 200."""

    final_url: str
    status: int
    content_type: str
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class World(Protocol):
    async def fetch(self, url: str) -> Resource:
        ...
