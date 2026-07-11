"""Layer 2 — the ArtifactSink port.

A sink is where finished Artifacts go: the WebSocket stream to the renderer today,
a file/dev sink for tests, a multiplexer later. The engine only sees this port.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Artifact


@runtime_checkable
class ArtifactSink(Protocol):
    async def emit(self, artifact: Artifact) -> None:
        ...
