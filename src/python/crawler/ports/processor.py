"""Media processor port and registry."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Artifact, MediaCandidate
from .world import Resource


@runtime_checkable
class ArtifactProcessor(Protocol):
    kind: str

    async def process(
        self,
        candidate: MediaCandidate,
        resource: Resource,
        *,
        score: float,
    ) -> Artifact:
        ...


class ProcessorRegistry:
    def __init__(self, processors: list[ArtifactProcessor] | None = None) -> None:
        self._processors = {processor.kind: processor for processor in processors or []}

    def register(self, processor: ArtifactProcessor) -> None:
        self._processors[processor.kind] = processor

    def get(self, kind: str) -> ArtifactProcessor:
        try:
            return self._processors[kind]
        except KeyError as error:
            raise ValueError(f"No artifact processor registered for kind: {kind}") from error
