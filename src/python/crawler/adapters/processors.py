"""Built-in artifact processors."""
from __future__ import annotations

import asyncio

from .image_pipeline import normalize_image
from ..core.types import Artifact, MediaCandidate
from ..ports.world import Resource


class ImageProcessor:
    kind = "image"

    def __init__(
        self,
        *,
        size: int = 384,
        min_width: int = 64,
        min_height: int = 64,
        max_pixels: int = 40_000_000,
        producer: str = "web_crawler",
    ) -> None:
        self.size = size
        self.min_width = min_width
        self.min_height = min_height
        self.max_pixels = max_pixels
        self.producer = producer

    async def process(
        self,
        candidate: MediaCandidate,
        resource: Resource,
        *,
        score: float,
    ) -> Artifact:
        processed = await asyncio.to_thread(
            normalize_image,
            resource.body,
            size=self.size,
            min_width=self.min_width,
            min_height=self.min_height,
            max_pixels=self.max_pixels,
        )
        return Artifact(
            kind="image",
            payload=processed.data,
            width=self.size,
            height=self.size,
            source_url=resource.final_url,
            page_url=candidate.page_url,
            score=score,
            mime="image/png",
            producer=self.producer,
            rights=candidate.rights,
            metadata={
                "source_width": processed.width,
                "source_height": processed.height,
                "dhash": processed.dhash,
                "ahash": processed.ahash,
                "color_histogram": processed.color_histogram,
            },
        )
