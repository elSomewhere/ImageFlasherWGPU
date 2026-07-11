"""Layer 3 — image decode/normalize.

Turns raw image bytes into a fixed-size PNG Artifact payload for the renderer.
One media decoder among future ones (video/audio/text); the core never imports it.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class ProcessedImage:
    data: bytes
    width: int
    height: int
    ahash: int = 0  # 64-bit average-hash perceptual fingerprint (for novelty/dedup)


class ImageValidationError(Exception):
    pass


def average_hash(img: "Image.Image") -> int:
    small = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    average = sum(pixels) / len(pixels)
    fingerprint = 0
    for index, pixel in enumerate(pixels):
        if pixel >= average:
            fingerprint |= 1 << index
    return fingerprint


def normalize_image(
    image_bytes: bytes,
    *,
    size: int = 512,
    min_width: int = 64,
    min_height: int = 64,
) -> ProcessedImage:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            width, height = img.size
            if width < min_width or height < min_height:
                raise ImageValidationError("Image is below minimum dimensions")

            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            fingerprint = average_hash(img)
            canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
            x = (size - img.width) // 2
            y = (size - img.height) // 2
            canvas.paste(img, (x, y))

            output = io.BytesIO()
            canvas.save(output, format="PNG")
            return ProcessedImage(data=output.getvalue(), width=width, height=height, ahash=fingerprint)
    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError(f"Invalid image: {error}") from error
