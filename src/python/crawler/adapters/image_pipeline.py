"""Layer 3 — image decode/normalize.

Turns raw image bytes into a fixed-size PNG Artifact payload for the renderer.
One media decoder among future ones (video/audio/text); the core never imports it.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class ProcessedImage:
    data: bytes
    width: int
    height: int
    ahash: int = 0  # 64-bit average-hash perceptual fingerprint (for novelty/dedup)
    dhash: int = 0
    color_histogram: tuple[float, ...] = ()


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


def difference_hash(img: "Image.Image") -> int:
    small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    fingerprint = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            if pixels[row + x] > pixels[row + x + 1]:
                fingerprint |= 1 << (y * 8 + x)
    return fingerprint


def color_histogram(img: "Image.Image") -> tuple[float, ...]:
    hsv = img.convert("HSV").resize((32, 32), Image.Resampling.BILINEAR)
    bins = [0] * 16
    for hue, saturation, value in hsv.getdata():
        hue_bin = min(7, hue * 8 // 256)
        tone_bin = 1 if (saturation >= 48 and value >= 32) else 0
        bins[hue_bin * 2 + tone_bin] += 1
    total = float(sum(bins)) or 1.0
    return tuple(value / total for value in bins)


def normalize_image(
    image_bytes: bytes,
    *,
    size: int = 512,
    min_width: int = 64,
    min_height: int = 64,
    max_pixels: int = 40_000_000,
) -> ProcessedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                raise ImageValidationError("Image is below minimum dimensions")
            if width * height > max_pixels:
                raise ImageValidationError("Image exceeds maximum pixel count")
            # Enforce the pixel budget before conversion forces a full decode.
            img = ImageOps.exif_transpose(img).convert("RGB")

            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            fingerprint = average_hash(img)
            dhash = difference_hash(img)
            histogram = color_histogram(img)
            canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
            x = (size - img.width) // 2
            y = (size - img.height) // 2
            canvas.paste(img, (x, y))

            output = io.BytesIO()
            canvas.save(output, format="PNG")
            return ProcessedImage(
                data=output.getvalue(),
                width=width,
                height=height,
                ahash=fingerprint,
                dhash=dhash,
                color_histogram=histogram,
            )
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:
        raise ImageValidationError(f"Invalid image: {error}") from error
