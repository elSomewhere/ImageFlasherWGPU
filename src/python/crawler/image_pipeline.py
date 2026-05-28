import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CrawlerConfig


@dataclass
class ProcessedImage:
    data: bytes
    width: int
    height: int


class ImageValidationError(Exception):
    pass


def normalize_image(image_bytes: bytes, config: CrawlerConfig) -> ProcessedImage:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            width, height = img.size
            if width < config.min_image_width or height < config.min_image_height:
                raise ImageValidationError("Image is below minimum dimensions")

            img.thumbnail((config.image_size, config.image_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (config.image_size, config.image_size), color=(0, 0, 0))
            x = (config.image_size - img.width) // 2
            y = (config.image_size - img.height) // 2
            canvas.paste(img, (x, y))

            output = io.BytesIO()
            canvas.save(output, format="PNG")
            return ProcessedImage(data=output.getvalue(), width=width, height=height)
    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError(f"Invalid image: {error}") from error

