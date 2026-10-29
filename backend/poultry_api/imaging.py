"""Validate an uploaded image and turn it into the 224 x 224 uint8 image the model sees.

Checks, in order: the file type from its magic bytes (the client's file name and content type are not
trusted), the pixel count from the header (before any pixel data is decoded, so a small file that
expands to gigabytes is rejected), then a full decode. JPEG and PNG go through the training decoder
(``tf.io.decode_image``, no EXIF rotation). HEIC, which training never saw, is decoded with libheif via
pillow-heif and then follows the same resize.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from poultry_api.errors import ApiError

ImageFormat = Literal["jpeg", "png", "heic"]
HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover - the api extra installs it
    HEIC_SUPPORTED = False


@dataclass(frozen=True)
class DecodedImage:
    format: ImageFormat
    width: int
    height: int
    #: (224, 224, 3) uint8 RGB, the model's view
    seen: np.ndarray


def sniff(data: bytes) -> ImageFormat | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[4:8] == b"ftyp" and data[8:12] in HEIF_BRANDS:
        return "heic"
    return None


def accepted_formats() -> list[str]:
    return ["jpeg", "png", *(["heic"] if HEIC_SUPPORTED else [])]


def _too_large(width: int, height: int, max_pixels: int) -> ApiError:
    return ApiError(
        413,
        "image_too_large",
        f"The image is {width} x {height} pixels; the limit is {max_pixels / 1e6:.0f} megapixels.",
    )


def _header_size(data: bytes, max_pixels: int) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as im:  # reads the header only
            return im.size
    except Image.DecompressionBombError as exc:  # Pillow's own limit, far above ours
        raise ApiError(413, "image_too_large", f"The image is over {max_pixels / 1e6:.0f} megapixels.") from exc
    except Exception as exc:
        raise ApiError(400, "invalid_image", "The file could not be read as an image.") from exc


def load_image(data: bytes, max_pixels: int) -> DecodedImage:
    from poultry_disease.preprocess import decode_image, resize_uint8

    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    fmt = sniff(data)
    if fmt is None or (fmt == "heic" and not HEIC_SUPPORTED):
        names = [f.upper() for f in accepted_formats()]
        listed = ", ".join(names[:-1]) + " or " + names[-1]
        raise ApiError(415, "unsupported_media_type", f"Upload a {listed} image.")

    width, height = _header_size(data, max_pixels)
    if width * height > max_pixels:
        raise _too_large(width, height, max_pixels)

    try:
        if fmt == "heic":
            with Image.open(io.BytesIO(data)) as im:
                rgb = np.asarray(im.convert("RGB"))
        else:
            rgb = decode_image(data)
    except Exception as exc:
        raise ApiError(400, "invalid_image", "The image data is corrupt or truncated.") from exc
    if rgb.ndim != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ApiError(400, "invalid_image", "The image has no pixels.")
    return DecodedImage(format=fmt, width=width, height=height, seen=resize_uint8(rgb))
