"""Reference image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    warnings: List[str]


def _edge_color(image: np.ndarray) -> np.ndarray:
    top = image[0, :, :]
    bottom = image[-1, :, :]
    left = image[:, 0, :]
    right = image[:, -1, :]
    edges = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(edges, axis=0)


def _simple_background_alpha(image: Image.Image, threshold: float = 28.0) -> Image.Image:
    """Approximate background removal using edge-color distance."""
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    rgb = arr[:, :, :3]
    bg = _edge_color(rgb)
    dist = np.linalg.norm(rgb - bg, axis=2)
    alpha = np.where(dist < threshold, 0.0, 255.0)
    arr[:, :, 3] = np.maximum(arr[:, :, 3], alpha)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _bbox_from_alpha(image: Image.Image) -> Tuple[int, int, int, int]:
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    coords = np.argwhere(alpha > 12)
    if coords.size == 0:
        width, height = image.size
        return 0, 0, width, height
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return int(x0), int(y0), int(x1 + 1), int(y1 + 1)


def _pad_to_square(image: Image.Image, fill=(0, 0, 0, 0)) -> Image.Image:
    w, h = image.size
    side = max(w, h)
    out = Image.new("RGBA", (side, side), color=fill)
    x = (side - w) // 2
    y = (side - h) // 2
    out.paste(image, (x, y), image)
    return out


def _blur_score(image: Image.Image) -> float:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    # Variance of Laplacian approximation with ImageFilter kernel.
    lap = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    _ = arr  # Kept for readability if we later swap operators.
    return float(lap.var())


def assess_reference_quality(image: Image.Image) -> List[str]:
    warnings: List[str] = []
    width, height = image.size

    if min(width, height) < 128:
        warnings.append(
            "Reference image is low resolution; results may look noisy. "
            "Try an image with at least 512x512."
        )

    if _blur_score(image) < 85.0:
        warnings.append(
            "Reference image looks blurry; identity details may not transfer well."
        )

    rgba = image.convert("RGBA")
    alpha_ratio = np.asarray(rgba)[:, :, 3].mean() / 255.0
    if alpha_ratio < 0.15:
        warnings.append(
            "Subject may be too small in frame; use a tighter crop around the character."
        )
    return warnings


def preprocess_reference_image(image: Image.Image, target_size: int = 256) -> PreprocessResult:
    """Background remove, center crop, square pad, resize."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")

    source = image.convert("RGBA")
    fg = _simple_background_alpha(source)
    bbox = _bbox_from_alpha(fg)
    cropped = fg.crop(bbox)
    squared = _pad_to_square(cropped)
    resized = squared.resize((target_size, target_size), Image.Resampling.LANCZOS)
    warnings = assess_reference_quality(resized)
    return PreprocessResult(image=resized, warnings=warnings)
