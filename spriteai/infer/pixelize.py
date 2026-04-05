"""Pixel-art post-processing utilities."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from PIL import Image

Palette = Sequence[Tuple[int, int, int]]

DEFAULT_16_COLOR_PALETTE: Palette = (
    (20, 12, 28),
    (68, 36, 52),
    (48, 52, 109),
    (78, 74, 78),
    (133, 76, 48),
    (52, 101, 36),
    (208, 70, 72),
    (117, 113, 97),
    (89, 125, 206),
    (210, 125, 44),
    (133, 149, 161),
    (109, 170, 44),
    (210, 170, 153),
    (109, 194, 202),
    (218, 212, 94),
    (222, 238, 214),
)


def _flatten_palette(palette: Palette) -> List[int]:
    values: List[int] = []
    for r, g, b in palette:
        values.extend([int(r), int(g), int(b)])
    pad = values[-3:] if len(values) >= 3 else [0, 0, 0]
    while len(values) < 768:
        values.extend(pad)
    return values[:768]


def make_palette_image(palette: Palette = DEFAULT_16_COLOR_PALETTE) -> Image.Image:
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(_flatten_palette(palette))
    return pal_img


def nearest_neighbor_downscale(image: Image.Image, size: int = 32) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    return image.convert("RGBA").resize((size, size), Image.Resampling.NEAREST)


def quantize_to_palette(image: Image.Image, palette: Palette = DEFAULT_16_COLOR_PALETTE) -> Image.Image:
    pal_img = make_palette_image(palette)
    quantized = image.convert("RGB").quantize(palette=pal_img, dither=Image.Dither.NONE)
    return quantized.convert("RGBA")


def pixelize_image(
    image: Image.Image,
    size: int = 32,
    palette: Palette = DEFAULT_16_COLOR_PALETTE,
) -> Image.Image:
    down = nearest_neighbor_downscale(image, size=size)
    return quantize_to_palette(down, palette=palette)


def count_unique_colors(image: Image.Image) -> int:
    colors = image.convert("RGBA").getcolors(maxcolors=1 << 20)
    if colors is None:
        # Defensive fallback for pathological images.
        return len(set(image.convert("RGBA").getdata()))
    return len(colors)


def enforce_palette_cap(image: Image.Image, max_colors: int = 16) -> Image.Image:
    """Hard cap color count while retaining alpha channel."""
    out = image.convert("RGB").quantize(colors=max_colors, dither=Image.Dither.NONE)
    return out.convert("RGBA")
