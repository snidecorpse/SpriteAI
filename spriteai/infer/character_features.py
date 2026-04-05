"""Extract compact likeness cues from a reference photo for character prompts."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CharacterFeatureProfile:
    glasses_likelihood: float
    has_glasses: bool
    skin_tone: str
    hair_tone: str
    clothing_color: str
    prompt_fragments: List[str]
    summary: str


def _crop_fraction(
    arr: np.ndarray,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
) -> np.ndarray:
    h, w = arr.shape[:2]
    yy0 = int(max(0, min(h - 1, round(y0 * h))))
    yy1 = int(max(yy0 + 1, min(h, round(y1 * h))))
    xx0 = int(max(0, min(w - 1, round(x0 * w))))
    xx1 = int(max(xx0 + 1, min(w, round(x1 * w))))
    return arr[yy0:yy1, xx0:xx1]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    return (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((cmax - cmin) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )


def _dominant_rgb(region: np.ndarray) -> Tuple[int, int, int]:
    if region.size == 0:
        return (128, 128, 128)
    flat = region.reshape(-1, 3)
    bins = (flat // 32).astype(np.int32)
    keys = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    best = values[int(np.argmax(counts))]
    mask = keys == best
    chosen = flat[mask]
    mean_rgb = chosen.mean(axis=0) if chosen.size else flat.mean(axis=0)
    return (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))


def _skin_tone_label(rgb: Tuple[int, int, int]) -> str:
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if lum < 85:
        return "deep skin tone"
    if lum < 125:
        return "medium skin tone"
    if lum < 170:
        return "tan skin tone"
    return "fair skin tone"


def _hair_tone_label(rgb: Tuple[int, int, int]) -> str:
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if lum < 40:
        return "black hair"
    if lum < 80:
        return "dark brown hair"
    if lum < 120:
        return "brown hair"
    if lum < 165:
        return "light brown hair"
    return "blond hair"


def _color_label(rgb: Tuple[int, int, int]) -> str:
    r, g, b = (c / 255.0 for c in rgb)
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = hue * 360.0
    if sat < 0.12:
        if val < 0.2:
            return "black top"
        if val < 0.45:
            return "gray top"
        if val < 0.78:
            return "light gray top"
        return "white top"
    if hue_deg < 18 or hue_deg >= 345:
        return "red top"
    if hue_deg < 42:
        return "orange top"
    if hue_deg < 70:
        return "yellow top"
    if hue_deg < 100:
        return "green top"
    if hue_deg < 140:
        return "emerald top"
    if hue_deg < 180:
        return "mint top"
    if hue_deg < 220:
        return "blue top"
    if hue_deg < 268:
        return "indigo top"
    if hue_deg < 315:
        return "purple top"
    return "pink top"


def _glasses_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0

    gray = _luma(region.astype(np.float32))
    dark_ratio = float((gray < 58.0).mean())

    h, w = gray.shape
    left = gray[:, : max(1, int(w * 0.45))]
    right = gray[:, int(w * 0.55) :]
    left_dark = float((left < 58.0).mean())
    right_dark = float((right < 58.0).mean())
    symmetry = max(0.0, 1.0 - abs(left_dark - right_dark) * 3.0)

    bridge = gray[:, int(w * 0.45) : max(int(w * 0.55), int(w * 0.45) + 1)]
    bridge_ratio = float((bridge < 70.0).mean()) if bridge.size else 0.0

    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if w > 1 else 0.0
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if h > 1 else 0.0
    edge_score = min(1.0, (gx + gy) / 110.0)

    score = dark_ratio * 1.8 + bridge_ratio * 2.0 + symmetry * 0.8 + edge_score * 0.9
    return max(0.0, min(1.0, score / 2.7))


def extract_character_features(image: Image.Image) -> CharacterFeatureProfile:
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        raise ValueError("Input image is empty.")

    face_region = _crop_fraction(arr, 0.12, 0.62, 0.18, 0.82)
    skin_mask = _skin_mask(face_region)
    if int(skin_mask.sum()) >= 80:
        skin_rgb = tuple(int(x) for x in face_region[skin_mask].mean(axis=0))
    else:
        skin_rgb = _dominant_rgb(_crop_fraction(arr, 0.22, 0.68, 0.28, 0.72))
    skin_tone = _skin_tone_label(skin_rgb)

    skin_lum = float(0.299 * skin_rgb[0] + 0.587 * skin_rgb[1] + 0.114 * skin_rgb[2])
    hair_region = _crop_fraction(arr, 0.03, 0.40, 0.18, 0.82)
    hair_skin_mask = _skin_mask(hair_region)
    hair_lum = _luma(hair_region.astype(np.float32))
    dark_hair_mask = (~hair_skin_mask) & (hair_lum < (skin_lum - 10.0))
    hair_candidates = hair_region[dark_hair_mask]
    if hair_candidates.size == 0:
        hair_candidates = hair_region[~hair_skin_mask]
    if hair_candidates.size == 0:
        hair_candidates = hair_region.reshape(-1, 3)
    hair_rgb = _dominant_rgb(hair_candidates.reshape(-1, 1, 3))
    hair_tone = _hair_tone_label(hair_rgb)

    cloth_region = _crop_fraction(arr, 0.50, 0.98, 0.30, 0.70)
    cloth_rgb = _dominant_rgb(cloth_region)
    clothing_color = _color_label(cloth_rgb)

    eye_region = _crop_fraction(arr, 0.32, 0.46, 0.18, 0.82)
    glasses_likelihood = float(_glasses_score(eye_region))
    has_glasses = bool(glasses_likelihood >= 0.55)

    fragments: List[str] = [skin_tone, hair_tone, clothing_color]
    if has_glasses:
        fragments.insert(0, "wearing glasses")

    summary = (
        f"glasses={'yes' if has_glasses else 'no'}({glasses_likelihood:.2f}); "
        f"skin={skin_tone}; hair={hair_tone}; clothing={clothing_color}"
    )
    return CharacterFeatureProfile(
        glasses_likelihood=glasses_likelihood,
        has_glasses=has_glasses,
        skin_tone=skin_tone,
        hair_tone=hair_tone,
        clothing_color=clothing_color,
        prompt_fragments=fragments,
        summary=summary,
    )
