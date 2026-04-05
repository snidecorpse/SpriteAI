"""Post-processing helpers to enforce one crisp centered sprite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .pixelize import quantize_to_palette


@dataclass(frozen=True)
class CandidateScore:
    image: Image.Image
    score: float
    component_count: int
    largest_component_ratio: float


def _border_median_color(arr: np.ndarray) -> np.ndarray:
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    edges = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(edges, axis=0)


def _foreground_mask(arr: np.ndarray, threshold: float = 34.0) -> np.ndarray:
    bg = _border_median_color(arr).reshape(1, 1, 3).astype(np.float32)
    dist = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
    mask = dist > threshold

    # Light denoise: keep pixels that have local support.
    h, w = mask.shape
    refined = np.zeros_like(mask)
    for y in range(h):
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        for x in range(w):
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            support = int(mask[y0:y1, x0:x1].sum())
            refined[y, x] = bool(mask[y, x] and support >= 3)
    return refined


def _connected_components(mask: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Tuple[int, Tuple[int, int, int, int]]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            y_min = y_max = y
            x_min = x_max = x
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cy < y_min:
                    y_min = cy
                if cy > y_max:
                    y_max = cy
                if cx < x_min:
                    x_min = cx
                if cx > x_max:
                    x_max = cx

                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            components.append((area, (x_min, y_min, x_max + 1, y_max + 1)))

    components.sort(key=lambda c: c[0], reverse=True)
    return components


def _center_single_component(arr: np.ndarray, target_size: int, fg_mask: np.ndarray) -> np.ndarray:
    components = _connected_components(fg_mask)
    if not components:
        return arr

    largest_area, (x0, y0, x1, y1) = components[0]
    if largest_area < max(8, int(target_size * target_size * 0.01)):
        return arr

    crop = arr[y0:y1, x0:x1, :]
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return arr

    # Slightly aggressive frame fill for clearer silhouette.
    fit = int(round(target_size * 0.82))
    scale = min(fit / max(1, w), fit / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    crop_img = Image.fromarray(crop.astype(np.uint8), mode="RGB")
    resized = crop_img.resize((new_w, new_h), Image.Resampling.NEAREST)
    resized_arr = np.asarray(resized, dtype=np.uint8)

    bg = _border_median_color(arr).astype(np.uint8)
    out = np.tile(bg.reshape(1, 1, 3), (target_size, target_size, 1))
    x_off = (target_size - new_w) // 2
    # Place sprite slightly lower to preserve head + feet readability.
    y_off = int(round((target_size - new_h) * 0.56))
    y_off = max(0, min(y_off, target_size - new_h))
    out[y_off : y_off + new_h, x_off : x_off + new_w, :] = resized_arr
    return out


def _edge_strength(arr: np.ndarray) -> float:
    gray = arr.astype(np.float32).mean(axis=2)
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    return float(min(1.0, (gx + gy) / 70.0))


def _score_candidate(arr: np.ndarray, fg_mask: np.ndarray) -> Tuple[float, int, float]:
    components = _connected_components(fg_mask)
    if not components:
        return -1.0, 0, 0.0
    total_fg = int(fg_mask.sum())
    largest_area = int(components[0][0])
    comp_count = len(components)
    largest_ratio = largest_area / max(1, total_fg)
    area_ratio = largest_area / max(1, arr.shape[0] * arr.shape[1])
    area_score = max(0.0, 1.0 - abs(area_ratio - 0.27) / 0.27)
    edge = _edge_strength(arr)
    duplicate_penalty = max(0, comp_count - 1) * 0.35
    score = largest_ratio * 1.35 + area_score * 1.05 + edge * 0.45 - duplicate_penalty
    return float(score), comp_count, float(largest_ratio)


def postprocess_character_candidate(image: Image.Image, sprite_size: int = 64) -> CandidateScore:
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    target_size = max(16, int(sprite_size))
    rgb = image.convert("RGB").resize((target_size, target_size), Image.Resampling.BOX)
    arr = np.asarray(rgb, dtype=np.uint8)
    fg_mask = _foreground_mask(arr)
    centered = _center_single_component(arr, target_size=target_size, fg_mask=fg_mask)
    centered_mask = _foreground_mask(centered)
    score, comp_count, largest_ratio = _score_candidate(centered, centered_mask)
    quantized = quantize_to_palette(Image.fromarray(centered, mode="RGB"))
    return CandidateScore(
        image=quantized,
        score=score,
        component_count=comp_count,
        largest_component_ratio=largest_ratio,
    )


def choose_best_character_candidate(images: List[Image.Image], sprite_size: int = 64) -> CandidateScore:
    if not images:
        raise ValueError("Expected at least one candidate image.")
    scored = [postprocess_character_candidate(img, sprite_size=sprite_size) for img in images]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[0]

