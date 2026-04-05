"""Photo preprocessing for character-first sprite generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CharacterPreprocessResult:
    image: Image.Image
    warnings: List[str]
    crop_mode: str


def _square_box_from_center(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    side: int,
) -> Tuple[int, int, int, int]:
    side = max(1, min(side, min(width, height)))
    x0 = int(round(center_x - side / 2))
    y0 = int(round(center_y - side / 2))
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    return x0, y0, x0 + side, y0 + side


def _center_square_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return image.crop((x0, y0, x0 + side, y0 + side))


def _portrait_focus_crop(image: Image.Image, ratio: float = 0.78, y_bias: float = -0.08) -> Image.Image:
    w, h = image.size
    side = int(min(w, h) * ratio)
    side = max(1, min(side, min(w, h)))
    cx = w // 2
    cy = int(h // 2 + y_bias * side)
    x0, y0, x1, y1 = _square_box_from_center(w, h, cx, cy, side)
    return image.crop((x0, y0, x1, y1))


def _detect_face_crop(image: Image.Image) -> Optional[Tuple[Image.Image, str]]:
    try:
        import cv2
    except Exception:
        return None

    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    if arr.size == 0:
        return None

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade_path = str(cv2.data.haarcascades) + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        return None

    h, w = gray.shape[:2]
    min_side = max(24, min(w, h) // 12)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    if faces is None or len(faces) == 0:
        return None

    x, y, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
    cx = float(x + fw / 2.0)
    cy = float(y + fh * 0.7)
    side = int(max(fw, fh) * 3.8)
    x0, y0, x1, y1 = _square_box_from_center(w, h, cx, cy, side)
    cropped = rgb.crop((x0, y0, x1, y1))
    return cropped, "face-detect"


def preprocess_character_reference_image(image: Image.Image, target_size: int = 384) -> CharacterPreprocessResult:
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")

    warnings: List[str] = []
    source = image.convert("RGBA")
    src_w, src_h = source.size
    aspect = max(src_w, src_h) / max(1, min(src_w, src_h))

    detected = _detect_face_crop(source)
    if detected is not None:
        cropped, mode = detected
        resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        warnings.append("Applied face-detection crop around head/torso region.")
        return CharacterPreprocessResult(image=resized.convert("RGBA"), warnings=warnings, crop_mode=mode)

    working = source
    if aspect > 1.1:
        working = _center_square_crop(working)
        warnings.append("Applied center-square crop to reduce wide background context.")
    working = _portrait_focus_crop(working, ratio=0.78, y_bias=-0.08)
    warnings.append("Face detection unavailable; used heuristic portrait crop.")
    resized = working.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return CharacterPreprocessResult(image=resized, warnings=warnings, crop_mode="heuristic")
