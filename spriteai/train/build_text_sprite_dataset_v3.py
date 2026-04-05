"""Build V3 text-caption sprite dataset from canonical character dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIEW_PHRASES: Dict[str, str] = {
    "front": "front-facing",
    "right": "right-facing",
    "left": "left-facing",
    "back": "back-facing",
}


@dataclass(frozen=True)
class TextSpriteRecord:
    image_path: str
    sample_id: str
    view: str
    identity_group: str
    prompt_v3: str
    attributes: Dict[str, object]


def _natural_sort_key(text: str) -> Tuple[int, object]:
    clean = text.strip()
    if clean.isdigit():
        return (0, int(clean))
    return (1, clean)


def _iter_images(path: Path) -> Iterable[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    for file in sorted(files, key=lambda p: _natural_sort_key(p.stem)):
        yield file


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[TextSpriteRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")


def _border_median(arr: np.ndarray) -> np.ndarray:
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    edge = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(edge, axis=0)


def _foreground_mask(arr: np.ndarray, threshold: float = 24.0) -> np.ndarray:
    bg = _border_median(arr).reshape(1, 1, 3).astype(np.float32)
    dist = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
    return dist > threshold


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        h, w = mask.shape
        return (0, 0, w, h)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (int(x0), int(y0), int(x1 + 1), int(y1 + 1))


def _crop_bbox_relative(
    arr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    rx0: float,
    ry0: float,
    rx1: float,
    ry1: float,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    ax0 = x0 + int(round(rx0 * w))
    ay0 = y0 + int(round(ry0 * h))
    ax1 = x0 + int(round(rx1 * w))
    ay1 = y0 + int(round(ry1 * h))
    ax0 = max(0, min(ax0, arr.shape[1] - 1))
    ay0 = max(0, min(ay0, arr.shape[0] - 1))
    ax1 = max(ax0 + 1, min(ax1, arr.shape[1]))
    ay1 = max(ay0 + 1, min(ay1, arr.shape[0]))
    return arr[ay0:ay1, ax0:ax1, :]


def _dominant_rgb(region: np.ndarray) -> Tuple[int, int, int]:
    if region.size == 0:
        return (128, 128, 128)
    flat = region.reshape(-1, 3)
    bins = (flat // 24).astype(np.int32)
    key = bins[:, 0] * 121 + bins[:, 1] * 11 + bins[:, 2]
    values, counts = np.unique(key, return_counts=True)
    best = values[int(np.argmax(counts))]
    chosen = flat[key == best]
    if chosen.size == 0:
        chosen = flat
    mean = chosen.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _skin_label(rgb: Tuple[int, int, int]) -> str:
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if lum < 85:
        return "deep skin tone"
    if lum < 125:
        return "medium skin tone"
    if lum < 170:
        return "tan skin tone"
    return "fair skin tone"


def _hair_label(rgb: Tuple[int, int, int]) -> str:
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if lum < 42:
        return "black hair"
    if lum < 82:
        return "dark brown hair"
    if lum < 120:
        return "brown hair"
    if lum < 165:
        return "light brown hair"
    return "blond hair"


def _top_label(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    sat = 0.0 if cmax == 0 else (cmax - cmin) / cmax
    if sat < 0.15:
        if cmax < 50:
            return "black top"
        if cmax < 115:
            return "gray top"
        if cmax < 190:
            return "light gray top"
        return "white top"
    if r >= g and r >= b:
        if g > b + 20:
            return "orange top"
        if b > g + 20:
            return "pink top"
        return "red top"
    if g >= r and g >= b:
        if b > r + 12:
            return "mint top"
        return "green top"
    if b >= r and b >= g:
        if r > g + 18:
            return "purple top"
        return "blue top"
    return "blue top"


def _extract_sprite_attributes(image: Image.Image, view: str) -> Dict[str, object]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = _foreground_mask(arr)
    bbox = _bbox_from_mask(mask)

    hair_region = _crop_bbox_relative(arr, bbox, 0.20, 0.00, 0.80, 0.33)
    skin_region = _crop_bbox_relative(arr, bbox, 0.28, 0.18, 0.72, 0.52)
    top_region = _crop_bbox_relative(arr, bbox, 0.22, 0.46, 0.78, 0.84)

    hair_rgb = _dominant_rgb(hair_region)
    skin_rgb = _dominant_rgb(skin_region)
    top_rgb = _dominant_rgb(top_region)

    hair_text = _hair_label(hair_rgb)
    skin_text = _skin_label(skin_rgb)
    top_text = _top_label(top_rgb)
    view_phrase = VIEW_PHRASES.get(view, "front-facing")

    prompt_v3 = (
        f"{view_phrase} pixel character sprite, {hair_text}, {skin_text}, {top_text}, "
        "full body, simple background, single character"
    )
    return {
        "prompt_v3": prompt_v3,
        "attributes": {
            "view_phrase": view_phrase,
            "hair": {"label": hair_text, "rgb": [int(c) for c in hair_rgb]},
            "skin": {"label": skin_text, "rgb": [int(c) for c in skin_rgb]},
            "top": {"label": top_text, "rgb": [int(c) for c in top_rgb]},
        },
    }


def build_text_sprite_dataset_v3(
    input_dir: Path,
    out_dir: Path,
    metadata_file: str = "all_metadata.jsonl",
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset dir does not exist: {input_dir}")
    src_meta = input_dir / metadata_file
    rows = _read_jsonl(src_meta)
    if not rows:
        raise RuntimeError(f"No rows found in metadata: {src_meta}")

    def sort_key(row: Dict[str, object]) -> Tuple[int, object]:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id.isdigit():
            return (0, int(sample_id))
        return (1, sample_id)

    sorted_rows = sorted(rows, key=sort_key)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    out_records: List[TextSpriteRecord] = []
    for idx, row in enumerate(sorted_rows, start=1):
        image_rel = Path(str(row["image_path"]))
        src_image = input_dir / image_rel
        if not src_image.exists():
            raise FileNotFoundError(f"Missing source image: {src_image}")

        with Image.open(src_image) as img:
            rgba = img.convert("RGBA")
            rgb = rgba.convert("RGB")
            extras = _extract_sprite_attributes(rgb, view=str(row.get("view", "front")))

        sample_id = f"{idx:06d}"
        out_name = f"{sample_id}.png"
        out_path = out_images / out_name
        rgba.save(out_path, format="PNG")

        out_records.append(
            TextSpriteRecord(
                image_path=str(Path("images") / out_name),
                sample_id=sample_id,
                view=str(row.get("view", "front")),
                identity_group=str(row.get("identity_group", "")),
                prompt_v3=str(extras["prompt_v3"]),
                attributes=dict(extras["attributes"]),
            )
        )

    _write_jsonl(out_dir / "all_metadata.jsonl", out_records)

    print(
        json.dumps(
            {
                "mode": "text_sprite_dataset_v3",
                "input_dir": str(input_dir),
                "input_metadata": metadata_file,
                "samples": len(out_records),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V3 text sprite dataset from canonical dataset.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Canonical dataset directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output V3 text dataset directory.")
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="all_metadata.jsonl",
        help="Metadata file in input_dir to process (default: all_metadata.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_text_sprite_dataset_v3(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        metadata_file=args.metadata_file,
    )


if __name__ == "__main__":
    main()

