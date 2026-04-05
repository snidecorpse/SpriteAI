"""Build canonical character-only dataset from multi-view sprite folders."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_VIEW_FOLDER_MAP: Dict[str, str] = {
    "2": "front",
    "1": "right",
    "3": "left",
    "0": "back",
}
VIEW_ORDER = ("front", "right", "left", "back")


@dataclass(frozen=True)
class CharacterRecord:
    image_path: str
    sample_id: str
    view: str
    identity_group: str
    source_image: str
    prompt: str


def _natural_sort_key(text: str):
    stem = text.strip()
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _iter_images(directory: Path) -> Iterable[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    for path in sorted(files, key=lambda p: _natural_sort_key(p.stem)):
        yield path


def _write_jsonl(path: Path, rows: List[CharacterRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")


def _identity_split(identity_ids: List[str], val_ratio: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    items = identity_ids[:]
    rng.shuffle(items)
    val_count = max(1, int(len(items) * val_ratio))
    return set(items[:val_count])


def _build_prompt(view: str) -> str:
    return (
        f"<sprite_char>, <view_{view}>, "
        "pixel character sprite, full body, simple background"
    )


def build_character_dataset(
    input_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    view_folder_map: Dict[str, str] | None = None,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("--val_ratio must be between 0 and 1 (exclusive).")

    view_folder_map = view_folder_map or DEFAULT_VIEW_FOLDER_MAP
    folder_view_map: Dict[Path, str] = {}
    for folder_name, view in view_folder_map.items():
        folder_path = input_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Expected view folder not found: {folder_path}")
        if view not in VIEW_ORDER:
            raise ValueError(f"Unsupported view '{view}'. Expected one of {VIEW_ORDER}.")
        folder_view_map[folder_path] = view

    view_to_paths: Dict[str, Dict[str, Path]] = {view: {} for view in VIEW_ORDER}
    for folder_path, view in folder_view_map.items():
        for image_path in _iter_images(folder_path):
            identity_group = image_path.stem
            if identity_group in view_to_paths[view]:
                raise RuntimeError(
                    f"Duplicate identity '{identity_group}' for view '{view}' in {folder_path}"
                )
            view_to_paths[view][identity_group] = image_path

    identity_sets = [set(view_to_paths[v].keys()) for v in VIEW_ORDER]
    common_identities = set.intersection(*identity_sets) if identity_sets else set()
    if not common_identities:
        raise RuntimeError("No shared identities found across all required view folders.")

    all_identities = set.union(*identity_sets)
    dropped = len(all_identities) - len(common_identities)
    sorted_identities = sorted(common_identities, key=_natural_sort_key)
    val_identities = _identity_split(sorted_identities, val_ratio=val_ratio, seed=seed)

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    all_rows: List[CharacterRecord] = []
    sample_counter = 1
    for identity_group in sorted_identities:
        for view in VIEW_ORDER:
            src = view_to_paths[view][identity_group]
            with Image.open(src) as img:
                rgba = img.convert("RGBA")

            sample_id = f"{sample_counter:06d}"
            sample_name = f"{sample_id}.png"
            out_path = out_images / sample_name
            rgba.save(out_path, format="PNG")

            row = CharacterRecord(
                image_path=str(Path("images") / sample_name),
                sample_id=sample_id,
                view=view,
                identity_group=identity_group,
                source_image=str(src),
                prompt=_build_prompt(view),
            )
            all_rows.append(row)
            sample_counter += 1

    train_rows = [r for r in all_rows if r.identity_group not in val_identities]
    val_rows = [r for r in all_rows if r.identity_group in val_identities]
    _write_jsonl(out_dir / "train_metadata.jsonl", train_rows)
    _write_jsonl(out_dir / "val_metadata.jsonl", val_rows)
    _write_jsonl(out_dir / "all_metadata.jsonl", all_rows)

    print(
        json.dumps(
            {
                "mode": "character_dataset_v2",
                "input_dir": str(input_dir),
                "view_folder_map": view_folder_map,
                "identities": len(sorted_identities),
                "dropped_incomplete_identities": dropped,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical character dataset from multi-view folders.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing folders 0/1/2/3.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split by identity group.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/val split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_character_dataset(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
