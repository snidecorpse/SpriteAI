"""Build training dataset for SpriteAI from raw refs or labeled state folders."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from PIL import Image, ImageDraw

from spriteai.infer.preprocess import preprocess_reference_image
from spriteai.infer.state_prompts import STATE_INSTRUCTIONS, STATE_KEYS

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Record:
    image_path: str
    prompt: str
    state: str
    source_image: str
    identity_id: str


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def _has_images(directory: Path) -> bool:
    return any(path.is_file() and path.suffix.lower() in SUPPORTED_EXTS for path in directory.iterdir())


def _has_subdirs(directory: Path) -> bool:
    return any(path.is_dir() for path in directory.iterdir())


def _pad_to_square(image: Image.Image, fill=(0, 0, 0, 0)) -> Image.Image:
    w, h = image.size
    side = max(w, h)
    out = Image.new("RGBA", (side, side), color=fill)
    x = (side - w) // 2
    y = (side - h) // 2
    out.paste(image, (x, y), image)
    return out


def _make_base_style(image: Image.Image, resolution: int = 256) -> Image.Image:
    base = image.convert("RGBA").resize((resolution, resolution), Image.Resampling.LANCZOS)
    # Pixel-art look prior to state overlays: downscale then nearest upscale.
    small = base.resize((64, 64), Image.Resampling.NEAREST)
    return small.resize((resolution, resolution), Image.Resampling.NEAREST)


def _add_state_overlay(image: Image.Image, state: str, rng: random.Random) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if state == "eating":
        draw.rounded_rectangle((82, 178, 174, 210), radius=10, fill=(112, 78, 48), outline=(40, 30, 20), width=4)
        draw.ellipse((124, 156, 138, 170), fill=(245, 220, 140), outline=(130, 100, 60), width=2)
    elif state == "feeding":
        draw.rounded_rectangle((22, 140, 92, 168), radius=10, fill=(231, 190, 160), outline=(115, 85, 70), width=4)
        draw.rectangle((88, 149, 140, 154), fill=(180, 180, 180))
        draw.ellipse((138, 144, 154, 160), fill=(220, 220, 220), outline=(100, 100, 100), width=2)
    elif state == "sleeping":
        draw.line((98, 122, 116, 122), fill=(40, 40, 40), width=4)
        draw.line((140, 122, 158, 122), fill=(40, 40, 40), width=4)
        z_x = 172 + rng.randint(-4, 4)
        draw.text((z_x, 76), "Z", fill=(240, 240, 255))
    elif state == "hygiene":
        for _ in range(10):
            x = rng.randint(72, 190)
            y = rng.randint(86, 194)
            r = rng.randint(6, 13)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(190, 228, 248, 170), outline=(120, 180, 220))
        draw.rectangle((78, 184, 180, 212), fill=(190, 230, 245), outline=(110, 150, 180), width=3)
    else:
        raise ValueError(f"Unknown state '{state}'")
    return out


def _write_jsonl(path: Path, rows: List[Record]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")


def _split_val_ids(identity_ids: List[str], val_ratio: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    shuffled = identity_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return set(shuffled[:val_count])


def _build_state_prompt(state: str) -> str:
    token = f"<state_{state}>"
    instruction = STATE_INSTRUCTIONS.get(state, state)
    return (
        f"{token}, tiny tamagotchi-like pixel pet, {instruction}, "
        "consistent identity, readable silhouette"
    )


def _prepare_labeled_image(image: Image.Image, resolution: int = 256) -> Image.Image:
    rgba = image.convert("RGBA")
    squared = _pad_to_square(rgba)
    resample = Image.Resampling.NEAREST if max(rgba.size) <= 128 else Image.Resampling.LANCZOS
    return squared.resize((resolution, resolution), resample)


def _parse_state_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    cleaned = (raw or "").strip()
    if not cleaned:
        return mapping

    for chunk in cleaned.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Invalid --state_map format. Expected comma-separated folder=state pairs. "
                "Example: 0=eating,1=feeding,2=sleeping,3=hygiene"
            )
        folder, state = [part.strip() for part in item.split("=", 1)]
        if not folder:
            raise ValueError("Invalid --state_map entry with empty folder name.")
        if state not in STATE_KEYS:
            raise ValueError(f"Invalid state '{state}' in --state_map. Expected one of {STATE_KEYS}.")
        if folder in mapping:
            raise ValueError(f"Duplicate folder '{folder}' in --state_map.")
        mapping[folder] = state
    return mapping


def _resolve_state_dirs(input_dir: Path, explicit_map: Mapping[str, str]) -> Dict[Path, str]:
    subdirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not subdirs:
        raise RuntimeError(f"No subfolders found in labeled dataset directory: {input_dir}")

    resolved: Dict[Path, str] = {}
    if explicit_map:
        for folder, state in explicit_map.items():
            folder_path = input_dir / folder
            if not folder_path.exists() or not folder_path.is_dir():
                raise FileNotFoundError(f"State folder from --state_map not found: {folder_path}")
            resolved[folder_path] = state
        return resolved

    used_states: set[str] = set()
    for subdir in subdirs:
        name = subdir.name.strip().lower()
        state: str | None = None
        if name in STATE_KEYS:
            state = name
        elif name.isdigit():
            idx = int(name)
            if 0 <= idx < len(STATE_KEYS):
                state = STATE_KEYS[idx]

        if state is None:
            continue
        if state in used_states:
            raise RuntimeError(
                f"Ambiguous state mapping. Multiple folders map to '{state}'. "
                "Use --state_map to define folders explicitly."
            )
        used_states.add(state)
        resolved[subdir] = state

    states_found = set(resolved.values())
    if states_found != set(STATE_KEYS):
        raise RuntimeError(
            "Could not auto-map all 4 states from folders. "
            f"Found states: {sorted(states_found)}. Expected: {list(STATE_KEYS)}. "
            "Use --state_map, for example: 0=eating,1=feeding,2=sleeping,3=hygiene"
        )
    return resolved


def _build_dataset_from_raw_refs(input_dir: Path, out_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> None:
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    image_paths = list(_iter_images(input_dir))
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_dir}")

    identity_ids = [p.stem for p in image_paths]
    val_ids = _split_val_ids(identity_ids, val_ratio=val_ratio, seed=seed)

    train_rows: List[Record] = []
    val_rows: List[Record] = []

    for src in image_paths:
        identity_id = src.stem
        with Image.open(src) as img:
            pre = preprocess_reference_image(img, target_size=256)
        base = _make_base_style(pre.image, resolution=256)

        for state in STATE_KEYS:
            local_rng = random.Random(f"{seed}:{identity_id}:{state}")
            variant = _add_state_overlay(base, state, local_rng)
            filename = f"{identity_id}_{state}.png"
            out_path = out_images / filename
            variant.save(out_path, format="PNG")

            token = f"<state_{state}>"
            prompt = (
                f"{token}, tiny tamagotchi-like pixel pet, 32x32 sprite style, "
                "consistent identity, readable silhouette"
            )
            record = Record(
                image_path=str(out_path.relative_to(out_dir)),
                prompt=prompt,
                state=state,
                source_image=str(src),
                identity_id=identity_id,
            )
            if identity_id in val_ids:
                val_rows.append(record)
            else:
                train_rows.append(record)

    _write_jsonl(out_dir / "train_metadata.jsonl", train_rows)
    _write_jsonl(out_dir / "val_metadata.jsonl", val_rows)
    _write_jsonl(out_dir / "all_metadata.jsonl", train_rows + val_rows)

    print(
        json.dumps(
            {
                "mode": "synth_from_raw_refs",
                "input_images": len(image_paths),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )


def _build_dataset_from_state_folders(
    input_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    state_map: Mapping[str, str] | None = None,
) -> None:
    state_map = state_map or {}
    resolved_state_dirs = _resolve_state_dirs(input_dir, state_map)
    state_to_images: Dict[str, Dict[str, Path]] = {state: {} for state in STATE_KEYS}

    for state_dir, state in resolved_state_dirs.items():
        for path in _iter_images(state_dir):
            identity = path.stem
            if identity in state_to_images[state]:
                raise RuntimeError(
                    f"Duplicate identity '{identity}' in state folder '{state_dir}'. "
                    "Each file stem should be unique per state."
                )
            state_to_images[state][identity] = path

    identity_sets = [set(state_to_images[state].keys()) for state in STATE_KEYS]
    common_ids = set.intersection(*identity_sets) if identity_sets else set()
    if not common_ids:
        raise RuntimeError(
            "No identities are present in all 4 state folders. "
            "Ensure matching filenames across state directories."
        )

    dropped_count = len(set.union(*identity_sets)) - len(common_ids)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    val_ids = _split_val_ids(sorted(common_ids), val_ratio=val_ratio, seed=seed)
    train_rows: List[Record] = []
    val_rows: List[Record] = []

    for identity_id in sorted(common_ids):
        for state in STATE_KEYS:
            src = state_to_images[state][identity_id]
            with Image.open(src) as img:
                prepared = _prepare_labeled_image(img, resolution=256)

            filename = f"{identity_id}_{state}.png"
            out_path = out_images / filename
            prepared.save(out_path, format="PNG")

            record = Record(
                image_path=str(out_path.relative_to(out_dir)),
                prompt=_build_state_prompt(state),
                state=state,
                source_image=str(src),
                identity_id=identity_id,
            )
            if identity_id in val_ids:
                val_rows.append(record)
            else:
                train_rows.append(record)

    _write_jsonl(out_dir / "train_metadata.jsonl", train_rows)
    _write_jsonl(out_dir / "val_metadata.jsonl", val_rows)
    _write_jsonl(out_dir / "all_metadata.jsonl", train_rows + val_rows)

    print(
        json.dumps(
            {
                "mode": "from_labeled_state_folders",
                "state_dirs": {str(path): state for path, state in resolved_state_dirs.items()},
                "shared_identities": len(common_ids),
                "dropped_incomplete_identities": dropped_count,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )


def build_dataset(
    input_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    mode: str = "auto",
    state_map: Mapping[str, str] | None = None,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("--val_ratio must be between 0 and 1 (exclusive).")

    has_images = _has_images(input_dir)
    has_dirs = _has_subdirs(input_dir)

    run_mode = mode
    if mode == "auto":
        if has_images and not has_dirs:
            run_mode = "synth"
        elif has_dirs and not has_images:
            run_mode = "labeled"
        elif has_images and has_dirs:
            raise RuntimeError(
                "Input directory has both images and subfolders. "
                "Please pass --mode synth or --mode labeled explicitly."
            )
        else:
            raise RuntimeError(f"No supported images or subfolders found in {input_dir}")

    if run_mode == "synth":
        _build_dataset_from_raw_refs(input_dir, out_dir, val_ratio=val_ratio, seed=seed)
        return

    if run_mode == "labeled":
        _build_dataset_from_state_folders(
            input_dir=input_dir,
            out_dir=out_dir,
            val_ratio=val_ratio,
            seed=seed,
            state_map=state_map,
        )
        return

    raise ValueError(f"Unsupported mode '{mode}'. Expected one of auto/synth/labeled.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SpriteAI training dataset.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with source reference images.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio by identity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and overlays.")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "synth", "labeled"],
        help="auto: detect from input layout, synth: raw refs to synthetic states, labeled: use 4 state subfolders.",
    )
    parser.add_argument(
        "--state_map",
        type=str,
        default="",
        help=(
            "Optional folder-to-state map for labeled mode. "
            "Example: 0=eating,1=feeding,2=sleeping,3=hygiene"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_map = _parse_state_map(args.state_map)
    build_dataset(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mode=args.mode,
        state_map=state_map,
    )


if __name__ == "__main__":
    main()
