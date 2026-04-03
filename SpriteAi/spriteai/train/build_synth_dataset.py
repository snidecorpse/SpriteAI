"""Build synthetic 4-state training dataset from user-provided reference images."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw

from spriteai.infer.preprocess import preprocess_reference_image
from spriteai.infer.state_prompts import STATE_KEYS

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


def build_dataset(input_dir: Path, out_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    image_paths = list(_iter_images(input_dir))
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_dir}")

    rng = random.Random(seed)
    identity_ids = [p.stem for p in image_paths]
    shuffled = identity_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_ids = set(shuffled[:val_count])

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
                "input_images": len(image_paths),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic SpriteAI training dataset.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with source reference images.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio by identity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and overlays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
