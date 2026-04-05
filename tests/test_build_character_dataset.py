import json
import shutil
from pathlib import Path

from PIL import Image

from spriteai.train.build_character_dataset import build_character_dataset


def _write_sprite(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), color).save(path, format="PNG")


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _test_root(name: str) -> Path:
    root = Path.cwd() / "tests_tmp_check" / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_build_character_dataset_mapping_and_numbering():
    tmp_path = _test_root("test_build_character_dataset_mapping")
    root = tmp_path / "dataset"
    # 2=front,1=right,3=left,0=back
    for folder, color in [("2", (220, 180, 150, 255)), ("1", (200, 160, 130, 255)), ("3", (180, 140, 110, 255)), ("0", (160, 120, 90, 255))]:
        _write_sprite(root / folder / "10.png", color)
        _write_sprite(root / folder / "11.png", color)

    out_dir = tmp_path / "out"
    build_character_dataset(input_dir=root, out_dir=out_dir, val_ratio=0.5, seed=7)

    all_rows = _read_jsonl(out_dir / "all_metadata.jsonl")
    train_rows = _read_jsonl(out_dir / "train_metadata.jsonl")
    val_rows = _read_jsonl(out_dir / "val_metadata.jsonl")
    image_dir = out_dir / "images"

    assert len(all_rows) == 8
    assert len(train_rows) + len(val_rows) == 8

    files = sorted([p.name for p in image_dir.iterdir() if p.is_file()])
    assert files == [f"{i:06d}.png" for i in range(1, 9)]

    sample_ids = [row["sample_id"] for row in all_rows]
    assert sample_ids == [f"{i:06d}" for i in range(1, 9)]

    views = {row["view"] for row in all_rows}
    assert views == {"front", "right", "left", "back"}
    assert all("<sprite_char>" in row["prompt"] for row in all_rows)
    assert all("<view_" in row["prompt"] for row in all_rows)


def test_character_dataset_split_is_by_identity_group():
    tmp_path = _test_root("test_build_character_dataset_split")
    root = tmp_path / "dataset"
    for folder in ("0", "1", "2", "3"):
        _write_sprite(root / folder / "a.png", (100, 100, 100, 255))
        _write_sprite(root / folder / "b.png", (120, 120, 120, 255))
        _write_sprite(root / folder / "c.png", (140, 140, 140, 255))

    out_dir = tmp_path / "out"
    build_character_dataset(input_dir=root, out_dir=out_dir, val_ratio=0.34, seed=42)

    train_rows = _read_jsonl(out_dir / "train_metadata.jsonl")
    val_rows = _read_jsonl(out_dir / "val_metadata.jsonl")
    train_ids = {row["identity_group"] for row in train_rows}
    val_ids = {row["identity_group"] for row in val_rows}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.union(val_ids) == {"a", "b", "c"}
