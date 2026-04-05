import json
import shutil
from pathlib import Path

from PIL import Image

from spriteai.train.build_synth_dataset import build_dataset


def _write_png(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), color).save(path, format="PNG")


def _read_jsonl(path):
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


def test_build_dataset_from_numeric_state_folders():
    tmp_path = _test_root("test_build_synth_dataset_numeric")
    input_dir = tmp_path / "dataset"
    for folder in ["0", "1", "2", "3"]:
        _write_png(input_dir / folder / "a.png", (10, 20, 30, 255))
        _write_png(input_dir / folder / "b.png", (40, 50, 60, 255))

    out_dir = tmp_path / "out"
    build_dataset(
        input_dir=input_dir,
        out_dir=out_dir,
        val_ratio=0.5,
        seed=7,
        mode="labeled",
        state_map={},
    )

    train_rows = _read_jsonl(out_dir / "train_metadata.jsonl")
    val_rows = _read_jsonl(out_dir / "val_metadata.jsonl")
    all_rows = _read_jsonl(out_dir / "all_metadata.jsonl")

    assert len(train_rows) + len(val_rows) == 8
    assert len(all_rows) == 8
    assert {row["state"] for row in all_rows} == {"eating", "feeding", "sleeping", "hygiene"}
    assert all("<state_" in row["prompt"] for row in all_rows)


def test_labeled_mode_drops_incomplete_identities():
    tmp_path = _test_root("test_build_synth_dataset_incomplete")
    input_dir = tmp_path / "dataset"
    for folder in ["0", "1", "2", "3"]:
        _write_png(input_dir / folder / "shared.png", (10, 20, 30, 255))
    _write_png(input_dir / "0" / "missing.png", (10, 20, 30, 255))

    out_dir = tmp_path / "out"
    build_dataset(
        input_dir=input_dir,
        out_dir=out_dir,
        val_ratio=0.5,
        seed=9,
        mode="labeled",
        state_map={"0": "eating", "1": "feeding", "2": "sleeping", "3": "hygiene"},
    )

    all_rows = _read_jsonl(out_dir / "all_metadata.jsonl")
    assert len(all_rows) == 4
    assert {row["identity_id"] for row in all_rows} == {"shared"}
