import json
import shutil
from pathlib import Path

from PIL import Image

from spriteai.train.build_text_sprite_dataset_v3 import build_text_sprite_dataset_v3


def _test_root(name: str) -> Path:
    root = Path.cwd() / "tests_tmp_check" / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_png(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), color).save(path, format="PNG")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_build_text_sprite_dataset_v3_captions_and_numbering():
    root = _test_root("test_build_text_sprite_dataset_v3")
    inp = root / "input"
    images = inp / "images"
    rows = []
    for idx, (view, color) in enumerate(
        [
            ("front", (200, 160, 130, 255)),
            ("right", (180, 140, 120, 255)),
            ("left", (150, 120, 100, 255)),
            ("back", (110, 90, 80, 255)),
        ],
        start=1,
    ):
        name = f"{idx:06d}.png"
        _write_png(images / name, color)
        rows.append(
            {
                "image_path": str(Path("images") / name),
                "sample_id": f"{idx:06d}",
                "view": view,
                "identity_group": "a",
            }
        )
    _write_jsonl(inp / "all_metadata.jsonl", rows)

    out = root / "out"
    build_text_sprite_dataset_v3(input_dir=inp, out_dir=out)
    data = _read_jsonl(out / "all_metadata.jsonl")

    assert len(data) == 4
    assert [r["sample_id"] for r in data] == ["000001", "000002", "000003", "000004"]
    image_names = sorted([p.name for p in (out / "images").iterdir() if p.is_file()])
    assert image_names == ["000001.png", "000002.png", "000003.png", "000004.png"]
    assert "front-facing" in data[0]["prompt_v3"]
    assert "right-facing" in data[1]["prompt_v3"]
    assert "left-facing" in data[2]["prompt_v3"]
    assert "back-facing" in data[3]["prompt_v3"]
    for row in data:
        attrs = row["attributes"]
        assert "hair" in attrs and "skin" in attrs and "top" in attrs
        assert "label" in attrs["hair"]


def test_build_text_sprite_dataset_v3_is_deterministic():
    root = _test_root("test_build_text_sprite_dataset_v3_deterministic")
    inp = root / "input"
    images = inp / "images"
    rows = []
    for idx in range(1, 4):
        name = f"{idx:06d}.png"
        _write_png(images / name, (80 + idx * 20, 90 + idx * 10, 100 + idx * 5, 255))
        rows.append(
            {
                "image_path": str(Path("images") / name),
                "sample_id": f"{idx:06d}",
                "view": "front",
                "identity_group": "z",
            }
        )
    _write_jsonl(inp / "all_metadata.jsonl", rows)

    out_a = root / "out_a"
    out_b = root / "out_b"
    build_text_sprite_dataset_v3(input_dir=inp, out_dir=out_a)
    build_text_sprite_dataset_v3(input_dir=inp, out_dir=out_b)
    a_rows = _read_jsonl(out_a / "all_metadata.jsonl")
    b_rows = _read_jsonl(out_b / "all_metadata.jsonl")
    assert a_rows == b_rows

