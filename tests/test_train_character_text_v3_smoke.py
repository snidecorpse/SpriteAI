import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


def _write_sample(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), color).save(path, format="PNG")


@pytest.mark.skipif(
    os.getenv("SPRITEAI_RUN_TRAIN_SMOKE", "") not in {"1", "true", "yes"},
    reason="Set SPRITEAI_RUN_TRAIN_SMOKE=1 to run integration smoke test.",
)
def test_train_character_text_v3_smoke(tmp_path):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (view, color) in enumerate(
        [
            ("front", (210, 170, 140, 255)),
            ("right", (200, 160, 130, 255)),
            ("left", (190, 150, 120, 255)),
            ("back", (180, 140, 110, 255)),
        ],
        start=1,
    ):
        name = f"{idx:06d}.png"
        _write_sample(images / name, color)
        rows.append(
            {
                "image_path": str(Path("images") / name),
                "sample_id": f"{idx:06d}",
                "view": view,
                "identity_group": "000001",
                "prompt_v3": f"{view}-facing pixel character sprite, full body, simple background, single character",
                "attributes": {"hair": {"label": "brown hair"}},
            }
        )

    all_meta = dataset / "all_metadata.jsonl"
    with all_meta.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "out_v3"
    cmd = [
        sys.executable,
        "-m",
        "spriteai.train.train_character_text_v3",
        "--dataset_dir",
        str(dataset),
        "--output_dir",
        str(out_dir),
        "--model_id",
        "runwayml/stable-diffusion-v1-5",
        "--metadata_file",
        "all_metadata.jsonl",
        "--resolution",
        "256",
        "--batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--max_train_steps",
        "1",
        "--learning_rate",
        "1e-5",
        "--precision",
        "no",
        "--num_workers",
        "0",
        "--preview_every_steps",
        "1",
        "--preview_seeds",
        "7,11",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "unet").exists()
    assert (out_dir / "training_config.json").exists()
    previews = list((out_dir / "previews").glob("step_*.png"))
    preview_meta = list((out_dir / "previews").glob("step_*.json"))
    assert previews
    assert preview_meta

