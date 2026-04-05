import importlib
import shutil
from pathlib import Path

import pytest
from PIL import Image


def _make_reference() -> Image.Image:
    img = Image.new("RGB", (320, 240), (200, 170, 140))
    px = img.load()
    for y in range(70, 190):
        for x in range(110, 210):
            px[x, y] = (120, 95, 75)
    return img


def _test_root(name: str) -> Path:
    root = Path.cwd() / "tests_tmp_check" / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_generate_character_sprite_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("SPRITEAI_CHARACTER_FORCE_FALLBACK", "1")
    import spriteai.infer.character_pipeline as character_pipeline

    importlib.reload(character_pipeline)
    ref = _make_reference()
    first = character_pipeline.generate_character_sprite_with_meta(
        reference_image=ref,
        prompt="round glasses",
        seed=17,
        creativity=0.18,
        view="front",
    )
    second = character_pipeline.generate_character_sprite_with_meta(
        reference_image=ref,
        prompt="round glasses",
        seed=17,
        creativity=0.18,
        view="front",
    )

    assert first.backend == "fallback"
    assert second.backend == "fallback"
    assert first.image.size == (64, 64)
    assert second.image.size == (64, 64)
    assert first.preview_image.size == (512, 512)
    assert second.preview_image.size == (512, 512)
    assert first.sprite_size == 64
    assert first.preview_size == 512
    assert first.image.tobytes() == second.image.tobytes()
    assert first.crop_mode in {"face-detect", "heuristic"}
    assert any(w.startswith("Crop mode: ") for w in first.warnings)
    assert "skin=" in first.feature_summary
    assert first.lora_status


def test_character_pipeline_autodetects_default_lora_path(monkeypatch):
    tmp_path = _test_root("test_character_pipeline_autodetect_lora")
    monkeypatch.chdir(tmp_path)
    lora_dir = tmp_path / "artifacts" / "lora_sd15_character_v2"
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"dummy")

    monkeypatch.delenv("SPRITEAI_CHARACTER_LORA_PATH", raising=False)
    monkeypatch.delenv("SPRITEAI_LORA_PATH", raising=False)
    monkeypatch.setenv("SPRITEAI_CHARACTER_FORCE_FALLBACK", "1")

    import spriteai.infer.character_pipeline as character_pipeline

    importlib.reload(character_pipeline)
    pipe = character_pipeline.CharacterSpritePipeline()
    assert pipe.lora_path is not None
    assert "lora_sd15_character_v2" in pipe.lora_path

    result = pipe.generate(_make_reference(), seed=123)
    assert "ignored on fallback backend" in result.lora_status


def test_generate_character_sprite_invalid_view(monkeypatch):
    monkeypatch.setenv("SPRITEAI_CHARACTER_FORCE_FALLBACK", "1")
    import spriteai.infer.character_pipeline as character_pipeline

    importlib.reload(character_pipeline)
    with pytest.raises(ValueError):
        character_pipeline.generate_character_sprite(
            reference_image=_make_reference(),
            prompt="",
            seed=1,
            view="diagonal",
        )
