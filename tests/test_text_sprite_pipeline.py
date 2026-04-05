import importlib

import pytest


def test_generate_text_sprite_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("SPRITEAI_TEXT_V3_FORCE_FALLBACK", "1")
    import spriteai.infer.text_sprite_pipeline as text_sprite_pipeline

    importlib.reload(text_sprite_pipeline)
    first = text_sprite_pipeline.generate_text_sprite_with_meta(
        prompt="front-facing pixel character sprite, black hair, white hoodie",
        seed=17,
        creativity=0.14,
        view="front",
    )
    second = text_sprite_pipeline.generate_text_sprite_with_meta(
        prompt="front-facing pixel character sprite, black hair, white hoodie",
        seed=17,
        creativity=0.14,
        view="front",
    )

    assert first.backend == "fallback"
    assert second.backend == "fallback"
    assert first.image.size == (64, 64)
    assert second.image.size == (64, 64)
    assert first.preview_image.size == (512, 512)
    assert second.preview_image.size == (512, 512)
    assert first.image.tobytes() == second.image.tobytes()
    assert "fallback" in first.model_status
    assert any(w.startswith("Candidate count: ") for w in first.warnings)


def test_generate_text_sprite_invalid_view(monkeypatch):
    monkeypatch.setenv("SPRITEAI_TEXT_V3_FORCE_FALLBACK", "1")
    import spriteai.infer.text_sprite_pipeline as text_sprite_pipeline

    importlib.reload(text_sprite_pipeline)
    with pytest.raises(ValueError):
        text_sprite_pipeline.generate_text_sprite(
            prompt="pixel character sprite",
            seed=1,
            view="diagonal",
        )

