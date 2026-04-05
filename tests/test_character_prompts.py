import pytest

from spriteai.infer.character_prompts import CHARACTER_VIEWS, build_character_prompt, normalize_view


@pytest.mark.parametrize("view", CHARACTER_VIEWS)
def test_build_character_prompt_contains_view_token(view):
    prompt = build_character_prompt("round glasses", view=view)
    assert "<sprite_char>" in prompt
    assert f"<view_{view}>" in prompt
    assert "round glasses" in prompt


def test_normalize_view_rejects_invalid():
    with pytest.raises(ValueError):
        normalize_view("top")
