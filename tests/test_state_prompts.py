import pytest

from spriteai.infer.state_prompts import STATE_KEYS, build_state_prompt


@pytest.mark.parametrize("state", STATE_KEYS)
def test_state_prompt_includes_state_token(state):
    prompt = build_state_prompt(state, "blue hair")
    assert f"<state_{state}>" in prompt
    assert "blue hair" in prompt
