"""Prompt templates for sprite state generation."""

from __future__ import annotations

from typing import Dict, Tuple

STATE_KEYS: Tuple[str, ...] = ("eating", "feeding", "sleeping", "hygiene")

BASE_STYLE_PROMPT = (
    "tiny tamagotchi-like pet character, pixel-art game sprite, single character, "
    "clean silhouette, simple background, front-facing, compact proportions, readable at tiny scale"
)

NEGATIVE_PROMPT = (
    "photorealistic, realistic skin, realistic eyes, camera photo, room interior, furniture, "
    "wall, bed, cluttered background, blurry, anti-aliased edges, text, watermark, logo, "
    "deformed, multiple characters"
)

STATE_INSTRUCTIONS: Dict[str, str] = {
    "eating": "pet independently eating from a small bowl, visible food bite",
    "feeding": "owner hand feeding pet with a spoon or bottle, caring interaction",
    "sleeping": "pet asleep with closed eyes, cozy resting posture, subtle z symbol",
    "hygiene": "pet in hygiene moment, tiny bubbles or bath-cleaning cue, fresh look",
}


def _clean_user_prompt(prompt: str) -> str:
    return " ".join((prompt or "").strip().split())


def build_base_prompt(user_prompt: str) -> str:
    """Build the canonical base prompt from style + user modifications."""
    user = _clean_user_prompt(user_prompt)
    if not user:
        return BASE_STYLE_PROMPT
    return f"{BASE_STYLE_PROMPT}, {user}"


def build_state_prompt(state: str, user_prompt: str) -> str:
    """Build state prompt for one of the supported state keys."""
    if state not in STATE_KEYS:
        raise ValueError(f"Unsupported state '{state}'. Expected one of {STATE_KEYS}.")
    base = build_base_prompt(user_prompt)
    state_token = f"<state_{state}>"
    return f"{state_token}, {base}, {STATE_INSTRUCTIONS[state]}"
