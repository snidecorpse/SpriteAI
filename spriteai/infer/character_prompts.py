"""Prompt templates for character-first sprite generation."""

from __future__ import annotations

from typing import Dict, Tuple

CHARACTER_VIEWS: Tuple[str, ...] = ("front", "right", "left", "back")

BASE_CHARACTER_PROMPT = (
    "<sprite_char>, pixel character sprite, one single character only, full body in frame, "
    "centered standing pose, clean silhouette, crisp pixel edges, compact shading, "
    "simple solid background, game asset sheet style, readable at tiny scale"
)

NEGATIVE_CHARACTER_PROMPT = (
    "photorealistic, camera photo, realistic skin pores, background clutter, room interior, "
    "furniture, text, watermark, logo, blur, anti-aliased edges, multiple characters, two people, "
    "duplicate character, mirrored twin, side by side characters, extra body, extra head, cut off body"
)

VIEW_TOKENS: Dict[str, str] = {
    "front": "<view_front>",
    "right": "<view_right>",
    "left": "<view_left>",
    "back": "<view_back>",
}


def _clean_user_prompt(prompt: str) -> str:
    return " ".join((prompt or "").strip().split())


def normalize_view(view: str) -> str:
    clean = (view or "front").strip().lower()
    if clean not in CHARACTER_VIEWS:
        raise ValueError(f"Unsupported view '{view}'. Expected one of {CHARACTER_VIEWS}.")
    return clean


def build_character_prompt(user_prompt: str, view: str = "front") -> str:
    clean_view = normalize_view(view)
    token = VIEW_TOKENS[clean_view]
    user = _clean_user_prompt(user_prompt)
    if not user:
        return f"{BASE_CHARACTER_PROMPT}, {token}"
    return f"{BASE_CHARACTER_PROMPT}, {token}, {user}"
