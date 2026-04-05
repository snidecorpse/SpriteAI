from __future__ import annotations

from PIL import Image, ImageDraw

from spriteai.infer.character_postprocess import choose_best_character_candidate, postprocess_character_candidate


def _make_dual_candidate() -> Image.Image:
    img = Image.new("RGB", (256, 256), (18, 16, 34))
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 46, 104, 226), fill=(210, 132, 102))
    draw.rectangle((162, 58, 228, 220), fill=(120, 166, 210))
    return img


def _make_single_candidate() -> Image.Image:
    img = Image.new("RGB", (256, 256), (18, 16, 34))
    draw = ImageDraw.Draw(img)
    draw.rectangle((78, 34, 176, 232), fill=(188, 128, 98))
    return img


def test_postprocess_character_candidate_recenters_and_reduces_dual_subject():
    scored = postprocess_character_candidate(_make_dual_candidate(), sprite_size=64)
    assert scored.image.size == (64, 64)
    assert scored.component_count <= 2
    assert scored.largest_component_ratio >= 0.82


def test_choose_best_character_candidate_prefers_single_subject():
    single = _make_single_candidate()
    dual = _make_dual_candidate()
    best = choose_best_character_candidate([dual, single], sprite_size=64)
    single_scored = postprocess_character_candidate(single, sprite_size=64)
    assert best.image.size == (64, 64)
    assert best.score >= single_scored.score - 0.2
