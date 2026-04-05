from PIL import Image, ImageDraw

from spriteai.infer.character_features import extract_character_features


def _make_synthetic_face(with_glasses: bool) -> Image.Image:
    img = Image.new("RGB", (384, 384), (236, 236, 238))
    draw = ImageDraw.Draw(img)
    draw.ellipse((118, 62, 266, 212), fill=(182, 140, 110))
    draw.pieslice((112, 40, 272, 194), start=180, end=360, fill=(42, 34, 30))
    draw.rounded_rectangle((118, 198, 266, 350), radius=24, fill=(106, 178, 162))
    draw.ellipse((160, 132, 184, 148), fill=(28, 28, 34))
    draw.ellipse((202, 132, 226, 148), fill=(28, 28, 34))

    if with_glasses:
        draw.rectangle((151, 122, 191, 156), outline=(32, 32, 40), width=4)
        draw.rectangle((194, 122, 234, 156), outline=(32, 32, 40), width=4)
        draw.line((191, 139, 194, 139), fill=(32, 32, 40), width=3)
    return img


def test_extract_character_features_with_glasses():
    profile = extract_character_features(_make_synthetic_face(with_glasses=True))
    assert profile.has_glasses is True
    assert profile.skin_tone in {"medium skin tone", "tan skin tone"}
    assert profile.hair_tone in {"black hair", "dark brown hair", "brown hair"}
    assert profile.clothing_color in {"mint top", "emerald top"}
    assert "wearing glasses" in profile.prompt_fragments
    assert "glasses=yes" in profile.summary


def test_extract_character_features_without_glasses():
    profile = extract_character_features(_make_synthetic_face(with_glasses=False))
    assert profile.has_glasses is False
    assert "wearing glasses" not in profile.prompt_fragments
    assert "skin=" in profile.summary
    assert "hair=" in profile.summary
    assert "clothing=" in profile.summary

