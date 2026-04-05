from PIL import Image

from spriteai.infer.character_preprocess import preprocess_character_reference_image


def test_preprocess_character_reference_image_shape_and_mode():
    img = Image.new("RGB", (480, 320), (180, 170, 175))
    result = preprocess_character_reference_image(img, target_size=384)
    assert result.image.size == (384, 384)
    assert result.crop_mode in {"face-detect", "heuristic"}
    assert result.warnings
