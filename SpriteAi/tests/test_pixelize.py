from PIL import Image

from spriteai.infer.pixelize import count_unique_colors, pixelize_image


def test_pixelize_image_shape_and_color_budget():
    img = Image.new("RGB", (256, 256))
    pixels = img.load()
    for y in range(256):
        for x in range(256):
            pixels[x, y] = ((x * 17) % 256, (y * 13) % 256, ((x + y) * 11) % 256)

    out = pixelize_image(img, size=32)
    assert out.size == (32, 32)
    assert count_unique_colors(out) <= 16
