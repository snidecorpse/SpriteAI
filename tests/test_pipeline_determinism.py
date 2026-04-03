import importlib

from PIL import Image


def _make_reference() -> Image.Image:
    img = Image.new("RGB", (256, 256), (210, 170, 120))
    px = img.load()
    for y in range(80, 180):
        for x in range(90, 170):
            px[x, y] = (120, 90, 70)
    return img


def test_generate_states_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("SPRITEAI_FORCE_FALLBACK", "1")
    import spriteai.infer.pipeline as pipeline_module

    importlib.reload(pipeline_module)

    ref = _make_reference()
    first = pipeline_module.generate_states(ref, prompt="red scarf", seed=7, creativity=0.35)
    second = pipeline_module.generate_states(ref, prompt="red scarf", seed=7, creativity=0.35)

    assert list(first.keys()) == ["eating", "feeding", "sleeping", "hygiene"]
    assert list(second.keys()) == ["eating", "feeding", "sleeping", "hygiene"]

    for state in first:
        assert first[state].size == (32, 32)
        assert second[state].size == (32, 32)
        assert first[state].tobytes() == second[state].tobytes()
