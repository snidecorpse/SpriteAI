"""V3 text-only sprite generation pipeline."""

from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

from .character_postprocess import choose_best_character_candidate, postprocess_character_candidate
from .character_prompts import normalize_view

NEGATIVE_TEXT_SPRITE_PROMPT = (
    "photorealistic, camera photo, room, furniture, text, watermark, blurry, anti-aliased edges, "
    "multiple characters, mirrored twin, side by side characters, extra limbs, cut off body"
)

VIEW_PHRASES = {
    "front": "front-facing",
    "right": "right-facing",
    "left": "left-facing",
    "back": "back-facing",
}


@dataclass(frozen=True)
class TextSpriteGenerationResult:
    image: Image.Image
    preview_image: Image.Image
    warnings: List[str]
    backend: str
    latency_seconds: float
    model_status: str
    sprite_size: int
    preview_size: int


def _hash_to_seed(base_seed: int, text: str) -> int:
    payload = f"{base_seed}:{text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _nearest_preview(image: Image.Image, scale: int) -> Image.Image:
    factor = max(1, int(scale))
    if factor == 1:
        return image.copy()
    return image.resize((image.size[0] * factor, image.size[1] * factor), Image.Resampling.NEAREST)


def _normalize_prompt(prompt: str) -> str:
    clean = " ".join((prompt or "").strip().split())
    if not clean:
        clean = "pixel character sprite, full body, single character, simple background"
    return clean


def _build_text_prompt(prompt: str, view: str) -> str:
    view_key = normalize_view(view)
    view_phrase = VIEW_PHRASES[view_key]
    clean = _normalize_prompt(prompt)
    return f"{view_phrase} {clean}, pixel character sprite, full body, single character, simple background"


def _default_model_dir(configured: Optional[str]) -> str:
    if configured and configured.strip():
        return configured.strip()
    return str(Path("artifacts") / "sd15_character_text_v3")


def _seed_color(seed: int, low: int = 50, high: int = 210) -> tuple[int, int, int]:
    rng = random.Random(seed)
    return (rng.randint(low, high), rng.randint(low, high), rng.randint(low, high))


def _render_fallback_text_character(prompt: str, seed: int, view: str) -> Image.Image:
    char_seed = _hash_to_seed(seed, f"{prompt}:{view}")
    skin = _seed_color(_hash_to_seed(char_seed, "skin"), 105, 220)
    hair = _seed_color(_hash_to_seed(char_seed, "hair"), 20, 160)
    cloth = _seed_color(_hash_to_seed(char_seed, "cloth"), 35, 210)
    outline = (36, 36, 46)

    canvas = Image.new("RGBA", (256, 256), (12, 12, 20, 255))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((86, 34, 170, 122), fill=skin, outline=outline, width=4)
    draw.pieslice((80, 24, 176, 120), start=180, end=360, fill=hair, outline=hair, width=2)
    draw.rounded_rectangle((84, 112, 172, 206), radius=20, fill=cloth, outline=outline, width=4)
    draw.rounded_rectangle((58, 124, 88, 190), radius=12, fill=skin, outline=outline, width=3)
    draw.rounded_rectangle((168, 124, 198, 190), radius=12, fill=skin, outline=outline, width=3)
    draw.rounded_rectangle((96, 194, 126, 240), radius=10, fill=skin, outline=outline, width=3)
    draw.rounded_rectangle((130, 194, 160, 240), radius=10, fill=skin, outline=outline, width=3)

    eye_y = 80
    if view == "front":
        eye_positions = (112, 144)
    elif view == "left":
        eye_positions = (108,)
    elif view == "right":
        eye_positions = (148,)
    else:
        eye_positions = ()
    for cx in eye_positions:
        draw.ellipse((cx - 8, eye_y - 6, cx + 8, eye_y + 6), fill=(238, 238, 242), outline=outline, width=2)
        draw.ellipse((cx - 2, eye_y - 1, cx + 2, eye_y + 1), fill=(24, 24, 32))

    if view == "back":
        draw.rounded_rectangle((94, 62, 162, 102), radius=12, fill=hair, outline=hair, width=1)
    return canvas


class TextSpritePipeline:
    """Generate one sprite from prompt text using V3 finetuned UNet when available."""

    def __init__(self, model_id: Optional[str] = None, model_dir: Optional[str] = None):
        self.model_id = model_id or os.getenv("SPRITEAI_TEXT_V3_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        self.model_dir = _default_model_dir(model_dir or os.getenv("SPRITEAI_TEXT_V3_MODEL_DIR"))
        self.model_input_size = int(os.getenv("SPRITEAI_TEXT_V3_MODEL_INPUT_SIZE", "384"))
        self.default_candidate_count = max(1, int(os.getenv("SPRITEAI_TEXT_V3_CANDIDATES", "2")))
        self.force_fallback = os.getenv(
            "SPRITEAI_TEXT_V3_FORCE_FALLBACK",
            os.getenv("SPRITEAI_FORCE_FALLBACK", ""),
        ).lower() in {"1", "true", "yes"}

        self._pipe = None
        self._torch = None
        self._backend = "uninitialized"
        self._backend_initialized = False
        self._load_warnings: List[str] = []
        self._model_status = "not initialized"

    @property
    def backend(self) -> str:
        return self._backend

    def _load_diffusers_backend(self) -> bool:
        if self.force_fallback:
            self._load_warnings.append("SPRITEAI_TEXT_V3_FORCE_FALLBACK enabled; using fallback backend.")
            self._model_status = "fallback forced by environment"
            return False

        try:
            import torch
            from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        except Exception as exc:  # pragma: no cover
            self._load_warnings.append(f"Diffusers backend unavailable: {exc}")
            self._model_status = f"diffusers unavailable: {exc}"
            return False

        try:
            use_cuda = bool(torch.cuda.is_available())
            device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            model_path = Path(self.model_dir)
            unet_path = model_path / "unet"
            if unet_path.exists():
                finetuned_unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
                pipe.unet = finetuned_unet
                self._model_status = f"loaded finetuned UNet from: {unet_path}"
            else:
                self._model_status = f"V3 model not found at {unet_path}; using base model UNet"
                self._load_warnings.append(self._model_status)

            pipe.to(device)
            pipe.enable_attention_slicing()
            self._pipe = pipe
            self._torch = torch
            self._backend = "diffusers"
            return True
        except Exception as exc:
            self._load_warnings.append(f"Failed to initialize V3 diffusers backend: {exc}")
            self._model_status = f"failed to initialize diffusers backend: {exc}"
            return False

    def _ensure_backend(self) -> None:
        if self._backend_initialized:
            return
        loaded = self._load_diffusers_backend()
        if not loaded:
            self._backend = "fallback"
        self._backend_initialized = True

    def _generate_with_diffusers(
        self,
        prompt: str,
        seed: int,
        creativity: float,
        sprite_size: int,
        candidate_count: int,
    ) -> Image.Image:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("V3 diffusers pipeline is not initialized.")
        torch = self._torch
        device = self._pipe.device.type

        creativity = _clamp(creativity, 0.08, 0.38)
        guidance = _clamp(8.8 - creativity * 4.0, 6.8, 8.8)
        steps = int(round(38 + creativity * 22))
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = self._pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_TEXT_SPRITE_PROMPT,
            guidance_scale=guidance,
            num_inference_steps=steps,
            width=self.model_input_size,
            height=self.model_input_size,
            num_images_per_prompt=candidate_count,
            generator=generator,
        ).images
        best = choose_best_character_candidate(outputs, sprite_size=sprite_size)
        return best.image

    def _generate_with_fallback(
        self,
        prompt: str,
        seed: int,
        view: str,
        sprite_size: int,
    ) -> Image.Image:
        image = _render_fallback_text_character(prompt=prompt, seed=seed, view=view)
        self._model_status = "fallback renderer active"
        return postprocess_character_candidate(image, sprite_size=sprite_size).image

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        creativity: float = 0.14,
        view: str = "front",
        sprite_size: int = 64,
        preview_scale: int = 8,
        candidate_count: Optional[int] = None,
    ) -> TextSpriteGenerationResult:
        start = time.perf_counter()
        if seed is None:
            seed = random.randint(0, 2_147_483_647)
        clean_view = normalize_view(view)
        full_prompt = _build_text_prompt(prompt, clean_view)
        clean_size = max(16, int(sprite_size))
        clean_preview_scale = max(1, int(preview_scale))
        clean_candidates = max(1, int(candidate_count or self.default_candidate_count))

        self._ensure_backend()
        warnings = list(self._load_warnings)
        candidate_msg = f"Candidate count: {clean_candidates}"
        if candidate_msg not in warnings:
            warnings.append(candidate_msg)

        if self.backend == "diffusers" and self._torch is not None and not bool(self._torch.cuda.is_available()):
            cpu_warn = "V3 diffusers backend is running on CPU; CUDA-enabled PyTorch is required for faster generation."
            if cpu_warn not in warnings:
                warnings.append(cpu_warn)

        if self.backend == "diffusers":
            image = self._generate_with_diffusers(
                prompt=full_prompt,
                seed=int(seed),
                creativity=float(creativity),
                sprite_size=clean_size,
                candidate_count=clean_candidates,
            )
        else:
            image = self._generate_with_fallback(
                prompt=full_prompt,
                seed=int(seed),
                view=clean_view,
                sprite_size=clean_size,
            )

        preview = _nearest_preview(image, clean_preview_scale)
        elapsed = time.perf_counter() - start
        return TextSpriteGenerationResult(
            image=image,
            preview_image=preview,
            warnings=warnings,
            backend=self.backend,
            latency_seconds=elapsed,
            model_status=self._model_status,
            sprite_size=clean_size,
            preview_size=preview.size[0],
        )


_DEFAULT_TEXT_PIPELINE: Optional[TextSpritePipeline] = None


def _get_default_text_pipeline() -> TextSpritePipeline:
    global _DEFAULT_TEXT_PIPELINE
    if _DEFAULT_TEXT_PIPELINE is None:
        _DEFAULT_TEXT_PIPELINE = TextSpritePipeline()
    return _DEFAULT_TEXT_PIPELINE


def generate_text_sprite(
    prompt: str,
    seed: Optional[int] = None,
    creativity: float = 0.14,
    view: str = "front",
    sprite_size: int = 64,
    candidate_count: Optional[int] = None,
) -> Image.Image:
    result = _get_default_text_pipeline().generate(
        prompt=prompt,
        seed=seed,
        creativity=creativity,
        view=view,
        sprite_size=sprite_size,
        candidate_count=candidate_count,
    )
    return result.image


def generate_text_sprite_with_meta(
    prompt: str,
    seed: Optional[int] = None,
    creativity: float = 0.14,
    view: str = "front",
    sprite_size: int = 64,
    preview_scale: int = 8,
    candidate_count: Optional[int] = None,
) -> TextSpriteGenerationResult:
    return _get_default_text_pipeline().generate(
        prompt=prompt,
        seed=seed,
        creativity=creativity,
        view=view,
        sprite_size=sprite_size,
        preview_scale=preview_scale,
        candidate_count=candidate_count,
    )

