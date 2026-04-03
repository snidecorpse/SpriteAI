"""Main generation pipeline for 4-state pixel sprites."""

from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from .pixelize import pixelize_image
from .preprocess import preprocess_reference_image
from .state_prompts import NEGATIVE_PROMPT, STATE_KEYS, build_base_prompt, build_state_prompt


@dataclass(frozen=True)
class GenerationResult:
    images: Dict[str, Image.Image]
    warnings: List[str]
    backend: str
    latency_seconds: float


def _hash_to_seed(base_seed: int, text: str) -> int:
    payload = f"{base_seed}:{text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _dominant_colors(image: Image.Image, colors: int = 4) -> List[tuple[int, int, int]]:
    reduced = image.convert("RGB").resize((48, 48), Image.Resampling.LANCZOS)
    q = reduced.quantize(colors=colors, dither=Image.Dither.NONE).convert("RGB")
    counts = q.getcolors(maxcolors=colors * 8) or []
    counts.sort(key=lambda x: x[0], reverse=True)
    swatches = [rgb for _, rgb in counts]
    if not swatches:
        swatches = [(170, 170, 170), (90, 90, 90), (220, 220, 220), (60, 60, 60)]
    while len(swatches) < colors:
        swatches.append(swatches[-1])
    return swatches[:colors]


def _render_base_character(reference: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    colors = _dominant_colors(reference, colors=4)
    base, outline, eye, accent = colors[0], colors[1], colors[2], colors[3]

    canvas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    body_w = rng.randint(128, 160)
    body_h = rng.randint(118, 150)
    cx, cy = 128, 132
    x0 = cx - body_w // 2
    y0 = cy - body_h // 2
    x1 = cx + body_w // 2
    y1 = cy + body_h // 2

    draw.ellipse((x0, y0, x1, y1), fill=base, outline=outline, width=6)

    ear_size = rng.randint(26, 40)
    draw.ellipse((x0 + 10, y0 - 18, x0 + 10 + ear_size, y0 + 20), fill=base, outline=outline, width=5)
    draw.ellipse((x1 - ear_size - 10, y0 - 18, x1 - 10, y0 + 20), fill=base, outline=outline, width=5)

    eye_y = cy - 10
    eye_dx = rng.randint(20, 30)
    eye_w = 8
    eye_h = 12
    draw.ellipse((cx - eye_dx - eye_w, eye_y - eye_h, cx - eye_dx + eye_w, eye_y + eye_h), fill=eye)
    draw.ellipse((cx + eye_dx - eye_w, eye_y - eye_h, cx + eye_dx + eye_w, eye_y + eye_h), fill=eye)

    draw.rounded_rectangle((cx - 20, cy + 20, cx + 20, cy + 32), radius=4, fill=accent, outline=outline, width=3)
    return canvas


def _decorate_state(base: Image.Image, state: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    out = base.copy()
    draw = ImageDraw.Draw(out)

    if state == "eating":
        draw.rounded_rectangle((88, 172, 168, 206), radius=8, fill=(110, 78, 48), outline=(50, 34, 24), width=4)
        draw.ellipse((120, 152, 136, 166), fill=(236, 210, 120), outline=(120, 94, 42), width=2)
    elif state == "feeding":
        draw.rounded_rectangle((26, 136, 98, 164), radius=10, fill=(233, 194, 165), outline=(120, 90, 70), width=4)
        draw.rectangle((88, 147, 140, 154), fill=(180, 180, 180))
        draw.ellipse((137, 143, 155, 159), fill=(220, 220, 220), outline=(100, 100, 100), width=2)
    elif state == "sleeping":
        draw.line((98, 121, 116, 121), fill=(35, 35, 35), width=4)
        draw.line((140, 121, 158, 121), fill=(35, 35, 35), width=4)
        draw.arc((106, 146, 150, 172), start=0, end=180, fill=(64, 64, 110), width=4)
        z_x = 172 + rng.randint(-5, 5)
        draw.text((z_x, 74), "Z", fill=(230, 230, 255))
    elif state == "hygiene":
        for _ in range(9):
            x = rng.randint(70, 190)
            y = rng.randint(82, 190)
            r = rng.randint(6, 14)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(190, 230, 250, 180), outline=(120, 180, 220))
        draw.rectangle((76, 182, 180, 210), fill=(190, 230, 245), outline=(110, 150, 180), width=3)
    else:
        raise ValueError(f"Unknown state '{state}'")
    return out


class SpriteGenerationPipeline:
    """Local generation pipeline with diffusers backend and deterministic fallback."""

    def __init__(self, model_id: Optional[str] = None, lora_path: Optional[str] = None):
        self.model_id = model_id or os.getenv("SPRITEAI_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        self.lora_path = lora_path or os.getenv("SPRITEAI_LORA_PATH")
        self.force_fallback = os.getenv("SPRITEAI_FORCE_FALLBACK", "").lower() in {"1", "true", "yes"}

        self._pipe = None
        self._torch = None
        self._backend = "uninitialized"
        self._backend_initialized = False
        self._load_warnings: List[str] = []
        self._ip_adapter_loaded = False

    @property
    def backend(self) -> str:
        return self._backend

    def _load_diffusers_backend(self) -> bool:
        if self.force_fallback:
            self._load_warnings.append("SPRITEAI_FORCE_FALLBACK enabled; using deterministic fallback backend.")
            return False

        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
        except Exception as exc:  # pragma: no cover - import path depends on env
            self._load_warnings.append(f"Diffusers backend not available: {exc}")
            return False

        try:
            use_cuda = bool(torch.cuda.is_available())
            device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe.to(device)
            pipe.enable_attention_slicing()

            if self.lora_path:
                pipe.load_lora_weights(self.lora_path)

            # Optional image-conditioning adapter when configured.
            ip_repo = os.getenv("SPRITEAI_IP_ADAPTER_REPO")
            ip_weight = os.getenv("SPRITEAI_IP_ADAPTER_WEIGHT")
            ip_subfolder = os.getenv("SPRITEAI_IP_ADAPTER_SUBFOLDER", "models")
            if ip_repo and ip_weight and hasattr(pipe, "load_ip_adapter"):
                try:
                    pipe.load_ip_adapter(ip_repo, subfolder=ip_subfolder, weight_name=ip_weight)
                    if hasattr(pipe, "set_ip_adapter_scale"):
                        pipe.set_ip_adapter_scale(float(os.getenv("SPRITEAI_IP_ADAPTER_SCALE", "0.8")))
                    self._ip_adapter_loaded = True
                except Exception as exc:
                    self._load_warnings.append(f"IP-Adapter unavailable, continuing without it: {exc}")

            self._pipe = pipe
            self._torch = torch
            self._backend = "diffusers"
            return True
        except Exception as exc:
            self._load_warnings.append(f"Failed to initialize diffusers backend: {exc}")
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
        preprocessed_ref: Image.Image,
        prompt: str,
        seed: int,
        creativity: float,
    ) -> Dict[str, Image.Image]:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("Diffusers pipeline is not initialized.")

        torch = self._torch
        device = self._pipe.device.type
        strength = _clamp(creativity, 0.3, 0.45)
        base_strength = _clamp(0.45 + creativity * 0.15, 0.45, 0.6)

        common = {
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 7.0,
            "num_inference_steps": 24,
        }
        ip_kwargs = {}
        if self._ip_adapter_loaded:
            ip_kwargs["ip_adapter_image"] = preprocessed_ref.convert("RGB")

        base_generator = torch.Generator(device=device).manual_seed(seed)
        base_image = self._pipe(
            prompt=build_base_prompt(prompt),
            image=preprocessed_ref.convert("RGB"),
            strength=base_strength,
            generator=base_generator,
            **common,
            **ip_kwargs,
        ).images[0]

        outputs: Dict[str, Image.Image] = {}
        for state in STATE_KEYS:
            state_generator = torch.Generator(device=device).manual_seed(seed)
            state_image = self._pipe(
                prompt=build_state_prompt(state, prompt),
                image=base_image,
                strength=strength,
                generator=state_generator,
                **common,
                **ip_kwargs,
            ).images[0]
            outputs[state] = pixelize_image(state_image, size=32)
        return outputs

    def _generate_with_fallback(
        self,
        preprocessed_ref: Image.Image,
        prompt: str,
        seed: int,
    ) -> Dict[str, Image.Image]:
        base_seed = _hash_to_seed(seed, prompt)
        character = _render_base_character(preprocessed_ref, seed=base_seed)
        outputs: Dict[str, Image.Image] = {}
        for state in STATE_KEYS:
            state_seed = _hash_to_seed(seed, f"{prompt}:{state}")
            state_image = _decorate_state(character, state, seed=state_seed)
            outputs[state] = pixelize_image(state_image, size=32)
        return outputs

    def generate(
        self,
        reference_image: Image.Image,
        prompt: str,
        seed: Optional[int] = None,
        creativity: float = 0.35,
    ) -> GenerationResult:
        if not isinstance(reference_image, Image.Image):
            raise ValueError("Expected a valid image input.")
        if prompt is None:
            prompt = ""

        start = time.perf_counter()
        if seed is None:
            seed = random.randint(0, 2_147_483_647)

        pre = preprocess_reference_image(reference_image, target_size=256)
        self._ensure_backend()

        warnings = list(pre.warnings)
        for warning in self._load_warnings:
            if warning not in warnings:
                warnings.append(warning)

        if self.backend == "diffusers" and self._pipe is not None:
            images = self._generate_with_diffusers(pre.image, prompt, int(seed), float(creativity))
        else:
            images = self._generate_with_fallback(pre.image, prompt, int(seed))

        elapsed = time.perf_counter() - start
        return GenerationResult(
            images=images,
            warnings=warnings,
            backend=self.backend,
            latency_seconds=elapsed,
        )


_DEFAULT_PIPELINE: Optional[SpriteGenerationPipeline] = None


def _get_default_pipeline() -> SpriteGenerationPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = SpriteGenerationPipeline()
    return _DEFAULT_PIPELINE


def generate_states(
    reference_image: Image.Image,
    prompt: str,
    seed: Optional[int] = None,
    creativity: float = 0.35,
) -> Dict[str, Image.Image]:
    """Public inference API that returns exactly 4 state images."""
    result = _get_default_pipeline().generate(
        reference_image=reference_image,
        prompt=prompt,
        seed=seed,
        creativity=creativity,
    )
    return result.images


def generate_states_with_meta(
    reference_image: Image.Image,
    prompt: str,
    seed: Optional[int] = None,
    creativity: float = 0.35,
) -> GenerationResult:
    """Extended API with warnings/backend/latency metadata."""
    return _get_default_pipeline().generate(
        reference_image=reference_image,
        prompt=prompt,
        seed=seed,
        creativity=creativity,
    )
