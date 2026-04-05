"""Character-first generation pipeline for one sprite output."""

from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

from .character_features import CharacterFeatureProfile, extract_character_features
from .character_postprocess import choose_best_character_candidate, postprocess_character_candidate
from .character_preprocess import preprocess_character_reference_image
from .character_prompts import NEGATIVE_CHARACTER_PROMPT, build_character_prompt, normalize_view


@dataclass(frozen=True)
class CharacterGenerationResult:
    image: Image.Image
    preview_image: Image.Image
    warnings: List[str]
    backend: str
    latency_seconds: float
    crop_mode: str
    feature_summary: str
    lora_status: str
    sprite_size: int
    preview_size: int


def _hash_to_seed(base_seed: int, text: str) -> int:
    payload = f"{base_seed}:{text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_default_lora_path(configured_path: Optional[str]) -> tuple[Optional[str], bool]:
    if configured_path:
        clean = configured_path.strip()
        if clean:
            return clean, False

    candidate = Path("artifacts") / "lora_sd15_character_v2"
    weight_file = candidate / "pytorch_lora_weights.safetensors"
    if weight_file.exists():
        return str(candidate), True

    return None, False


def _skin_rgb(label: str) -> tuple[int, int, int]:
    return {
        "deep skin tone": (108, 78, 60),
        "medium skin tone": (154, 112, 86),
        "tan skin tone": (192, 146, 114),
        "fair skin tone": (224, 188, 162),
    }.get(label, (170, 130, 100))


def _hair_rgb(label: str) -> tuple[int, int, int]:
    return {
        "black hair": (30, 28, 34),
        "dark brown hair": (58, 42, 34),
        "brown hair": (92, 66, 48),
        "light brown hair": (130, 96, 72),
        "blond hair": (196, 168, 96),
    }.get(label, (62, 44, 36))


def _top_rgb(label: str) -> tuple[int, int, int]:
    return {
        "black top": (32, 34, 40),
        "gray top": (92, 94, 102),
        "light gray top": (148, 152, 165),
        "white top": (222, 226, 230),
        "red top": (190, 72, 84),
        "orange top": (208, 122, 64),
        "yellow top": (208, 176, 74),
        "green top": (78, 142, 88),
        "emerald top": (56, 148, 116),
        "mint top": (106, 178, 162),
        "blue top": (80, 120, 192),
        "indigo top": (84, 86, 168),
        "purple top": (126, 88, 166),
        "pink top": (196, 120, 172),
    }.get(label, (96, 126, 180))


def _build_conditioned_prompt(
    user_prompt: str,
    view: str,
    feature_profile: CharacterFeatureProfile,
) -> str:
    parts = list(feature_profile.prompt_fragments)
    clean_user = " ".join((user_prompt or "").strip().split())
    if clean_user:
        parts.append(clean_user)
    return build_character_prompt(", ".join(parts), view=view)


def _nearest_preview(image: Image.Image, scale: int) -> Image.Image:
    clean_scale = max(1, int(scale))
    if clean_scale == 1:
        return image.copy()
    return image.resize((image.size[0] * clean_scale, image.size[1] * clean_scale), Image.Resampling.NEAREST)


def _render_fallback_character(
    feature_profile: CharacterFeatureProfile,
    seed: int,
    view: str,
) -> Image.Image:
    rng = random.Random(seed)
    skin = _skin_rgb(feature_profile.skin_tone)
    hair = _hair_rgb(feature_profile.hair_tone)
    cloth = _top_rgb(feature_profile.clothing_color)
    accent = tuple(max(0, min(255, int(c * 0.58))) for c in cloth)
    outline = (40, 38, 48)

    canvas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.ellipse((86, 34, 170, 122), fill=skin, outline=outline, width=4)
    draw.pieslice((80, 24, 176, 120), start=180, end=360, fill=hair, outline=hair, width=2)
    draw.rounded_rectangle((84, 110, 172, 204), radius=20, fill=cloth, outline=outline, width=4)
    draw.rounded_rectangle((58, 122, 88, 190), radius=12, fill=skin, outline=outline, width=3)
    draw.rounded_rectangle((168, 122, 198, 190), radius=12, fill=skin, outline=outline, width=3)
    draw.rounded_rectangle((96, 194, 126, 240), radius=10, fill=skin, outline=outline, width=3)
    draw.rounded_rectangle((130, 194, 160, 240), radius=10, fill=skin, outline=outline, width=3)

    eye_y = 78
    if view == "front":
        eye_positions = (112, 144)
    elif view == "left":
        eye_positions = (106,)
    elif view == "right":
        eye_positions = (150,)
    else:
        eye_positions = ()

    for cx in eye_positions:
        draw.ellipse((cx - 8, eye_y - 6, cx + 8, eye_y + 6), fill=(244, 244, 248), outline=outline, width=2)
        draw.ellipse((cx - 2, eye_y - 1, cx + 2, eye_y + 1), fill=(32, 32, 42))

    if feature_profile.has_glasses and eye_positions:
        for cx in eye_positions:
            draw.ellipse((cx - 11, eye_y - 9, cx + 11, eye_y + 9), outline=accent, width=2)
        if len(eye_positions) == 2:
            draw.line((eye_positions[0] + 11, eye_y, eye_positions[1] - 11, eye_y), fill=accent, width=2)

    if view == "back":
        draw.rounded_rectangle((94, 62, 162, 102), radius=12, fill=hair, outline=hair, width=1)
    if rng.random() > 0.5:
        draw.rounded_rectangle((102, 136, 154, 158), radius=8, fill=tuple(min(255, c + 15) for c in cloth))
    return canvas


class CharacterSpritePipeline:
    """Generate one sprite character from photo features and optional prompt."""

    def __init__(self, model_id: Optional[str] = None, lora_path: Optional[str] = None):
        self.model_id = model_id or os.getenv(
            "SPRITEAI_CHARACTER_MODEL_ID",
            os.getenv("SPRITEAI_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        )
        configured_lora = lora_path or os.getenv("SPRITEAI_CHARACTER_LORA_PATH", os.getenv("SPRITEAI_LORA_PATH"))
        self.lora_path, self._lora_auto_detected = _resolve_default_lora_path(configured_lora)
        self.lora_scale = float(
            os.getenv("SPRITEAI_CHARACTER_LORA_SCALE", os.getenv("SPRITEAI_LORA_SCALE", "1.50")),
        )
        self.model_input_size = int(
            os.getenv("SPRITEAI_CHARACTER_MODEL_INPUT_SIZE", os.getenv("SPRITEAI_MODEL_INPUT_SIZE", "384")),
        )
        self.candidate_count = max(1, int(os.getenv("SPRITEAI_CHARACTER_CANDIDATES", "2")))
        self.force_fallback = os.getenv(
            "SPRITEAI_CHARACTER_FORCE_FALLBACK",
            os.getenv("SPRITEAI_FORCE_FALLBACK", ""),
        ).lower() in {"1", "true", "yes"}

        self._pipe = None
        self._torch = None
        self._backend = "uninitialized"
        self._backend_initialized = False
        self._load_warnings: List[str] = []
        self._lora_loaded = False
        self._lora_load_error: Optional[str] = None
        self._lora_status = "not configured"
        if self._lora_auto_detected and self.lora_path:
            self._load_warnings.append(f"Auto-detected character LoRA path: {self.lora_path}")

    @property
    def backend(self) -> str:
        return self._backend

    def _load_diffusers_backend(self) -> bool:
        if self.force_fallback:
            self._load_warnings.append("SPRITEAI_CHARACTER_FORCE_FALLBACK enabled; using deterministic fallback backend.")
            return False

        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except Exception as exc:  # pragma: no cover
            self._load_warnings.append(f"Diffusers backend not available: {exc}")
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
            pipe.to(device)
            pipe.enable_attention_slicing()

            if self.lora_path:
                try:
                    pipe.load_lora_weights(self.lora_path)
                    self._lora_loaded = True
                    self._lora_status = f"loaded from: {self.lora_path}"
                except Exception as exc:
                    self._lora_load_error = str(exc)
                    self._lora_status = f"configured but failed to load: {self.lora_path}"
                    self._load_warnings.append(f"Failed to load LoRA from '{self.lora_path}': {exc}")
            else:
                self._lora_status = "not configured (base model only)"

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
        prompt: str,
        seed: int,
        creativity: float,
        sprite_size: int,
        lora_scale: float,
    ) -> Image.Image:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("Diffusers pipeline is not initialized.")

        torch = self._torch
        device = self._pipe.device.type
        creativity = _clamp(creativity, 0.1, 0.45)
        guidance_scale = _clamp(9.0 - creativity * 3.0, 7.4, 9.0)
        num_steps = int(round(42 + creativity * 20))
        common = {
            "negative_prompt": NEGATIVE_CHARACTER_PROMPT,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "width": self.model_input_size,
            "height": self.model_input_size,
            "num_images_per_prompt": int(self.candidate_count),
        }
        if self._lora_loaded:
            common["cross_attention_kwargs"] = {"scale": float(lora_scale)}

        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = self._pipe(
            prompt=prompt,
            generator=generator,
            **common,
        ).images
        if not outputs:
            raise RuntimeError("Diffusers returned no candidate images.")
        best = choose_best_character_candidate(outputs, sprite_size=sprite_size)
        return best.image

    def _generate_with_fallback(
        self,
        feature_profile: CharacterFeatureProfile,
        prompt: str,
        seed: int,
        view: str,
        sprite_size: int,
    ) -> Image.Image:
        char_seed = _hash_to_seed(seed, f"{prompt}:{feature_profile.summary}:{view}")
        out = _render_fallback_character(feature_profile, seed=char_seed, view=view)
        self._lora_status = "ignored on fallback backend"
        return postprocess_character_candidate(out, sprite_size=sprite_size).image

    def generate(
        self,
        reference_image: Image.Image,
        prompt: str = "",
        seed: Optional[int] = None,
        creativity: float = 0.18,
        view: str = "front",
        sprite_size: int = 64,
        preview_scale: int = 8,
        lora_scale: Optional[float] = None,
    ) -> CharacterGenerationResult:
        if not isinstance(reference_image, Image.Image):
            raise ValueError("Expected a valid image input.")

        start = time.perf_counter()
        if seed is None:
            seed = random.randint(0, 2_147_483_647)
        clean_view = normalize_view(view)
        effective_lora_scale = float(self.lora_scale if lora_scale is None else lora_scale)
        clean_sprite_size = max(16, int(sprite_size))
        clean_preview_scale = max(1, int(preview_scale))

        pre = preprocess_character_reference_image(reference_image, target_size=self.model_input_size)
        feature_profile = extract_character_features(pre.image)
        conditioned_prompt = _build_conditioned_prompt(prompt, clean_view, feature_profile)
        self._ensure_backend()

        warnings = list(pre.warnings)
        for warning in self._load_warnings:
            if warning not in warnings:
                warnings.append(warning)

        if self.backend == "diffusers" and self._torch is not None and not bool(self._torch.cuda.is_available()):
            msg = "Diffusers backend is running on CPU. Install CUDA-enabled PyTorch to use your NVIDIA GPU."
            if msg not in warnings:
                warnings.append(msg)
        if self.backend == "fallback" and self.lora_path:
            msg = f"LoRA path is set but fallback backend is active: {self.lora_path}"
            if msg not in warnings:
                warnings.append(msg)
        if self.backend == "diffusers" and self.lora_path and self._lora_loaded:
            self._lora_status = f"loaded from: {self.lora_path} (scale={effective_lora_scale:.2f})"
        elif self.backend == "diffusers" and self.lora_path and not self._lora_loaded:
            self._lora_status = f"configured but failed to load: {self.lora_path}"
        elif self.backend == "diffusers":
            self._lora_status = "not configured (base model only)"

        crop_msg = f"Crop mode: {pre.crop_mode}"
        if crop_msg not in warnings:
            warnings.append(crop_msg)
        candidate_msg = f"Candidate count: {self.candidate_count}"
        if candidate_msg not in warnings:
            warnings.append(candidate_msg)

        if self.backend == "diffusers" and self._pipe is not None:
            image = self._generate_with_diffusers(
                prompt=conditioned_prompt,
                seed=int(seed),
                creativity=float(creativity),
                sprite_size=clean_sprite_size,
                lora_scale=effective_lora_scale,
            )
        else:
            image = self._generate_with_fallback(
                feature_profile=feature_profile,
                prompt=conditioned_prompt,
                seed=int(seed),
                view=clean_view,
                sprite_size=clean_sprite_size,
            )

        preview = _nearest_preview(image, clean_preview_scale)
        elapsed = time.perf_counter() - start
        return CharacterGenerationResult(
            image=image,
            preview_image=preview,
            warnings=warnings,
            backend=self.backend,
            latency_seconds=elapsed,
            crop_mode=pre.crop_mode,
            feature_summary=feature_profile.summary,
            lora_status=self._lora_status,
            sprite_size=clean_sprite_size,
            preview_size=preview.size[0],
        )


_DEFAULT_CHARACTER_PIPELINE: Optional[CharacterSpritePipeline] = None


def _get_default_pipeline() -> CharacterSpritePipeline:
    global _DEFAULT_CHARACTER_PIPELINE
    if _DEFAULT_CHARACTER_PIPELINE is None:
        _DEFAULT_CHARACTER_PIPELINE = CharacterSpritePipeline()
    return _DEFAULT_CHARACTER_PIPELINE


def generate_character_sprite(
    reference_image: Image.Image,
    prompt: str = "",
    seed: Optional[int] = None,
    creativity: float = 0.18,
    view: str = "front",
    sprite_size: int = 64,
    lora_scale: Optional[float] = None,
) -> Image.Image:
    result = _get_default_pipeline().generate(
        reference_image=reference_image,
        prompt=prompt,
        seed=seed,
        creativity=creativity,
        view=view,
        sprite_size=sprite_size,
        lora_scale=lora_scale,
    )
    return result.image


def generate_character_sprite_with_meta(
    reference_image: Image.Image,
    prompt: str = "",
    seed: Optional[int] = None,
    creativity: float = 0.18,
    view: str = "front",
    sprite_size: int = 64,
    preview_scale: int = 8,
    lora_scale: Optional[float] = None,
) -> CharacterGenerationResult:
    return _get_default_pipeline().generate(
        reference_image=reference_image,
        prompt=prompt,
        seed=seed,
        creativity=creativity,
        view=view,
        sprite_size=sprite_size,
        preview_scale=preview_scale,
        lora_scale=lora_scale,
    )
