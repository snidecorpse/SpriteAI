"""Local Gradio app for SpriteAI inference."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

from spriteai.infer.character_pipeline import generate_character_sprite_with_meta
from spriteai.infer.character_prompts import CHARACTER_VIEWS
from spriteai.infer.pipeline import generate_states_with_meta
from spriteai.infer.state_prompts import STATE_KEYS
from spriteai.infer.text_sprite_pipeline import generate_text_sprite_with_meta


def _save_character_outputs(image: Image.Image, preview_image: Image.Image) -> tuple[Path, Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="spriteai_char_"))
    sprite_path = out_dir / "character_64.png"
    preview_path = out_dir / "character_preview_512.png"
    image.save(sprite_path, format="PNG")
    preview_image.save(preview_path, format="PNG")
    return sprite_path, preview_path


def _save_text_v3_outputs(image: Image.Image, preview_image: Image.Image) -> tuple[Path, Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="spriteai_text_v3_"))
    sprite_path = out_dir / "text_v3_sprite_64.png"
    preview_path = out_dir / "text_v3_sprite_preview_512.png"
    image.save(sprite_path, format="PNG")
    preview_image.save(preview_path, format="PNG")
    return sprite_path, preview_path


def _save_legacy_outputs(images: dict[str, Image.Image]) -> dict[str, Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="spriteai_legacy_"))
    saved: dict[str, Path] = {}
    for state in STATE_KEYS:
        path = out_dir / f"{state}.png"
        images[state].save(path, format="PNG")
        saved[state] = path
    return saved


def _build_preview_gallery(images: dict[str, Image.Image]) -> List[Tuple[Image.Image, str]]:
    return [(images[state], state) for state in STATE_KEYS]


def _clean_seed(seed: Optional[float]) -> Optional[int]:
    if seed is None or (isinstance(seed, float) and math.isnan(seed)):
        return None
    return int(seed)


def run_character_generation(
    image: Optional[Image.Image],
    prompt: str,
    seed: Optional[float],
    creativity: float,
    view: str,
    lora_scale: float,
):
    if image is None:
        raise gr.Error("Please upload a reference image.")

    try:
        result = generate_character_sprite_with_meta(
            reference_image=image,
            prompt=prompt or "",
            seed=_clean_seed(seed),
            creativity=float(creativity),
            view=str(view),
            lora_scale=float(lora_scale),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise gr.Error(f"Generation failed: {exc}") from exc

    sprite_path, preview_path = _save_character_outputs(result.image, result.preview_image)
    warnings = "\n".join(result.warnings) if result.warnings else "None"
    info = (
        f"Backend: {result.backend}\n"
        f"Latency: {result.latency_seconds:.2f}s\n"
        f"LoRA: {result.lora_status}\n"
        f"Crop Mode: {result.crop_mode}\n"
        f"Features: {result.feature_summary}\n"
        f"Sprite Size: {result.sprite_size}x{result.sprite_size}\n"
        f"Preview Size: {result.preview_size}x{result.preview_size}\n"
        f"Canonical Sprite File: {sprite_path}\n"
        f"Warnings: {warnings}"
    )
    return result.preview_image, str(preview_path), info


def run_text_v3_generation(
    prompt: str,
    seed: Optional[float],
    creativity: float,
    view: str,
    candidate_count: float,
):
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise gr.Error("Please enter a text prompt.")

    try:
        result = generate_text_sprite_with_meta(
            prompt=clean_prompt,
            seed=_clean_seed(seed),
            creativity=float(creativity),
            view=str(view),
            candidate_count=max(1, int(candidate_count)),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise gr.Error(f"Generation failed: {exc}") from exc

    sprite_path, preview_path = _save_text_v3_outputs(result.image, result.preview_image)
    warnings = "\n".join(result.warnings) if result.warnings else "None"
    info = (
        f"Backend: {result.backend}\n"
        f"Latency: {result.latency_seconds:.2f}s\n"
        f"Model Status: {result.model_status}\n"
        f"Sprite Size: {result.sprite_size}x{result.sprite_size}\n"
        f"Preview Size: {result.preview_size}x{result.preview_size}\n"
        f"Canonical Sprite File: {sprite_path}\n"
        f"Warnings: {warnings}"
    )
    return result.preview_image, str(preview_path), info


def run_legacy_generation(
    image: Optional[Image.Image],
    prompt: str,
    seed: Optional[float],
    creativity: float,
):
    if image is None:
        raise gr.Error("Please upload a reference image.")

    try:
        result = generate_states_with_meta(
            reference_image=image,
            prompt=prompt or "",
            seed=_clean_seed(seed),
            creativity=float(creativity),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise gr.Error(f"Generation failed: {exc}") from exc

    saved = _save_legacy_outputs(result.images)
    preview = _build_preview_gallery(result.images)
    warnings = "\n".join(result.warnings) if result.warnings else "None"
    info = (
        f"Backend: {result.backend}\n"
        f"Latency: {result.latency_seconds:.2f}s\n"
        f"Warnings: {warnings}"
    )
    return (
        preview,
        str(saved["eating"]),
        str(saved["feeding"]),
        str(saved["sleeping"]),
        str(saved["hygiene"]),
        info,
    )


def build_app() -> gr.Blocks:
    default_lora_scale = float(os.getenv("SPRITEAI_CHARACTER_LORA_SCALE", os.getenv("SPRITEAI_LORA_SCALE", "1.50")))
    default_text_candidates = max(1, int(os.getenv("SPRITEAI_TEXT_V3_CANDIDATES", "2")))
    with gr.Blocks(title="SpriteAI") as demo:
        gr.Markdown(
            "# SpriteAI\n"
            "V3 adds text-only sprite generation. "
            "V2 image-conditioned flow and legacy state flow remain available."
        )

        with gr.Tabs():
            with gr.Tab("V3 Text -> Sprite", id="text-v3"):
                with gr.Row():
                    with gr.Column():
                        text_prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g. front-facing pixel character sprite, black hair, round glasses, green hoodie",
                            lines=3,
                        )
                        text_seed_input = gr.Number(label="Seed (optional)", value=None, precision=0)
                        text_creativity_input = gr.Slider(
                            label="Creativity (style strength)",
                            minimum=0.08,
                            maximum=0.38,
                            value=0.14,
                            step=0.01,
                        )
                        with gr.Accordion("Advanced", open=False):
                            text_view_input = gr.Dropdown(
                                label="View",
                                choices=list(CHARACTER_VIEWS),
                                value="front",
                            )
                            text_candidate_count = gr.Slider(
                                label="Candidate Count",
                                minimum=1,
                                maximum=4,
                                value=default_text_candidates,
                                step=1,
                            )
                        text_run_btn = gr.Button("Generate V3 Text Sprite", variant="primary")

                    with gr.Column():
                        text_preview_output = gr.Image(label="V3 Preview", type="pil")
                        text_file_output = gr.File(label="Download: text_v3_sprite_preview_512.png")
                        text_meta_output = gr.Textbox(label="Run Info", lines=9)

                text_run_btn.click(
                    fn=run_text_v3_generation,
                    inputs=[
                        text_prompt_input,
                        text_seed_input,
                        text_creativity_input,
                        text_view_input,
                        text_candidate_count,
                    ],
                    outputs=[text_preview_output, text_file_output, text_meta_output],
                )

            with gr.Tab("Character V2 (Default)", id="character-v2"):
                with gr.Row():
                    with gr.Column():
                        char_image_input = gr.Image(label="Reference Photo", type="pil")
                        char_prompt_input = gr.Textbox(
                            label="Optional Prompt",
                            placeholder="e.g. round glasses, dark hair, mint sweatshirt",
                            lines=2,
                        )
                        char_seed_input = gr.Number(label="Seed (optional)", value=None, precision=0)
                        char_creativity_input = gr.Slider(
                            label="Creativity (style strength)",
                            minimum=0.1,
                            maximum=0.32,
                            value=0.14,
                            step=0.01,
                        )
                        with gr.Accordion("Advanced", open=False):
                            char_view_input = gr.Dropdown(
                                label="View",
                                choices=list(CHARACTER_VIEWS),
                                value="front",
                            )
                            char_lora_scale_input = gr.Slider(
                                label="LoRA Scale",
                                minimum=0.5,
                                maximum=2.0,
                                value=default_lora_scale,
                                step=0.05,
                            )
                        char_run_btn = gr.Button("Generate Character Sprite", variant="primary")

                    with gr.Column():
                        char_preview_output = gr.Image(label="Character Preview", type="pil")
                        char_file_output = gr.File(label="Download: character_preview_512.png")
                        char_meta_output = gr.Textbox(label="Run Info", lines=10)

                char_run_btn.click(
                    fn=run_character_generation,
                    inputs=[
                        char_image_input,
                        char_prompt_input,
                        char_seed_input,
                        char_creativity_input,
                        char_view_input,
                        char_lora_scale_input,
                    ],
                    outputs=[char_preview_output, char_file_output, char_meta_output],
                )

            with gr.Tab("Legacy 4 States", id="legacy-states"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Reference Image", type="pil")
                        prompt_input = gr.Textbox(
                            label="Prompt Modifications",
                            placeholder="e.g. red scarf, happy eyes, tiny backpack",
                            lines=2,
                        )
                        seed_input = gr.Number(label="Seed (optional)", value=None, precision=0)
                        creativity_input = gr.Slider(
                            label="Creativity (controls state edit strength)",
                            minimum=0.12,
                            maximum=0.45,
                            value=0.18,
                            step=0.01,
                        )
                        run_btn = gr.Button("Generate 4 States", variant="secondary")

                    with gr.Column():
                        preview_output = gr.Gallery(label="Preview Grid", columns=2, rows=2, height=320)
                        eating_file = gr.File(label="Download: eating.png")
                        feeding_file = gr.File(label="Download: feeding.png")
                        sleeping_file = gr.File(label="Download: sleeping.png")
                        hygiene_file = gr.File(label="Download: hygiene.png")
                        meta_text = gr.Textbox(label="Run Info", lines=5)

                run_btn.click(
                    fn=run_legacy_generation,
                    inputs=[image_input, prompt_input, seed_input, creativity_input],
                    outputs=[
                        preview_output,
                        eating_file,
                        feeding_file,
                        sleeping_file,
                        hygiene_file,
                        meta_text,
                    ],
                )
    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
