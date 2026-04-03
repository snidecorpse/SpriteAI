"""Local Gradio app for SpriteAI inference."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

from spriteai.infer.pipeline import generate_states_with_meta
from spriteai.infer.state_prompts import STATE_KEYS


def _save_outputs(images: dict[str, Image.Image]) -> dict[str, Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="spriteai_"))
    saved: dict[str, Path] = {}
    for state in STATE_KEYS:
        path = out_dir / f"{state}.png"
        images[state].save(path, format="PNG")
        saved[state] = path
    return saved


def _build_preview_gallery(images: dict[str, Image.Image]) -> List[Tuple[Image.Image, str]]:
    return [(images[state], state) for state in STATE_KEYS]


def run_generation(
    image: Optional[Image.Image],
    prompt: str,
    seed: Optional[float],
    creativity: float,
):
    if image is None:
        raise gr.Error("Please upload a reference image.")

    clean_seed: Optional[int] = None
    if seed is not None and not (isinstance(seed, float) and math.isnan(seed)):
        clean_seed = int(seed)

    try:
        result = generate_states_with_meta(
            reference_image=image,
            prompt=prompt or "",
            seed=clean_seed,
            creativity=creativity,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime/model environment dependent
        raise gr.Error(f"Generation failed: {exc}") from exc

    saved = _save_outputs(result.images)
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
    with gr.Blocks(title="SpriteAI") as demo:
        gr.Markdown(
            "# SpriteAI MVP\n"
            "Upload a reference image and optional prompt modifications, then generate 4 Tamagotchi-like pixel states."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Reference Image", type="pil")
                prompt_input = gr.Textbox(
                    label="Prompt Modifications",
                    placeholder="e.g. red scarf, happy eyes, tiny backpack",
                    lines=2,
                )
                seed_input = gr.Number(
                    label="Seed (optional)",
                    value=None,
                    precision=0,
                )
                creativity_input = gr.Slider(
                    label="Creativity (controls state edit strength)",
                    minimum=0.3,
                    maximum=0.45,
                    value=0.35,
                    step=0.01,
                )
                run_btn = gr.Button("Generate 4 States", variant="primary")

            with gr.Column():
                preview_output = gr.Gallery(label="Preview Grid", columns=2, rows=2, height=320)
                eating_file = gr.File(label="Download: eating.png")
                feeding_file = gr.File(label="Download: feeding.png")
                sleeping_file = gr.File(label="Download: sleeping.png")
                hygiene_file = gr.File(label="Download: hygiene.png")
                meta_text = gr.Textbox(label="Run Info", lines=5)

        run_btn.click(
            fn=run_generation,
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
