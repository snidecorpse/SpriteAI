"""V3 text-only full UNet finetune trainer with periodic preview grids."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from spriteai.infer.character_postprocess import postprocess_character_candidate

DEFAULT_PREVIEW_PROMPTS: List[str] = [
    "front-facing pixel character sprite, black hair, fair skin tone, white top, full body, simple background",
    "front-facing pixel character sprite, brown hair, tan skin tone, blue top, full body, simple background",
    "right-facing pixel character sprite, blond hair, medium skin tone, red top, full body, simple background",
    "left-facing pixel character sprite, dark brown hair, deep skin tone, green top, full body, simple background",
    "back-facing pixel character sprite, black hair, tan skin tone, gray top, full body, simple background",
]

NEGATIVE_PREVIEW_PROMPT = (
    "photo, realistic, camera, blurry, multiple characters, twin characters, cluttered background, text"
)


@dataclass(frozen=True)
class TrainTextV3Config:
    dataset_dir: Path
    output_dir: Path
    model_id: str
    metadata_file: str
    resolution: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    max_train_steps: int
    precision: str
    num_workers: int
    seed: int
    preview_every_steps: int
    preview_prompts_file: str
    preview_seeds: str
    gradient_checkpointing: bool


class JsonlTextSpriteDataset(Dataset):
    def __init__(self, dataset_dir: Path, metadata_file: str, resolution: int = 256):
        self.dataset_dir = dataset_dir
        self.records = self._load_jsonl(dataset_dir / metadata_file)
        self.resolution = resolution
        if not self.records:
            raise RuntimeError(f"No records found in {dataset_dir / metadata_file}")

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {path}")
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[idx]
        image_path = self.dataset_dir / str(record["image_path"])
        with Image.open(image_path) as img:
            rgb = img.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        arr = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        prompt_v3 = str(record.get("prompt_v3", "")).strip()
        if not prompt_v3:
            prompt_v3 = "front-facing pixel character sprite, full body, simple background, single character"
        return {"pixel_values": tensor, "prompt": prompt_v3}


class TextCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).to(
            memory_format=torch.contiguous_format
        )
        prompts = [str(ex["prompt"]) for ex in examples]
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": pixel_values, "input_ids": tokenized.input_ids}


def _assert_dependencies() -> None:
    missing = []
    try:
        import accelerate  # noqa: F401
    except Exception:
        missing.append("accelerate")
    try:
        import diffusers  # noqa: F401
    except Exception:
        missing.append("diffusers")
    try:
        import transformers  # noqa: F401
    except Exception:
        missing.append("transformers")
    if missing:
        raise RuntimeError(
            "Missing dependencies for V3 training: "
            + ", ".join(missing)
            + ". Install requirements.txt first."
        )


def _weight_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _parse_preview_seeds(raw: str) -> List[int]:
    text = (raw or "").strip()
    if not text:
        return [7, 17, 27, 37, 47]
    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        out = [7, 17, 27, 37, 47]
    return out


def _load_preview_prompts(path: str) -> List[str]:
    clean = (path or "").strip()
    if not clean:
        return list(DEFAULT_PREVIEW_PROMPTS)
    file = Path(clean)
    if not file.exists():
        raise FileNotFoundError(f"Preview prompts file not found: {file}")
    prompts: List[str] = []
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise RuntimeError(f"Preview prompts file has no prompts: {file}")
    return prompts


def _make_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    if not images:
        raise ValueError("No images for grid.")
    n = len(images)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), (18, 16, 30))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img.convert("RGB"), (x, y))
    return grid


def _write_preview(
    *,
    output_dir: Path,
    step: int,
    loss_value: float,
    lr: float,
    prompts: List[str],
    seeds: List[int],
    images: List[Image.Image],
    elapsed_seconds: float,
) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step_{step:06d}"
    grid = _make_grid(images, cols=min(3, len(images)))
    grid_path = preview_dir / f"{stem}.png"
    grid.save(grid_path, format="PNG")
    meta = {
        "step": int(step),
        "loss": float(loss_value),
        "lr": float(lr),
        "elapsed_seconds": float(round(elapsed_seconds, 3)),
        "prompts": prompts,
        "seeds": seeds,
        "grid_path": str(grid_path),
    }
    with (preview_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _generate_preview_images(
    *,
    accelerator,
    unet,
    vae,
    text_encoder,
    tokenizer,
    base_scheduler,
    prompts: List[str],
    seeds: List[int],
    resolution: int,
) -> List[Image.Image]:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    unwrapped_unet = accelerator.unwrap_model(unet)
    was_training = bool(unwrapped_unet.training)
    unwrapped_unet.eval()
    images: List[Image.Image] = []

    with torch.inference_mode():
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unwrapped_unet,
            scheduler=DPMSolverMultistepScheduler.from_config(base_scheduler.config),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.to(accelerator.device)
        pipe.enable_attention_slicing()

        device_str = pipe.device.type
        vae_dtype = next(pipe.vae.parameters()).dtype
        for idx, prompt in enumerate(prompts):
            seed = int(seeds[idx % len(seeds)])
            generator = torch.Generator(device=device_str).manual_seed(seed)
            latent_out = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PREVIEW_PROMPT,
                num_inference_steps=30,
                guidance_scale=7.8,
                width=resolution,
                height=resolution,
                generator=generator,
                output_type="latent",
            ).images
            if not isinstance(latent_out, torch.Tensor):
                raise RuntimeError("Expected latent tensor from preview pipeline output.")
            latents = latent_out.to(device=pipe.device, dtype=vae_dtype)
            decoded = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            out = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            sprite = postprocess_character_candidate(out, sprite_size=64).image
            images.append(sprite.resize((512, 512), Image.Resampling.NEAREST))

        del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if was_training:
        unwrapped_unet.train()
    return images


def _parse_args() -> TrainTextV3Config:
    parser = argparse.ArgumentParser(description="Train V3 text-only full UNet finetune for SpriteAI.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--metadata_file", type=str, default="all_metadata.jsonl")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 means derive from epochs.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_every_steps", type=int, default=100)
    parser.add_argument(
        "--preview_prompts_file",
        type=str,
        default="",
        help="Optional text file with one preview prompt per line.",
    )
    parser.add_argument("--preview_seeds", type=str, default="7,17,27,37,47")
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Disable UNet gradient checkpointing.",
    )
    args = parser.parse_args()

    return TrainTextV3Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        metadata_file=args.metadata_file,
        resolution=args.resolution,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        precision=args.precision,
        num_workers=args.num_workers,
        seed=args.seed,
        preview_every_steps=max(0, int(args.preview_every_steps)),
        preview_prompts_file=args.preview_prompts_file,
        preview_seeds=args.preview_seeds,
        gradient_checkpointing=not bool(args.disable_gradient_checkpointing),
    )


def train(config: TrainTextV3Config) -> None:
    _assert_dependencies()
    from accelerate import Accelerator
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler

    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    precision = str(config.precision)
    if precision == "fp16" and not torch.cuda.is_available():
        print("Warning: fp16 requested but CUDA unavailable; switching precision to 'no'.")
        precision = "no"

    if torch.cuda.is_available():
        print(f"Training device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        if torch.version.cuda is None:
            print(
                "Warning: This torch build has no CUDA support, so training runs on CPU. "
                "Install CUDA-enabled PyTorch to use your NVIDIA GPU."
            )
        else:
            print(
                "Warning: CUDA is not available at runtime, so training runs on CPU. "
                "Check NVIDIA driver/toolkit setup."
            )

    accelerator = Accelerator(
        mixed_precision=precision if precision != "no" else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(True)
    if config.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    unet.train()

    dataset = JsonlTextSpriteDataset(
        dataset_dir=config.dataset_dir,
        metadata_file=config.metadata_file,
        resolution=config.resolution,
    )
    collator = TextCollator(tokenizer)

    num_workers = int(config.num_workers)
    if sys.platform.startswith("win") and num_workers > 0:
        print(
            "Warning: forcing --num_workers=0 on Windows for training stability. "
            "Re-run on Linux if you want multi-worker dataloading."
        )
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps > 0:
        max_train_steps = int(config.max_train_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        num_train_epochs = int(config.epochs)
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=max(10, max_train_steps // 20),
        num_training_steps=max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        dataloader,
        lr_scheduler,
    )

    dtype = _weight_dtype(precision)
    text_encoder.to(accelerator.device, dtype=dtype)
    vae.to(accelerator.device, dtype=dtype)
    text_encoder.eval()
    vae.eval()

    preview_prompts = _load_preview_prompts(config.preview_prompts_file)
    preview_seeds = _parse_preview_seeds(config.preview_seeds)
    train_start = time.perf_counter()

    global_step = 0
    last_loss = 0.0
    for epoch in range(num_train_epochs):
        for _, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                last_loss = float(loss.detach().item())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 20 == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "loss": round(float(last_loss), 6),
                                "lr": lr,
                            }
                        )
                    )

                if (
                    accelerator.is_main_process
                    and config.preview_every_steps > 0
                    and global_step % config.preview_every_steps == 0
                ):
                    lr = lr_scheduler.get_last_lr()[0]
                    preview_images = _generate_preview_images(
                        accelerator=accelerator,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        base_scheduler=noise_scheduler,
                        prompts=preview_prompts,
                        seeds=preview_seeds,
                        resolution=config.resolution,
                    )
                    _write_preview(
                        output_dir=config.output_dir,
                        step=global_step,
                        loss_value=last_loss,
                        lr=float(lr),
                        prompts=preview_prompts,
                        seeds=preview_seeds,
                        images=preview_images,
                        elapsed_seconds=time.perf_counter() - train_start,
                    )
                    print(
                        json.dumps(
                            {
                                "preview_step": global_step,
                                "preview_images": len(preview_images),
                                "preview_dir": str(config.output_dir / "previews"),
                            }
                        )
                    )

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_out = config.output_dir / "unet"
        unwrapped_unet.save_pretrained(unet_out)
        with (config.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        with (config.output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_step": int(global_step),
                    "final_loss": float(last_loss),
                    "elapsed_seconds": float(round(time.perf_counter() - train_start, 3)),
                    "unet_dir": str(unet_out),
                },
                f,
                indent=2,
            )
        print(f"Saved V3 UNet weights to {unet_out}")


def main() -> None:
    cfg = _parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
