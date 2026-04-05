"""Character-only LoRA training entrypoint for SpriteAI (SD1.5)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class TrainCharacterConfig:
    dataset_dir: Path
    output_dir: Path
    model_id: str
    resolution: int
    rank: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    max_train_steps: int
    precision: str
    num_workers: int
    front_view_weight: float
    seed: int


class JsonlCharacterDataset(Dataset):
    def __init__(self, dataset_dir: Path, metadata_file: str, resolution: int = 256):
        self.dataset_dir = dataset_dir
        self.records = self._load_jsonl(dataset_dir / metadata_file)
        self.resolution = resolution
        if not self.records:
            raise RuntimeError(f"No records found in {dataset_dir / metadata_file}")

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {path}")
        rows: List[Dict[str, str]] = []
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
        image_path = self.dataset_dir / record["image_path"]
        with Image.open(image_path) as img:
            rgb = img.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        arr = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return {
            "pixel_values": tensor,
            "prompt": str(record["prompt"]),
            "view": str(record.get("view", "")),
        }


class CharacterCollator:
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


def _build_sample_weights(records: List[Dict[str, str]], front_view_weight: float) -> List[float]:
    weights: List[float] = []
    for row in records:
        view = str(row.get("view", "")).strip().lower()
        weights.append(float(front_view_weight) if view == "front" else 1.0)
    return weights


def _parse_args() -> TrainCharacterConfig:
    parser = argparse.ArgumentParser(description="Train character-only SD1.5 LoRA for SpriteAI.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 means derive from epochs.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--front_view_weight",
        type=float,
        default=2.0,
        help="Sampling weight multiplier for rows where view=front.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainCharacterConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        resolution=args.resolution,
        rank=args.rank,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        precision=args.precision,
        num_workers=args.num_workers,
        front_view_weight=args.front_view_weight,
        seed=args.seed,
    )


def _weight_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


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
    try:
        import peft  # noqa: F401
    except Exception:
        missing.append("peft")

    if missing:
        raise RuntimeError(
            "Missing dependencies for training: "
            + ", ".join(missing)
            + ". Install via requirements.txt first."
        )


def train(config: TrainCharacterConfig) -> None:
    _assert_dependencies()
    from accelerate import Accelerator
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler

    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print(f"Training device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        if torch.version.cuda is None:
            print(
                "Warning: This torch build has no CUDA support, so training will run on CPU. "
                "Install a CUDA-enabled PyTorch build to use your NVIDIA GPU."
            )
        else:
            print(
                "Warning: CUDA is not available at runtime, so training will run on CPU. "
                "Check NVIDIA driver/toolkit setup."
            )

    accelerator = Accelerator(
        mixed_precision=config.precision if config.precision != "no" else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet.train()

    dataset = JsonlCharacterDataset(config.dataset_dir, "train_metadata.jsonl", resolution=config.resolution)
    collator = CharacterCollator(tokenizer)

    num_workers = int(config.num_workers)
    if sys.platform.startswith("win") and num_workers > 0:
        print(
            "Warning: forcing --num_workers=0 on Windows for training stability. "
            "Re-run on Linux if you want multi-worker dataloading."
        )
        num_workers = 0

    weights = _build_sample_weights(dataset.records, front_view_weight=float(config.front_view_weight))
    sampler = None
    shuffle = True
    if any(w != 1.0 for w in weights):
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
        front_count = sum(1 for row in dataset.records if str(row.get("view", "")).lower() == "front")
        print(
            json.dumps(
                {
                    "sampling": "weighted",
                    "front_view_weight": float(config.front_view_weight),
                    "front_rows": int(front_count),
                    "total_rows": int(len(dataset.records)),
                }
            )
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps > 0:
        max_train_steps = config.max_train_steps
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        num_train_epochs = config.epochs
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

    dtype = _weight_dtype(config.precision)
    text_encoder.to(accelerator.device, dtype=dtype)
    vae.to(accelerator.device, dtype=dtype)
    text_encoder.eval()
    vae.eval()

    global_step = 0
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
                                "loss": round(float(loss.detach().item()), 6),
                                "lr": lr,
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
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(config.output_dir),
            unet_lora_layers=lora_state_dict,
        )
        with (config.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        print(f"Saved LoRA weights to {config.output_dir}")


def main() -> None:
    config = _parse_args()
    train(config)


if __name__ == "__main__":
    main()
