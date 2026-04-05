# SpriteAI V3 (Text -> Sprite) + V2 (Photo -> Sprite)

SpriteAI now has two additive local pipelines:

- **V3 (new primary)**: text prompt -> one pixel character sprite
- **V2 (existing)**: photo-conditioned -> one pixel character sprite

Both run locally and can be used without paid APIs.

## Is This Local And Free?

Yes:

- No paid API is required.
- No OpenAI API key is required.
- Training and inference run locally on your machine.

Note:

- First model load may download open-source weights from Hugging Face.
- GPU is strongly recommended for training.

## 1) Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Build Canonical Character Dataset (V2 Source)

Input expected:

- `spriteai/dataset/0/*.png`
- `spriteai/dataset/1/*.png`
- `spriteai/dataset/2/*.png`
- `spriteai/dataset/3/*.png`

View mapping:

- `2=front`
- `1=right`
- `3=left`
- `0=back`

Build:

```bash
python -m spriteai.train.build_character_dataset \
  --input_dir ./spriteai/dataset \
  --out_dir ./data/character_dataset_v2 \
  --val_ratio 0.1 \
  --seed 42
```

## 3) Build V3 Text-Caption Dataset

This generates natural-language prompts for every sprite and writes:

- `data/text_sprite_dataset_v3/images/*.png`
- `data/text_sprite_dataset_v3/all_metadata.jsonl`

Build:

```bash
python -m spriteai.train.build_text_sprite_dataset_v3 \
  --input_dir ./data/character_dataset_v2 \
  --out_dir ./data/text_sprite_dataset_v3 \
  --metadata_file all_metadata.jsonl
```

## 4) Train V3 Full UNet (Text-Only)

This trains on **all rows** by default and saves preview grids during training.

```bash
python -m spriteai.train.train_character_text_v3 \
  --dataset_dir ./data/text_sprite_dataset_v3 \
  --output_dir ./artifacts/sd15_character_text_v3 \
  --model_id runwayml/stable-diffusion-v1-5 \
  --metadata_file all_metadata.jsonl \
  --resolution 256 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --epochs 12 \
  --learning_rate 2e-5 \
  --precision fp16 \
  --num_workers 0 \
  --preview_every_steps 100 \
  --preview_seeds 7,17,27,37,47
```

Training previews are written to:

- `artifacts/sd15_character_text_v3/previews/step_XXXXXX.png`
- `artifacts/sd15_character_text_v3/previews/step_XXXXXX.json`

## 5) Run App

```powershell
$env:SPRITEAI_TEXT_V3_MODEL_DIR=".\artifacts\sd15_character_text_v3"
$env:SPRITEAI_TEXT_V3_MODEL_INPUT_SIZE="384"
$env:SPRITEAI_TEXT_V3_CANDIDATES="2"
python -m spriteai.app.gradio_app
```

Open `http://127.0.0.1:7860`.

Use tab:

- **V3 Text -> Sprite** for prompt-only generation

Other tabs still available:

- **Character V2 (Default)** (photo-conditioned)
- **Legacy 4 States**

## 6) V3 Python API

```python
from spriteai.infer.text_sprite_pipeline import generate_text_sprite

sprite = generate_text_sprite(
    prompt="front-facing pixel character sprite, black hair, round glasses, green hoodie",
    seed=7,
    creativity=0.14,
    view="front",
)
sprite.save("text_v3_sprite_64.png")
```

Extended API:

```python
from spriteai.infer.text_sprite_pipeline import generate_text_sprite_with_meta

result = generate_text_sprite_with_meta(
    prompt="left-facing pixel character sprite, brown hair, tan skin tone, white top",
    seed=12,
)
print(result.backend, result.model_status, result.latency_seconds)
result.preview_image.save("text_v3_sprite_preview_512.png")
```

## 7) V3 Environment Variables

- `SPRITEAI_TEXT_V3_MODEL_ID` (default: `runwayml/stable-diffusion-v1-5`)
- `SPRITEAI_TEXT_V3_MODEL_DIR` (default: `./artifacts/sd15_character_text_v3`)
- `SPRITEAI_TEXT_V3_MODEL_INPUT_SIZE` (default: `384`)
- `SPRITEAI_TEXT_V3_CANDIDATES` (default: `2`)
- `SPRITEAI_TEXT_V3_FORCE_FALLBACK` (`1` to force fallback renderer)

Quick V3 run-info checks:

- `Backend: diffusers`
- `Model Status: loaded finetuned UNet ...`
- `Candidate count: 2`

## V2 and Legacy (Kept)

- V2 API: `spriteai.infer.character_pipeline.generate_character_sprite(...)`
- Legacy API: `spriteai.infer.pipeline.generate_states(...)`
- Existing V2 training command remains available:
  `python -m spriteai.train.train_character_lora ...`

