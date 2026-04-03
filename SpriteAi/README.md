# SpriteAI MVP

Local-first MVP for generating a tiny Tamagotchi-like pixel character from a reference image + prompt, with 4 states:

- `eating`
- `feeding` (being fed by owner/hand)
- `sleeping`
- `hygiene`

Outputs are four separate `32x32` PNG files with a fixed 16-color palette.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the local Gradio app

```bash
python -m spriteai.app.gradio_app
```

Open `http://127.0.0.1:7860`.

### Model configuration

By default, inference tries SD1.5:

- `SPRITEAI_MODEL_ID` (default: `runwayml/stable-diffusion-v1-5`)
- `SPRITEAI_LORA_PATH` (optional path to trained LoRA directory)

Optional IP-Adapter support:

- `SPRITEAI_IP_ADAPTER_REPO` (for example `h94/IP-Adapter`)
- `SPRITEAI_IP_ADAPTER_WEIGHT` (for example `ip-adapter_sd15.bin`)
- `SPRITEAI_IP_ADAPTER_SUBFOLDER` (default: `models`)
- `SPRITEAI_IP_ADAPTER_SCALE` (default: `0.8`)

To force deterministic fallback renderer (no diffusers/model download):

```bash
export SPRITEAI_FORCE_FALLBACK=1
```

## 3) Build synthetic training dataset

```bash
python -m spriteai.train.build_synth_dataset \
  --input_dir ./data/raw_refs \
  --out_dir ./data/synth_dataset \
  --val_ratio 0.1 \
  --seed 42
```

Generates:

- `data/synth_dataset/images/*.png`
- `data/synth_dataset/train_metadata.jsonl`
- `data/synth_dataset/val_metadata.jsonl`
- `data/synth_dataset/all_metadata.jsonl`

Prompts include explicit state tokens:

- `<state_eating>`
- `<state_feeding>`
- `<state_sleeping>`
- `<state_hygiene>`

## 4) Train LoRA (SD1.5)

```bash
python -m spriteai.train.train_lora \
  --dataset_dir ./data/synth_dataset \
  --output_dir ./artifacts/lora_sd15_spriteai \
  --model_id runwayml/stable-diffusion-v1-5 \
  --resolution 256 \
  --rank 16 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --epochs 5 \
  --learning_rate 1e-4 \
  --precision fp16
```

Then point inference to the LoRA:

```bash
export SPRITEAI_LORA_PATH=./artifacts/lora_sd15_spriteai
```

## 5) Evaluate acceptance checks

```bash
python -m spriteai.eval.eval_states \
  --input_dir ./data/eval_refs \
  --prompt "red scarf, cheerful eyes" \
  --seed 1234 \
  --out_file ./reports/eval_report.json
```

Checks include:

- output keys exactly `eating, feeding, sleeping, hygiene`
- each output exactly `32x32`
- each output `<=16` colors
- feeding/eating visual distinction heuristic
- median latency threshold check (`<=15s`)

## 6) Public API

```python
from PIL import Image
from spriteai.infer.pipeline import generate_states

img = Image.open("my_ref.png")
states = generate_states(img, prompt="red scarf", seed=7)
for name, state_img in states.items():
    state_img.save(f"{name}.png")
```

`generate_states(reference_image, prompt, seed=None, creativity=0.35) -> dict[str, PIL.Image.Image]`

Returns keys exactly:

- `eating`
- `feeding`
- `sleeping`
- `hygiene`

