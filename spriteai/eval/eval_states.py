"""Evaluate state-generation outputs against MVP acceptance checks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageChops, ImageStat

from spriteai.infer.pipeline import generate_states_with_meta
from spriteai.infer.pixelize import count_unique_colors
from spriteai.infer.state_prompts import STATE_KEYS

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_inputs(input_dir: Path) -> List[Path]:
    return [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def _state_difference_score(a: Image.Image, b: Image.Image) -> float:
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    stat = ImageStat.Stat(diff)
    return float(sum(stat.mean) / 3.0)


def evaluate(input_dir: Path, prompt: str, seed: int, limit: int) -> Dict[str, object]:
    paths = _collect_inputs(input_dir)
    if not paths:
        raise RuntimeError(f"No input images found in {input_dir}")
    if limit > 0:
        paths = paths[:limit]

    runs: List[Dict[str, object]] = []
    latencies: List[float] = []

    for idx, path in enumerate(paths):
        with Image.open(path) as img:
            result = generate_states_with_meta(
                reference_image=img,
                prompt=prompt,
                seed=seed + idx,
                creativity=0.35,
            )
        outputs = result.images
        keys_ok = list(outputs.keys()) == list(STATE_KEYS)
        size_ok = all(outputs[s].size == (32, 32) for s in STATE_KEYS)
        color_ok = all(count_unique_colors(outputs[s]) <= 16 for s in STATE_KEYS)
        feed_vs_eat_delta = _state_difference_score(outputs["feeding"], outputs["eating"])
        distinct_ok = feed_vs_eat_delta >= 3.5

        run = {
            "source_image": str(path),
            "backend": result.backend,
            "latency_seconds": round(result.latency_seconds, 3),
            "warnings": result.warnings,
            "checks": {
                "keys_ok": keys_ok,
                "size_ok": size_ok,
                "color_ok": color_ok,
                "feeding_vs_eating_distinct_ok": distinct_ok,
                "feeding_vs_eating_delta": round(feed_vs_eat_delta, 3),
            },
        }
        runs.append(run)
        latencies.append(result.latency_seconds)

    summary = {
        "samples": len(runs),
        "prompt": prompt,
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "max_latency_seconds": round(max(latencies), 3),
        "acceptance": {
            "all_keys_ok": all(r["checks"]["keys_ok"] for r in runs),
            "all_size_ok": all(r["checks"]["size_ok"] for r in runs),
            "all_color_ok": all(r["checks"]["color_ok"] for r in runs),
            "all_feeding_eating_distinct_ok": all(
                r["checks"]["feeding_vs_eating_distinct_ok"] for r in runs
            ),
            "median_latency_le_15s": statistics.median(latencies) <= 15.0,
        },
        "runs": runs,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpriteAI state generation.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of reference images.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt modifications used for all eval runs.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images.")
    parser.add_argument("--out_file", type=Path, default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        input_dir=args.input_dir,
        prompt=args.prompt,
        seed=args.seed,
        limit=args.limit,
    )
    output = json.dumps(report, indent=2)
    print(output)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
