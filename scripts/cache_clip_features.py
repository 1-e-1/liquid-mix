from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import yaml

from download_dataset import prepare_dataset
from liquid_mix.data import load_jsonl
from run_clip_vlm_experiment import encode_split, load_clip, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_coco1k.yaml")
    parser.add_argument("--output", default="results/q1_clip_feature_cache")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    data_dir, _, _, _ = prepare_dataset(config, force=bool(args.force_data))
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    batch_size = int(args.batch_size or config["clip"]["batch_size"])
    samples = {split: load_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "val", "test")}

    model, processor = load_clip(
        config["clip"]["model_name"],
        bool(config["clip"].get("local_files_only", True)),
        device,
    )
    payload = {
        "model_name": config["clip"]["model_name"],
        "sizes": {split: len(split_samples) for split, split_samples in samples.items()},
    }
    for split in ("train", "val", "test"):
        payload[split] = encode_split(model, processor, samples[split], batch_size, device)
    torch.save(payload, out_dir / "clip_features.pt")
    print(f"output={out_dir / 'clip_features.pt'}")
    print(f"sizes={payload['sizes']}")


if __name__ == "__main__":
    main()
