from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
)
import yaml

from download_dataset import prepare_dataset
from liquid_mix.clip_fusion import recall_at_k
from liquid_mix.data import CaptionSample, load_jsonl


DEFAULT_MODELS = {
    "clip_b32_openai": {
        "family": "clip",
        "path": "openai/clip-vit-base-patch32",
        "local_files_only": True,
    },
    "openclip_b32_laion2b": {
        "family": "clip",
        "path": "weights/laion--CLIP-ViT-B-32-laion2B-s34B-b79K",
        "local_files_only": True,
    },
    "siglip_b16_google": {
        "family": "siglip",
        "path": "weights/google--siglip-base-patch16-224",
        "local_files_only": True,
    },
    "blip_itm_coco": {
        "family": "blip_itm",
        "path": "weights/Salesforce--blip-itm-base-coco",
        "local_files_only": True,
    },
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_model_path(path: str) -> str:
    candidate = ROOT / path
    if candidate.exists():
        return str(candidate)
    return path


def load_embedding_model(spec: dict[str, Any], device: torch.device):
    family = spec["family"]
    model_path = resolve_model_path(spec["path"])
    local_files_only = bool(spec.get("local_files_only", True))
    if family == "clip":
        model = CLIPModel.from_pretrained(model_path, local_files_only=local_files_only).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    elif family == "siglip":
        model = SiglipModel.from_pretrained(model_path, local_files_only=local_files_only).to(device).eval()
        processor = SiglipProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    else:
        raise ValueError(f"Unsupported embedding model family: {family}")
    return model, processor


def encode_embedding_split(
    model,
    processor,
    family: str,
    samples: list[CaptionSample],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_features = []
    text_features = []
    for start in tqdm(range(0, len(samples), batch_size), desc=f"encode_{family}"):
        batch = samples[start : start + batch_size]
        images = [Image.open(sample.image_path).convert("RGB") for sample in batch]
        texts = [sample.caption for sample in batch]
        with torch.no_grad():
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
            if family == "clip":
                image_out = model.vision_model(**image_inputs)
                text_out = model.text_model(**text_inputs)
                image_emb = model.visual_projection(image_out.pooler_output)
                text_emb = model.text_projection(text_out.pooler_output)
            elif family == "siglip":
                # SigLIP's public forward applies the correct pooling and
                # embedding heads. Raw tower pooler outputs are not aligned.
                joint_inputs = processor(
                    images=images,
                    text=texts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                out = model(**joint_inputs)
                image_emb = out.image_embeds
                text_emb = out.text_embeds
            else:
                raise ValueError(f"Unsupported family: {family}")
        image_features.append(image_emb.detach().cpu().float())
        text_features.append(text_emb.detach().cpu().float())
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def embedding_score_matrix(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    return image_emb @ text_emb.T


def summarize_retrieval(score_matrix: torch.Tensor, prefix: str) -> dict[str, float]:
    i2t = recall_at_k(score_matrix, ks=(1, 5, 10))
    t2i = recall_at_k(score_matrix.T, ks=(1, 5, 10))
    return {
        f"{prefix}_i2t_r@1": i2t["r@1"],
        f"{prefix}_i2t_r@5": i2t["r@5"],
        f"{prefix}_i2t_r@10": i2t["r@10"],
        f"{prefix}_t2i_r@1": t2i["r@1"],
        f"{prefix}_t2i_r@5": t2i["r@5"],
        f"{prefix}_t2i_r@10": t2i["r@10"],
    }


def evaluate_embedding_model(
    model_name: str,
    spec: dict[str, Any],
    samples: list[CaptionSample],
    out_dir: Path,
    batch_size: int,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    cache = out_dir / f"{model_name}_features.pt"
    if cache.exists() and not force:
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        image_emb = payload["image"]
        text_emb = payload["text"]
    else:
        model, processor = load_embedding_model(spec, device)
        image_emb, text_emb = encode_embedding_split(
            model,
            processor,
            spec["family"],
            samples,
            batch_size,
            device,
        )
        torch.save({"image": image_emb, "text": text_emb, "spec": spec}, cache)
    score_matrix = embedding_score_matrix(image_emb, text_emb)
    metrics = summarize_retrieval(score_matrix, "test")
    metrics.update(
        {
            "model": model_name,
            "family": spec["family"],
            "samples": len(samples),
            "score_type": "embedding_cosine",
            "feature_dim": int(image_emb.shape[1]),
        }
    )
    return metrics


def evaluate_blip_itm_subset(
    model_name: str,
    spec: dict[str, Any],
    samples: list[CaptionSample],
    out_dir: Path,
    batch_size: int,
    device: torch.device,
    max_items: int,
    force: bool,
) -> dict[str, Any]:
    subset = samples[: min(int(max_items), len(samples))]
    cache = out_dir / f"{model_name}_{len(subset)}_score_matrix.pt"
    if cache.exists() and not force:
        score_matrix = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        model_path = resolve_model_path(spec["path"])
        model = BlipForImageTextRetrieval.from_pretrained(
            model_path,
            local_files_only=bool(spec.get("local_files_only", True)),
        ).to(device).eval()
        processor = BlipProcessor.from_pretrained(
            model_path,
            local_files_only=bool(spec.get("local_files_only", True)),
        )
        scores = []
        pairs = [(i, j) for i in range(len(subset)) for j in range(len(subset))]
        for start in tqdm(range(0, len(pairs), batch_size), desc=f"score_{model_name}_{len(subset)}"):
            batch_pairs = pairs[start : start + batch_size]
            images = [Image.open(subset[i].image_path).convert("RGB") for i, _ in batch_pairs]
            texts = [subset[j].caption for _, j in batch_pairs]
            inputs = processor(images=images, text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
                logits = out.itm_score if hasattr(out, "itm_score") else out.logits_per_image
                if logits.ndim == 2 and logits.shape[-1] == 2:
                    batch_scores = logits[:, 1]
                else:
                    batch_scores = logits.reshape(-1)
            scores.append(batch_scores.detach().cpu().float())
        score_matrix = torch.cat(scores, dim=0).view(len(subset), len(subset))
        torch.save(score_matrix, cache)
    metrics = summarize_retrieval(score_matrix, "test")
    metrics.update(
        {
            "model": model_name,
            "family": "blip_itm",
            "samples": len(subset),
            "score_type": "cross_encoder_itm",
            "feature_dim": 0,
        }
    )
    return metrics


def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    plot_cols = ["test_i2t_r@1", "test_i2t_r@5", "test_t2i_r@1", "test_t2i_r@5"]
    plot_df = df.set_index("model")[plot_cols].sort_values("test_i2t_r@1", ascending=False)
    ax = plot_df.plot(kind="bar", figsize=(12, 4), ylim=(0, 1))
    ax.set_title("Foundation VLM retrieval benchmark")
    ax.set_xlabel("")
    ax.set_ylabel("Recall")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "foundation_vlm_retrieval.png", dpi=180)
    plt.close()


def write_report(df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    cols = [
        "model",
        "family",
        "samples",
        "score_type",
        "test_i2t_r@1",
        "test_i2t_r@5",
        "test_i2t_r@10",
        "test_t2i_r@1",
        "test_t2i_r@5",
        "test_t2i_r@10",
    ]
    table = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].sort_values("test_i2t_r@1", ascending=False).iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{float(value):.4f}" if isinstance(value, float) else str(value))
        table.append("| " + " | ".join(values) + " |")
    lines = [
        "# Foundation VLM Benchmark",
        "",
        f"Dataset: `{dataset_name}`.",
        "",
        "This benchmark reports embedding-based retrieval for CLIP/OpenCLIP/SigLIP and optional cross-encoder ITM retrieval for BLIP. BLIP-ITM scores are marked with the evaluated subset size because full 5K x 5K cross-encoder scoring is computationally much heavier than embedding retrieval.",
        "",
        "\n".join(table),
        "",
        "Claim boundary: these are zero-shot or pretrained-model retrieval baselines. They do not by themselves prove FluidMix superiority; they provide stronger reference points for the FluidMix residual-adapter experiments.",
    ]
    (out_dir / "foundation_vlm_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_coco1k.yaml")
    parser.add_argument("--output", default="results/q1_foundation_vlm_coco1k")
    parser.add_argument("--models", default="clip_b32_openai,openclip_b32_laion2b,siglip_b16_google,blip_itm_coco")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--blip-batch-size", type=int, default=16)
    parser.add_argument("--blip-max-items", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    data_dir, n_train, n_val, n_test = prepare_dataset(config, force=args.force_data)
    samples = load_jsonl(data_dir / "test.jsonl")
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    selected = [name.strip() for name in args.models.split(",") if name.strip()]
    rows = []
    failures = []
    for name in selected:
        spec = DEFAULT_MODELS[name]
        try:
            print(f"[BENCH] {name}")
            if spec["family"] == "blip_itm":
                row = evaluate_blip_itm_subset(
                    name,
                    spec,
                    samples,
                    out_dir,
                    args.blip_batch_size,
                    device,
                    args.blip_max_items,
                    args.force_features,
                )
            else:
                row = evaluate_embedding_model(
                    name,
                    spec,
                    samples,
                    out_dir,
                    args.batch_size,
                    device,
                    args.force_features,
                )
            rows.append(row)
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "foundation_vlm_benchmark.csv", index=False)
        plot_results(df, out_dir)
        write_report(df, out_dir, config["dataset"]["name"])
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    metadata = {
        "dataset": config["dataset"]["name"],
        "data_dir": str(data_dir),
        "sizes": {"train": n_train, "val": n_val, "test": n_test},
        "models": selected,
        "failures": failures,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not df.empty:
        print(df.sort_values("test_i2t_r@1", ascending=False).to_string(index=False))
    if failures:
        print("Failures:", failures)
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
