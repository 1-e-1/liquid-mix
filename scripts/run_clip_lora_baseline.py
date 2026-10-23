from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import yaml

from download_dataset import prepare_dataset
from liquid_mix.clip_fusion import recall_at_k
from liquid_mix.data import CaptionSample, load_jsonl

ImageFile.LOAD_TRUNCATED_IMAGES = True


def repair_image_paths(samples: list[CaptionSample], data_dir: Path, split: str) -> list[CaptionSample]:
    repaired = []
    for sample in samples:
        path = Path(sample.image_path)
        if path.exists():
            repaired.append(sample)
            continue
        fallback = data_dir / "images" / split / path.name
        if fallback.exists():
            repaired.append(
                CaptionSample(
                    image_path=str(fallback),
                    caption=sample.caption,
                    source_id=sample.source_id,
                )
            )
            continue
        repaired.append(sample)
    return repaired


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class CaptionDataset(Dataset):
    def __init__(self, samples: list[CaptionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CaptionSample:
        return self.samples[idx]


def collate_clip(processor: CLIPProcessor):
    def _collate(batch: list[CaptionSample]) -> dict[str, torch.Tensor]:
        images = [Image.open(sample.image_path).convert("RGB") for sample in batch]
        texts = [sample.caption for sample in batch]
        return processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    return _collate


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def clip_core(model: torch.nn.Module) -> torch.nn.Module:
    # PEFT wraps the original CLIPModel at model.base_model.model.
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return 0.5 * (
        F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)
    )


def build_clip(model_name: str, local_files_only: bool, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_safetensors=False,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    return model, processor


def apply_lora(model: CLIPModel, args: argparse.Namespace) -> torch.nn.Module:
    config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        lora_dropout=float(args.lora_dropout),
        bias="none",
    )
    model = get_peft_model(model, config)
    if bool(args.train_logit_scale):
        clip_core(model).logit_scale.requires_grad_(True)
    return model


def encode_split(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    samples: list[CaptionSample],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    core = clip_core(model)
    image_features = []
    text_features = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(samples), batch_size), desc="encode_eval"):
            batch = samples[start : start + batch_size]
            images = [Image.open(sample.image_path).convert("RGB") for sample in batch]
            texts = [sample.caption for sample in batch]
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
            image_out = core.vision_model(**image_inputs)
            text_out = core.text_model(**text_inputs)
            image_emb = core.visual_projection(image_out.pooler_output)
            text_emb = core.text_projection(text_out.pooler_output)
            image_features.append(image_emb.detach().cpu().float())
            text_features.append(text_emb.detach().cpu().float())
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def retrieval_metrics(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    score_matrix = image_emb @ text_emb.T
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


def evaluate_model(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    samples: list[CaptionSample],
    batch_size: int,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    image_emb, text_emb = encode_split(model, processor, samples, batch_size, device)
    return retrieval_metrics(image_emb, text_emb, prefix)


def train_lora(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    train_samples: list[CaptionSample],
    val_samples: list[CaptionSample],
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        CaptionDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
        collate_fn=collate_clip(processor),
    )
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(args.amp)))
    curves = []
    best_score = -1.0
    best_state = None
    patience_left = int(args.patience)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"lora_epoch_{epoch}"):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and bool(args.amp))):
                out = model(**batch)
                loss = clip_contrastive_loss(out.logits_per_image, out.logits_per_text)
            scaler.scale(loss).backward()
            if float(args.gradient_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, float(args.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(model, processor, val_samples, int(args.eval_batch_size), device, "val")
        val_score = 0.5 * (val_metrics["val_i2t_r@1"] + val_metrics["val_t2i_r@1"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_score": val_score,
            **val_metrics,
        }
        curves.append(row)
        print(json.dumps(row, indent=2))
        if val_score > best_score:
            best_score = val_score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(out_dir / "lora_adapter")
    return curves


def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["test_i2t_r@1", "test_i2t_r@5", "test_t2i_r@1", "test_t2i_r@5"]
    plot_df = df.set_index("model")[cols]
    ax = plot_df.plot(kind="bar", figsize=(10, 4), ylim=(0, 1))
    ax.set_title("CLIP zero-shot vs CLIP-LoRA")
    ax.set_xlabel("")
    ax.set_ylabel("Recall")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "clip_lora_retrieval.png", dpi=180)
    plt.close()


def write_report(df: pd.DataFrame, out_dir: Path, config: dict, args: argparse.Namespace, sizes: dict[str, int]) -> None:
    cols = [
        "model",
        "trainable_params",
        "epochs",
        "test_i2t_r@1",
        "test_i2t_r@5",
        "test_i2t_r@10",
        "test_t2i_r@1",
        "test_t2i_r@5",
        "test_t2i_r@10",
    ]
    table = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{float(value):.4f}" if isinstance(value, float) else str(value))
        table.append("| " + " | ".join(values) + " |")
    lines = [
        "# CLIP-LoRA baseline",
        "",
        f"Dataset: `{config['dataset']['name']}` with train/val/test={sizes['train']}/{sizes['val']}/{sizes['test']}.",
        f"Backbone: `{config['clip']['model_name']}`.",
        f"LoRA target modules: `{args.target_modules}`, rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}.",
        "Loss: symmetric CLIP contrastive cross entropy on in-batch image-text pairs.",
        "",
        "\n".join(table),
        "",
        "Claim boundary: this is a lightweight local fine-tuning baseline. It is not large-scale CLIP re-pretraining.",
    ]
    (out_dir / "clip_lora_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_coco1k.yaml")
    parser.add_argument("--output", default="results/q1_clip_lora_coco1k")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--train-logit-scale", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    data_dir, n_train, n_val, n_test = prepare_dataset(config, force=args.force_data)
    train_samples = repair_image_paths(load_jsonl(data_dir / "train.jsonl"), data_dir, "train")
    if int(args.max_train_items) > 0:
        train_samples = train_samples[: int(args.max_train_items)]
    val_samples = repair_image_paths(load_jsonl(data_dir / "val.jsonl"), data_dir, "val")
    test_samples = repair_image_paths(load_jsonl(data_dir / "test.jsonl"), data_dir, "test")
    device = resolve_device(args.device)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = build_clip(
        config["clip"]["model_name"],
        bool(config["clip"].get("local_files_only", True)),
        device,
    )
    zero = evaluate_model(model, processor, test_samples, int(args.eval_batch_size), device, "test")
    model = apply_lora(model, args).to(device)
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    total = int(sum(param.numel() for param in model.parameters()))
    curves = train_lora(model, processor, train_samples, val_samples, args, out_dir, device)
    lora = evaluate_model(model, processor, test_samples, int(args.eval_batch_size), device, "test")

    rows = [
        {
            "model": "clip_zero_shot_openai",
            "trainable_params": 0,
            "total_params": total,
            "epochs": 0,
            **zero,
        },
        {
            "model": "clip_lora_finetuned",
            "trainable_params": trainable,
            "total_params": total,
            "epochs": int(args.epochs),
            **lora,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "clip_lora_summary.csv", index=False)
    (out_dir / "curve.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    metadata = {
        "config": args.config,
        "dataset": config["dataset"]["name"],
        "data_dir": str(data_dir),
        "sizes": {"train": len(train_samples), "val": n_val, "test": n_test},
        "device": str(device),
        "seed": int(args.seed),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(df, out_dir)
    write_report(df, out_dir, config, args, metadata["sizes"])
    print(df.to_string(index=False))
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
