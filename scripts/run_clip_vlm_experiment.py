from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import yaml

from download_dataset import prepare_dataset
from liquid_mix.clip_fusion import (
    CLIPAdapterConfig,
    CLIPFusionAdapter,
    ResidualCLIPFusionScorer,
    best_threshold,
    binary_scores_at_threshold,
    clip_cosine_scores,
    make_hard_negative_pair_indices,
    make_pair_indices,
    recall_at_k,
    symmetric_contrastive_loss,
)
from liquid_mix.data import CaptionSample, load_jsonl


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


def load_clip(model_name: str, local_files_only: bool, device: torch.device):
    model = CLIPModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_safetensors=False,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    return model, processor


def encode_split(
    model,
    processor,
    samples: list[CaptionSample],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_features = []
    text_features = []
    for start in tqdm(range(0, len(samples), batch_size), desc="encode_clip"):
        batch = samples[start : start + batch_size]
        images = [Image.open(sample.image_path).convert("RGB") for sample in batch]
        texts = [sample.caption for sample in batch]
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            image_out = model.vision_model(**image_inputs)
            text_out = model.text_model(**text_inputs)
            image_features.append(model.visual_projection(image_out.pooler_output).detach().cpu())
            text_features.append(model.text_projection(text_out.pooler_output).detach().cpu())
    return {
        "image": torch.cat(image_features, dim=0).float(),
        "text": torch.cat(text_features, dim=0).float(),
    }


def load_or_create_features(config: dict, data_dir: Path, out_dir: Path, device: torch.device):
    feature_path = out_dir / "clip_features.pt"
    train = load_jsonl(data_dir / "train.jsonl")
    val = load_jsonl(data_dir / "val.jsonl")
    test = load_jsonl(data_dir / "test.jsonl")
    expected = {"train": len(train), "val": len(val), "test": len(test)}
    if feature_path.exists():
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        if payload.get("sizes") == expected and payload.get("model_name") == config["clip"]["model_name"]:
            return payload, train, val, test

    model, processor = load_clip(
        config["clip"]["model_name"],
        bool(config["clip"].get("local_files_only", True)),
        device,
    )
    payload = {
        "model_name": config["clip"]["model_name"],
        "sizes": expected,
        "train": encode_split(model, processor, train, int(config["clip"]["batch_size"]), device),
        "val": encode_split(model, processor, val, int(config["clip"]["batch_size"]), device),
        "test": encode_split(model, processor, test, int(config["clip"]["batch_size"]), device),
    }
    torch.save(payload, feature_path)
    return payload, train, val, test


def score_pairs_with_model(
    model: nn.Module,
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    image_idx: torch.Tensor,
    text_idx: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(image_idx), batch_size):
            ii = image_idx[start : start + batch_size]
            ti = text_idx[start : start + batch_size]
            scores.append(model(image_emb[ii].to(device), text_emb[ti].to(device)).detach().cpu())
    return torch.cat(scores, dim=0)


def score_matrix_with_model(
    model: nn.Module,
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    n = image_emb.shape[0]
    image_idx = torch.arange(n).repeat_interleave(n)
    text_idx = torch.arange(n).repeat(n)
    scores = score_pairs_with_model(model, image_emb, text_emb, image_idx, text_idx, batch_size, device)
    return scores.view(n, n)


def differentiable_score_matrix(
    model: nn.Module,
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
) -> torch.Tensor:
    n = image_emb.shape[0]
    pair_images = image_emb.repeat_interleave(n, dim=0)
    pair_texts = text_emb.repeat(n, 1)
    return model(pair_images, pair_texts).view(n, n)


def build_adapter_model(method: str, adapter_cfg: dict, clip_dim: int) -> nn.Module:
    residual = method.startswith("residual_")
    base_method = method.removeprefix("residual_")
    config = CLIPAdapterConfig(
        method=base_method,
        clip_dim=clip_dim,
        hidden_dim=int(adapter_cfg["hidden_dim"]),
        field_grid=int(adapter_cfg["field_grid"]),
        steps=int(adapter_cfg["steps"]),
        dt=float(adapter_cfg["dt"]),
        diffusion=float(adapter_cfg["diffusion"]),
        reaction=float(adapter_cfg["reaction"]),
        num_heads=int(adapter_cfg["num_heads"]),
        use_advection=bool(adapter_cfg.get("use_advection", True)),
        use_diffusion=bool(adapter_cfg.get("use_diffusion", True)),
        use_reaction=bool(adapter_cfg.get("use_reaction", True)),
    )
    if residual:
        return ResidualCLIPFusionScorer(config, clip_scale=float(adapter_cfg.get("clip_scale", 1.0)))
    return CLIPFusionAdapter(config)


def train_adapter(
    method: str,
    config: dict,
    features: dict,
    out_dir: Path,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    adapter_cfg = config["adapter"]
    train_f = features["train"]
    val_f = features["val"]
    test_f = features["test"]
    seed = int(adapter_cfg["seed"])
    set_seed(seed)

    hard_negatives = int(adapter_cfg.get("hard_negatives_per_positive", 0))
    if hard_negatives > 0:
        train_i, train_t, train_y = make_hard_negative_pair_indices(
            clip_cosine_scores(train_f["image"], train_f["text"]),
            negatives_per_positive=hard_negatives,
        )
    else:
        train_i, train_t, train_y = make_pair_indices(train_f["image"].shape[0], int(adapter_cfg["negative_offset"]))
    val_i, val_t, val_y = make_pair_indices(val_f["image"].shape[0], int(adapter_cfg["negative_offset"]))
    test_i, test_t, test_y = make_pair_indices(test_f["image"].shape[0], int(adapter_cfg["negative_offset"]))

    dataset = TensorDataset(train_i, train_t, train_y)
    loader = DataLoader(
        dataset,
        batch_size=int(adapter_cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_adapter_model(method, adapter_cfg, train_f["image"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(adapter_cfg["lr"]), weight_decay=float(adapter_cfg["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()
    contrastive_weight = float(adapter_cfg.get("contrastive_weight", 0.0))
    contrastive_temperature = float(adapter_cfg.get("contrastive_temperature", 0.07))
    contrastive_batch_size = int(adapter_cfg.get("contrastive_batch_size", 16))
    use_contrastive = method.startswith("residual_") and contrastive_weight > 0
    curves = []
    best_state = None
    best_val = -1.0
    for epoch in range(1, int(adapter_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for ii, ti, yy in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_f["image"][ii].to(device), train_f["text"][ti].to(device))
            loss = loss_fn(logits, yy.to(device))
            if use_contrastive:
                positive_count = min(contrastive_batch_size, train_f["image"].shape[0])
                pos_idx = torch.randperm(train_f["image"].shape[0], device=device)[:positive_count]
                score_matrix = differentiable_score_matrix(
                    model,
                    train_f["image"][pos_idx.cpu()].to(device),
                    train_f["text"][pos_idx.cpu()].to(device),
                )
                loss = loss + contrastive_weight * symmetric_contrastive_loss(
                    score_matrix,
                    temperature=contrastive_temperature,
                )
            loss.backward()
            if float(adapter_cfg["gradient_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(adapter_cfg["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_scores = score_pairs_with_model(
            model,
            val_f["image"],
            val_f["text"],
            val_i,
            val_t,
            int(adapter_cfg["batch_size"]),
            device,
        )
        threshold = best_threshold(val_scores, val_y)
        val_metrics = binary_scores_at_threshold(val_scores, val_y, threshold)
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_metrics["accuracy"], "val_auc": val_metrics["auc"]})
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    val_scores = score_pairs_with_model(model, val_f["image"], val_f["text"], val_i, val_t, int(adapter_cfg["batch_size"]), device)
    threshold = best_threshold(val_scores, val_y)
    test_scores = score_pairs_with_model(model, test_f["image"], test_f["text"], test_i, test_t, int(adapter_cfg["batch_size"]), device)
    val_metrics = binary_scores_at_threshold(val_scores, val_y, threshold)
    test_metrics = binary_scores_at_threshold(test_scores, test_y, threshold)
    val_retrieval = recall_at_k(score_matrix_with_model(model, val_f["image"], val_f["text"], int(adapter_cfg["batch_size"]), device), ks=(1, 5))
    test_retrieval = recall_at_k(score_matrix_with_model(model, test_f["image"], test_f["text"], int(adapter_cfg["batch_size"]), device), ks=(1, 5))

    metrics = {
        "model": f"clip_frozen_{method}",
        "encoder": config["clip"]["model_name"],
        "trainable": "adapter_only",
        "seed": seed,
        "epochs": int(adapter_cfg["epochs"]),
        "threshold": threshold,
        "val_accuracy": val_metrics["accuracy"],
        "val_auc": val_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "val_r@1": val_retrieval["r@1"],
        "val_r@5": val_retrieval["r@5"],
        "test_r@1": test_retrieval["r@1"],
        "test_r@5": test_retrieval["r@5"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "curve.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    model_config = asdict(model.adapter.config) if isinstance(model, ResidualCLIPFusionScorer) else asdict(model.config)
    torch.save({"model": model.state_dict(), "config": model_config}, out_dir / "adapter.pt")
    return metrics, curves


def evaluate_clip_zero_shot(config: dict, features: dict) -> dict:
    val_scores_matrix = clip_cosine_scores(features["val"]["image"], features["val"]["text"])
    test_scores_matrix = clip_cosine_scores(features["test"]["image"], features["test"]["text"])
    val_i, val_t, val_y = make_pair_indices(features["val"]["image"].shape[0], int(config["adapter"]["negative_offset"]))
    test_i, test_t, test_y = make_pair_indices(features["test"]["image"].shape[0], int(config["adapter"]["negative_offset"]))
    val_scores = val_scores_matrix[val_i, val_t]
    test_scores = test_scores_matrix[test_i, test_t]
    threshold = best_threshold(val_scores, val_y)
    val_metrics = binary_scores_at_threshold(val_scores, val_y, threshold)
    test_metrics = binary_scores_at_threshold(test_scores, test_y, threshold)
    val_retrieval = recall_at_k(val_scores_matrix, ks=(1, 5))
    test_retrieval = recall_at_k(test_scores_matrix, ks=(1, 5))
    return {
        "model": "clip_zero_shot_cosine",
        "encoder": config["clip"]["model_name"],
        "trainable": "none",
        "seed": int(config["adapter"]["seed"]),
        "epochs": 0,
        "threshold": threshold,
        "val_accuracy": val_metrics["accuracy"],
        "val_auc": val_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "val_r@1": val_retrieval["r@1"],
        "val_r@5": val_retrieval["r@5"],
        "test_r@1": test_retrieval["r@1"],
        "test_r@5": test_retrieval["r@5"],
    }


def plot_results(out_dir: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    order = df.sort_values("test_r@1", ascending=False)["model"].tolist()
    plot_df = df.set_index("model").loc[order]
    for columns, filename, title in [
        (["test_r@1", "test_r@5"], "retrieval.png", "Image-to-text retrieval on real image-caption data"),
        (["test_accuracy", "test_auc"], "binary_matching.png", "Binary image-caption matching"),
    ]:
        ax = plot_df[columns].plot(kind="bar", figsize=(11, 4), ylim=(0, 1))
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def write_report(out_dir: Path, rows: list[dict], config: dict, sizes: dict) -> None:
    df = pd.DataFrame(rows).sort_values(["test_r@1", "test_accuracy"], ascending=False)
    columns = ["model", "trainable", "test_accuracy", "test_auc", "test_r@1", "test_r@5"]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        table.append(
            "| "
            + " | ".join(str(row[col]) if col in {"model", "trainable"} else f"{float(row[col]):.4f}" for col in columns)
            + " |"
        )
    best = df.iloc[0]
    clip = df[df["model"] == "clip_zero_shot_cosine"].iloc[0]
    fluidmix_rows = df[df["model"] == "clip_frozen_fluidmix"]
    residual_rows = df[df["model"] == "clip_frozen_residual_fluidmix"]
    fluidmix = fluidmix_rows.iloc[0] if not fluidmix_rows.empty else None
    residual_fluidmix = residual_rows.iloc[0] if not residual_rows.empty else None
    residual_beats_clip = (
        residual_fluidmix is not None
        and all(float(residual_fluidmix[m]) > float(clip[m]) for m in ["test_accuracy", "test_auc", "test_r@1", "test_r@5"])
    )
    lines = [
        "# CLIP Frozen VLM + FluidMix Adapter 实验报告",
        "",
        f"数据集：`{config['dataset']['name']}`，train/validation/test = {sizes['train']}/{sizes['val']}/{sizes['test']}。",
        f"VLM encoder：`{config['clip']['model_name']}`。CLIP encoder 冻结，只训练 adapter。",
        "",
        "本实验用于比较液态流体混合模型的多模态融合能力。输入是真实 pretrained VLM 的 image/text embedding。",
        "",
        "## 结果",
        "",
        "\n".join(table),
        "",
        "## 结论",
        "",
        f"按 test_r@1 排序，最佳模型是 `{best['model']}`，test_r@1={best['test_r@1']:.4f}，test_accuracy={best['test_accuracy']:.4f}。",
        f"`clip_zero_shot_cosine` 的 test_r@1={clip['test_r@1']:.4f}，test_r@5={clip['test_r@5']:.4f}，test_accuracy={clip['test_accuracy']:.4f}。",
        (
            f"`clip_frozen_fluidmix` 的 test_r@1={fluidmix['test_r@1']:.4f}，test_r@5={fluidmix['test_r@5']:.4f}，test_accuracy={fluidmix['test_accuracy']:.4f}。"
            if fluidmix is not None
            else "`clip_frozen_fluidmix` 未在本配置中运行。"
        ),
        (
            f"`clip_frozen_residual_fluidmix` 的 test_r@1={residual_fluidmix['test_r@1']:.4f}，test_r@5={residual_fluidmix['test_r@5']:.4f}，test_accuracy={residual_fluidmix['test_accuracy']:.4f}。"
            if residual_fluidmix is not None
            else "`clip_frozen_residual_fluidmix` 未在本配置中运行。"
        ),
        "当前结论："
        + (
            "优化后的 residual FluidMix 在 test_accuracy、test_auc、test_r@1、test_r@5 四项指标上均超过原始 CLIP zero-shot。"
            if residual_beats_clip
            else "优化后的 residual FluidMix 尚未在四项指标上全部超过原始 CLIP zero-shot。"
        ),
        "关键机制：residual scoring 保留 CLIP cosine baseline，hard-negative BCE 提升细粒度区分，symmetric contrastive loss 直接优化 retrieval ranking。",
        "边界：当前仍是小规模单数据集结果，论文级结论需要多 seed、更大数据集和更多 VLM encoder 验证。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clip_vlm_pokemon.yaml")
    parser.add_argument("--output", default="results/clip_vlm_pokemon")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    set_seed(int(config["adapter"]["seed"]))
    device = resolve_device(str(config["adapter"]["device"]))
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir, n_train, n_val, n_test = prepare_dataset(config, force=args.force_data)
    features, _, _, _ = load_or_create_features(config, data_dir, out_dir, device)

    rows = [evaluate_clip_zero_shot(config, features)]
    curves = {}
    for method in config["adapter"]["methods"]:
        print(f"[ADAPTER] {method}")
        metrics, curve = train_adapter(method, config, features, out_dir / method, device)
        rows.append(metrics)
        curves[method] = curve

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    plot_results(out_dir, rows)
    write_report(out_dir, rows, config, {"train": n_train, "val": n_val, "test": n_test})
    print(df.sort_values(["test_r@1", "test_accuracy"], ascending=False).to_string(index=False))
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
