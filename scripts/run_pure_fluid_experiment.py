from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

from download_dataset import prepare_dataset
from liquid_mix.data import CaptionSample, load_jsonl
from liquid_mix.pure_fluid_vlm import PureFluidConfig, PureFluidVLM, SimpleCaptionTokenizer


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


def make_pair_indices(num_samples: int, negative_offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_indices = []
    text_indices = []
    labels = []
    for idx in range(num_samples):
        image_indices.extend([idx, idx])
        text_indices.append(idx)
        neg_idx = (idx + negative_offset) % num_samples
        if neg_idx == idx:
            neg_idx = (idx + 1) % num_samples
        text_indices.append(neg_idx)
        labels.extend([1.0, 0.0])
    return (
        torch.tensor(image_indices, dtype=torch.long),
        torch.tensor(text_indices, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def best_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores_np = scores.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    candidates = np.unique(scores_np)
    best_acc = -1.0
    best_t = 0.0
    for threshold in candidates:
        preds = (scores_np >= threshold).astype(np.float32)
        acc = float((preds == labels_np).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(threshold)
    return best_t


def binary_scores_at_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    scores_np = scores.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    preds = (scores_np >= threshold).astype(np.float32)
    auc = float("nan") if len(set(labels_np.tolist())) < 2 else float(roc_auc_score(labels_np, scores_np))
    return {"accuracy": float((preds == labels_np).mean()), "auc": auc}


def recall_at_k(score_matrix: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    order = torch.argsort(score_matrix, dim=1, descending=True)
    target = torch.arange(score_matrix.shape[0], device=score_matrix.device).view(-1, 1)
    out = {}
    for k in ks:
        kk = min(k, score_matrix.shape[1])
        out[f"r@{kk}"] = float((order[:, :kk] == target).any(dim=1).float().mean().item())
    return out


def symmetric_contrastive_loss(score_matrix: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(score_matrix.shape[0], device=score_matrix.device)
    logits = score_matrix / float(temperature)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def resolve_image_path(sample: CaptionSample, data_dir: Path) -> Path:
    path = Path(sample.image_path)
    if path.exists():
        return path
    fallback = data_dir / "images" / path.parent.name / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find image for sample {sample.source_id}: {sample.image_path}")


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    image = Image.open(path).convert("RGB").resize((image_size, image_size), resample)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PositivePairDataset(Dataset):
    def __init__(
        self,
        samples: list[CaptionSample],
        tokenizer: SimpleCaptionTokenizer,
        image_size: int,
        data_dir: Path,
    ):
        self.samples = samples
        self.images = torch.stack(
            [image_to_tensor(resolve_image_path(sample, data_dir), image_size) for sample in samples],
            dim=0,
        )
        self.tokens = tokenizer.batch_encode(sample.caption for sample in samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.tokens[idx]


def batch_itm_loss(score_matrix: torch.Tensor, negatives_per_positive: int, loss_fn: nn.Module) -> torch.Tensor:
    n = score_matrix.shape[0]
    positives = score_matrix.diag()
    masked = score_matrix.clone()
    masked[torch.arange(n, device=score_matrix.device), torch.arange(n, device=score_matrix.device)] = -torch.inf
    k = min(int(negatives_per_positive), n - 1)
    hard_text = torch.topk(masked, k=k, dim=1).indices
    row = torch.arange(n, device=score_matrix.device).view(-1, 1).expand_as(hard_text)
    negatives = score_matrix[row, hard_text].reshape(-1)
    logits = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones_like(positives), torch.zeros_like(negatives)], dim=0)
    return loss_fn(logits, labels)


def score_matrix_for_dataset(
    model: PureFluidVLM,
    dataset: PositivePairDataset,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return (
            model.score_matrix(dataset.images.to(device), dataset.tokens.to(device), chunk_size=chunk_size)
            .detach()
            .cpu()
        )


def evaluate_dataset(
    model: PureFluidVLM,
    dataset: PositivePairDataset,
    threshold: float | None,
    negative_offset: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[dict[str, float], float, torch.Tensor]:
    matrix = score_matrix_for_dataset(model, dataset, device, chunk_size)
    image_idx, text_idx, labels = make_pair_indices(len(dataset), negative_offset)
    scores = matrix[image_idx, text_idx]
    chosen_threshold = best_threshold(scores, labels) if threshold is None else float(threshold)
    binary = binary_scores_at_threshold(scores, labels, chosen_threshold)
    retrieval = recall_at_k(matrix, ks=(1, 5))
    metrics = {
        "accuracy": binary["accuracy"],
        "auc": binary["auc"],
        "r@1": retrieval["r@1"],
        "r@5": retrieval["r@5"],
    }
    return metrics, chosen_threshold, matrix


def build_model(config: dict, tokenizer: SimpleCaptionTokenizer) -> PureFluidVLM:
    model_cfg = config["model"]
    pure_config = PureFluidConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=int(model_cfg["max_len"]),
        image_size=int(model_cfg["image_size"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        field_grid=int(model_cfg["field_grid"]),
        text_layers=int(model_cfg["text_layers"]),
        text_heads=int(model_cfg["text_heads"]),
        dropout=float(model_cfg["dropout"]),
        steps=int(model_cfg["steps"]),
        dt=float(model_cfg["dt"]),
        diffusion=float(model_cfg["diffusion"]),
        reaction=float(model_cfg["reaction"]),
        num_heads=int(model_cfg["num_heads"]),
    )
    return PureFluidVLM(pure_config)


def load_clip_summary(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keep = df[df["model"].isin(["clip_zero_shot_cosine", "clip_frozen_residual_fluidmix"])].copy()
    return keep if not keep.empty else None


def plot_training_curve(out_dir: Path, curves: list[dict]) -> None:
    df = pd.DataFrame(curves)
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df["epoch"], df["train_loss"], label="train_loss", color="#2f5597")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_accuracy"], label="val_accuracy", color="#8c4b2f")
    ax2.plot(df["epoch"], df["val_r@1"], label="val_r@1", color="#3f7f4f")
    ax2.set_ylabel("validation score")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def plot_comparison(out_dir: Path, pure_metrics: dict, clip_summary: pd.DataFrame | None) -> None:
    rows = [pure_metrics]
    if clip_summary is not None:
        rows = clip_summary.to_dict("records") + rows
    df = pd.DataFrame(rows)
    columns = ["test_accuracy", "test_auc", "test_r@1", "test_r@5"]
    ax = df.set_index("model")[columns].plot(kind="bar", figsize=(10, 4), ylim=(0, 1))
    ax.set_xlabel("")
    ax.set_ylabel("score")
    ax.set_title("Pure FluidMix VLM vs preserved CLIP results")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "pure_vs_clip.png", dpi=180)
    plt.close()


def write_report(
    out_dir: Path,
    config: dict,
    pure_metrics: dict,
    curves: list[dict],
    clip_summary: pd.DataFrame | None,
    sizes: dict,
) -> None:
    rows = [pure_metrics]
    if clip_summary is not None:
        rows = clip_summary.to_dict("records") + rows
    df = pd.DataFrame(rows)
    columns = ["model", "trainable", "test_accuracy", "test_auc", "test_r@1", "test_r@5"]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        table.append(
            "| "
            + " | ".join(
                str(row[col]) if col in {"model", "trainable"} else f"{float(row[col]):.4f}"
                for col in columns
            )
            + " |"
        )

    conclusion = "未找到保留的 CLIP summary，因此本报告只给出 pure FluidMix VLM 的绝对结果。"
    if clip_summary is not None and (clip_summary["model"] == "clip_zero_shot_cosine").any():
        clip = clip_summary[clip_summary["model"] == "clip_zero_shot_cosine"].iloc[0]
        metrics = ["test_accuracy", "test_auc", "test_r@1", "test_r@5"]
        beats = [float(pure_metrics[m]) > float(clip[m]) for m in metrics]
        if all(beats):
            conclusion = "pure FluidMix VLM 在四项测试指标上均超过保留的 CLIP zero-shot baseline。"
        else:
            gaps = ", ".join(f"{m}={float(pure_metrics[m]) - float(clip[m]):+.4f}" for m in metrics)
            conclusion = (
                "pure FluidMix VLM 当前尚未超过 CLIP zero-shot；差值为 "
                f"{gaps}。这说明在 240 个训练样本上，从零训练视觉和文本 encoder 的样本效率仍明显不足。"
            )

    best_epoch = max(curves, key=lambda row: row["selection_score"])["epoch"] if curves else "NA"
    lines = [
        "# Pure FluidMix VLM 实验报告",
        "",
        "## 实验目的",
        "",
        "本实验保留原有 CLIP/residual FluidMix 结果不变，新增一条完全摒弃 CLIP 的端到端结构：不使用 CLIP encoder、CLIP tokenizer 或 CLIP cosine 分数，只使用本工程自建 image encoder、text encoder 和 FluidMix 动力学融合器。",
        "",
        "## 数据与协议",
        "",
        f"- 数据集：`{config['dataset']['name']}`",
        f"- split：train/validation/test = {sizes['train']}/{sizes['val']}/{sizes['test']}",
        "- observation：RGB 图像 `[B,3,96,96]` 与 caption token ids `[B,32]`",
        "- state：图像浓度场 `C_img [B,128,4,4]`、文本浓度场 `C_txt [B,128,4,4]`、混合场 `C_t`",
        "- action/reward：本任务是监督式 VLM matching/retrieval，不定义机器人 action/reward；训练信号是 image-caption 正负配对标签与 batch 内 retrieval 对角目标",
        "- timestep：FluidMix 内部连续时间离散步 `steps=4`，`dt=0.2`",
        "- loss：`contrastive_weight * symmetric_contrastive_loss + itm_weight * hard_negative_BCE`",
        f"- seed：{config['model']['seed']}",
        f"- best_epoch：{best_epoch}",
        "",
        "## 结构",
        "",
        "```text",
        "image pixels -> CNN patch encoder -> image_vec, C_img",
        "caption text -> project tokenizer -> Transformer text encoder -> text_vec, C_txt",
        "C_0 = gate(image_vec, text_vec) * C_img + (1 - gate) * C_txt",
        "C_{t+1} = C_t + dt * (advection + diffusion + reaction)",
        "score = learned image-text retrieval logit + learned FluidMix matching logit",
        "```",
        "",
        "## 结果",
        "",
        "\n".join(table),
        "",
        "## 结论",
        "",
        conclusion,
        "工程结论：该分支已经证明可以完全脱离 CLIP 跑通端到端多模态训练和检索评估；但如果目标是稳定超过 CLIP，当前小数据从零训练路线还需要更大数据、预训练或更强的视觉/文本自监督阶段。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def train(config: dict, out_dir: Path, force_data: bool) -> dict:
    model_cfg = config["model"]
    train_cfg = config["training"]
    set_seed(int(model_cfg["seed"]))
    device = resolve_device(str(train_cfg["device"]))
    data_dir, n_train, n_val, n_test = prepare_dataset(config, force=force_data)
    train_samples = load_jsonl(data_dir / "train.jsonl")
    val_samples = load_jsonl(data_dir / "val.jsonl")
    test_samples = load_jsonl(data_dir / "test.jsonl")

    tokenizer = SimpleCaptionTokenizer.fit(
        (sample.caption for sample in train_samples),
        max_vocab=int(model_cfg["max_vocab"]),
        max_len=int(model_cfg["max_len"]),
        min_freq=int(model_cfg["min_freq"]),
    )
    datasets = {
        "train": PositivePairDataset(train_samples, tokenizer, int(model_cfg["image_size"]), data_dir),
        "val": PositivePairDataset(val_samples, tokenizer, int(model_cfg["image_size"]), data_dir),
        "test": PositivePairDataset(test_samples, tokenizer, int(model_cfg["image_size"]), data_dir),
    }
    loader = DataLoader(
        datasets["train"],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(model_cfg["seed"])),
    )
    model = build_model(config, tokenizer).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    bce = nn.BCEWithLogitsLoss()
    curves = []
    best_state = None
    best_score = -float("inf")
    stale_epochs = 0
    pair_chunk_size = int(train_cfg["pair_chunk_size"])
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for images, tokens in tqdm(loader, desc=f"epoch {epoch:03d}", leave=False):
            images = images.to(device)
            tokens = tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            score_matrix = model.score_matrix(images, tokens, chunk_size=pair_chunk_size)
            contrastive = symmetric_contrastive_loss(score_matrix, float(train_cfg["contrastive_temperature"]))
            itm = batch_itm_loss(score_matrix, int(train_cfg["hard_negatives_per_positive"]), bce)
            loss = float(train_cfg["contrastive_weight"]) * contrastive + float(train_cfg["itm_weight"]) * itm
            loss.backward()
            if float(train_cfg["gradient_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_metrics, val_threshold, _ = evaluate_dataset(
            model,
            datasets["val"],
            threshold=None,
            negative_offset=int(train_cfg["negative_offset"]),
            device=device,
            chunk_size=pair_chunk_size,
        )
        selection_score = val_metrics["r@1"] + 0.5 * val_metrics["accuracy"] + 0.1 * val_metrics["auc"]
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_accuracy": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "val_r@1": val_metrics["r@1"],
                "val_r@5": val_metrics["r@5"],
                "threshold": val_threshold,
                "selection_score": selection_score,
            }
        )
        if selection_score > best_score:
            best_score = selection_score
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_r@1={val_metrics['r@1']:.4f} val_r@5={val_metrics['r@5']:.4f}"
        )
        if stale_epochs >= int(train_cfg["patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    val_metrics, threshold, _ = evaluate_dataset(
        model,
        datasets["val"],
        threshold=None,
        negative_offset=int(train_cfg["negative_offset"]),
        device=device,
        chunk_size=pair_chunk_size,
    )
    test_metrics, _, _ = evaluate_dataset(
        model,
        datasets["test"],
        threshold=threshold,
        negative_offset=int(train_cfg["negative_offset"]),
        device=device,
        chunk_size=pair_chunk_size,
    )

    pure_metrics = {
        "model": "pure_fluidmix_vlm",
        "encoder": "project_cnn_transformer_no_clip",
        "trainable": "end_to_end",
        "seed": int(model_cfg["seed"]),
        "epochs": len(curves),
        "threshold": threshold,
        "vocab_size": tokenizer.vocab_size,
        "val_accuracy": val_metrics["accuracy"],
        "val_auc": val_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "val_r@1": val_metrics["r@1"],
        "val_r@5": val_metrics["r@5"],
        "test_r@1": test_metrics["r@1"],
        "test_r@5": test_metrics["r@5"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(pure_metrics, indent=2), encoding="utf-8")
    (out_dir / "curve.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.state_dict(), indent=2), encoding="utf-8")
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config.to_dict(),
            "tokenizer": tokenizer.state_dict(),
            "metrics": pure_metrics,
        },
        out_dir / "model.pt",
    )
    summary_path = ROOT / str(train_cfg.get("clip_summary", "results/clip_vlm_pokemon/summary.csv"))
    clip_summary = load_clip_summary(summary_path)
    pd.DataFrame([pure_metrics]).to_csv(out_dir / "summary.csv", index=False)
    plot_training_curve(out_dir, curves)
    plot_comparison(out_dir, pure_metrics, clip_summary)
    write_report(
        out_dir,
        config,
        pure_metrics,
        curves,
        clip_summary,
        {"train": n_train, "val": n_val, "test": n_test},
    )
    return pure_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pure_fluid_pokemon.yaml")
    parser.add_argument("--output", default="results/pure_fluid_pokemon")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / args.output
    metrics = train(config, out_dir, force_data=args.force_data)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
