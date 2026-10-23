from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from liquid_mix.statistics import (
    exact_paired_mcnemar,
    holm_adjust,
    paired_bootstrap_metric_delta,
)


MODEL_LABELS = {
    "clip_anchor": "CLIP",
    "symmetric_mlp": "Symmetric MLP",
    "low_rank_bilinear": "Low-rank bilinear",
    "gated_multimodal": "Gated fusion",
    "bidirectional_cross_attention": "Bi-cross-attention",
    "late_interaction": "Late interaction",
    "transformer_cross_encoder": "Transformer cross-encoder",
    "adr": "ADR FluidMix",
    "adr_without_advection": "ADR w/o advection",
    "adr_without_diffusion": "ADR w/o diffusion",
    "adr_without_reaction": "ADR w/o reaction",
    "parameter_matched_mlp": "Parameter-matched MLP",
}


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_indicators(results: Path, dataset: str, model: str) -> dict[str, torch.Tensor]:
    path = results / "runs" / dataset / model / "test_indicators.npz"
    payload = np.load(path)
    return {
        key.replace("at", "@"): torch.from_numpy(payload[key]).float()
        for key in payload.files
    }


def paired_comparison_rows(
    results: Path,
    datasets: list[str],
    baseline: str,
    improved: str,
    metrics: tuple[str, ...],
    n_boot: int,
) -> list[dict]:
    rows = []
    for dataset in datasets:
        baseline_values = load_indicators(results, dataset, baseline)
        improved_values = load_indicators(results, dataset, improved)
        dataset_rows = []
        for key in sorted(baseline_values):
            if not any(key.endswith(metric) for metric in metrics):
                continue
            bootstrap = paired_bootstrap_metric_delta(
                baseline_values[key],
                improved_values[key],
                n_boot=n_boot,
                seed=13,
            )
            test = exact_paired_mcnemar(baseline_values[key], improved_values[key])
            direction, metric = key.split("_", maxsplit=1)
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "improved": improved,
                    "direction": direction,
                    "metric": metric,
                    **bootstrap,
                    **test,
                }
            )
        adjusted = holm_adjust([float(row["p_value"]) for row in dataset_rows])
        for row, p_holm in zip(dataset_rows, adjusted):
            row["p_holm"] = p_holm
            row["significant_0.05"] = p_holm < 0.05
        rows.extend(dataset_rows)
    return rows


def load_test_metrics(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(results / "aggregate_metrics.csv")
    test = metrics[metrics["split"] == "test"].copy()
    clip = test[test["model"] == "clip_anchor"].drop_duplicates(["dataset", "direction"])
    models = test[test["model"] != "clip_anchor"].copy()
    return models, clip


def build_figures(
    results: Path,
    models: pd.DataFrame,
    clip: pd.DataFrame,
    adr_statistics: pd.DataFrame,
    noise: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> None:
    figure_dir = results / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

    merged = models.merge(
        clip[["dataset", "direction", "r@1"]],
        on=["dataset", "direction"],
        suffixes=("", "_clip"),
    )
    merged["delta_r1"] = merged["r@1"] - merged["r@1_clip"]
    main_models = [
        "symmetric_mlp",
        "low_rank_bilinear",
        "gated_multimodal",
        "bidirectional_cross_attention",
        "late_interaction",
        "transformer_cross_encoder",
        "parameter_matched_mlp",
        "adr",
    ]
    summary = (
        merged[merged["model"].isin(main_models)]
        .groupby("model")["delta_r1"]
        .mean()
        .sort_values()
    )
    colors = ["#1f6f8b" if model == "adr" else "#8a98a6" for model in summary.index]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh([MODEL_LABELS[name] for name in summary.index], summary.values * 100, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean absolute R@1 change vs CLIP (percentage points)")
    ax.set_title("Second-stage reranking across five datasets")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "main_r1_delta.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    adr = merged[merged["model"] == "adr"].copy()
    order = ["pokemon", "coco1k", "coco_karpathy", "flickr8k", "flickr30k"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(order))
    for offset, direction, color in [(-0.13, "i2t", "#176b87"), (0.13, "t2i", "#c45d37")]:
        values = []
        lows = []
        highs = []
        for dataset in order:
            value = float(
                adr[(adr["dataset"] == dataset) & (adr["direction"] == direction)]["delta_r1"].iloc[0]
            )
            stat = adr_statistics[
                (adr_statistics["dataset"] == dataset)
                & (adr_statistics["direction"] == direction)
                & (adr_statistics["metric"] == "r@1")
            ].iloc[0]
            values.append(value * 100)
            lows.append((value - float(stat["ci_low"])) * 100)
            highs.append((float(stat["ci_high"]) - value) * 100)
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack([lows, highs]),
            fmt="o",
            capsize=3,
            color=color,
            label=direction,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, ["Pokemon", "COCO-1K", "COCO-5K", "Flickr8k", "Flickr30k"])
    ax.set_ylabel("Absolute R@1 change (percentage points)")
    ax.set_title("ADR FluidMix improvement with paired 95% bootstrap CI")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "adr_r1_confidence_intervals.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    ablation_names = [
        "adr",
        "adr_without_advection",
        "adr_without_diffusion",
        "adr_without_reaction",
        "parameter_matched_mlp",
    ]
    ablation = (
        models[models["model"].isin(ablation_names)]
        .groupby("model")[["r@1", "r@5", "r@10"]]
        .mean()
        .loc[ablation_names]
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    width = 0.24
    positions = np.arange(len(ablation))
    for index, metric in enumerate(("r@1", "r@5", "r@10")):
        ax.bar(
            positions + (index - 1) * width,
            ablation[metric].values * 100,
            width,
            label=metric.upper(),
        )
    ax.set_xticks(positions, [MODEL_LABELS[name] for name in ablation.index], rotation=18, ha="right")
    ax.set_ylabel("Mean recall (%)")
    ax.set_title("ADR operator ablation and parameter-matched control")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "adr_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    noise_models = ["clip_anchor", "adr", "bidirectional_cross_attention", "parameter_matched_mlp"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for model, color in zip(noise_models, ("#505050", "#176b87", "#c45d37", "#798c43")):
        values = noise[noise["model"] == model].groupby("severity")["r@1"].mean().sort_index()
        ax.plot(values.index, values.values * 100, marker="o", label=MODEL_LABELS[model], color=color)
    ax.set_xticks([0.1, 0.2, 0.3])
    ax.set_xlabel("Text corruption severity")
    ax.set_ylabel("Mean R@1 (%)")
    ax.set_title("Retrieval under lexical text corruption")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "noisy_text_robustness.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    performance = merged.groupby("model")["delta_r1"].mean()
    efficiency_plot = efficiency[efficiency["model"].isin(main_models)].copy()
    efficiency_plot["mean_delta_r1"] = efficiency_plot["model"].map(performance) * 100
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for _, row in efficiency_plot.iterrows():
        color = "#176b87" if row["model"] == "adr" else "#8a98a6"
        ax.scatter(
            row["single_query_topk_latency_median_ms"],
            row["mean_delta_r1"],
            s=60,
            color=color,
        )
        label_offsets = {
            "low_rank_bilinear": (5, 5),
            "gated_multimodal": (5, -14),
            "parameter_matched_mlp": (5, 5),
            "adr": (-88, 7),
        }
        ax.annotate(
            MODEL_LABELS[row["model"]],
            (row["single_query_topk_latency_median_ms"], row["mean_delta_r1"]),
            xytext=label_offsets.get(row["model"], (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Single-query top-200 latency, median (ms)")
    ax.set_ylabel("Mean R@1 change vs CLIP (percentage points)")
    ax.set_title("Accuracy-efficiency trade-off")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "efficiency_tradeoff.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/q1_extended")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    results = Path(args.results)
    if not results.is_absolute():
        results = ROOT / results

    models, clip = load_test_metrics(results)
    datasets = sorted(models["dataset"].unique())
    adr_stats = []
    for dataset in datasets:
        path = results / "runs" / dataset / "adr" / "statistics_vs_clip.csv"
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        adr_stats.append(frame)
    adr_stats_frame = pd.concat(adr_stats, ignore_index=True)

    selected_comparisons = []
    for baseline in ("bidirectional_cross_attention", "parameter_matched_mlp"):
        selected_comparisons.extend(
            paired_comparison_rows(
                results,
                datasets,
                baseline,
                "adr",
                ("r@1", "r@5", "r@10"),
                int(args.bootstrap),
            )
        )
    comparison_frame = pd.DataFrame(selected_comparisons)
    comparison_frame.to_csv(results / "statistics_adr_vs_selected_baselines.csv", index=False)

    ablation_comparisons = []
    for baseline in (
        "adr_without_advection",
        "adr_without_diffusion",
        "adr_without_reaction",
    ):
        ablation_comparisons.extend(
            paired_comparison_rows(
                results,
                datasets,
                baseline,
                "adr",
                ("r@1",),
                int(args.bootstrap),
            )
        )
    ablation_stats = pd.DataFrame(ablation_comparisons)
    ablation_stats.to_csv(results / "statistics_adr_vs_ablations.csv", index=False)

    robustness_dir = results / "robustness"
    cross_domain = pd.read_csv(robustness_dir / "cross_domain.csv")
    noise = pd.read_csv(robustness_dir / "noisy_text.csv")
    mismatch = pd.read_csv(robustness_dir / "mismatched_descriptions.csv")
    efficiency = pd.read_csv(results / "efficiency.csv")
    build_figures(results, models, clip, adr_stats_frame, noise, efficiency)

    merged = models.merge(
        clip[["dataset", "direction", "r@1", "r@5", "r@10"]],
        on=["dataset", "direction"],
        suffixes=("", "_clip"),
    )
    for metric in ("r@1", "r@5", "r@10"):
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_clip"]
    main_models = [
        "symmetric_mlp",
        "low_rank_bilinear",
        "gated_multimodal",
        "bidirectional_cross_attention",
        "late_interaction",
        "transformer_cross_encoder",
        "parameter_matched_mlp",
        "adr",
    ]
    main_summary = (
        merged[merged["model"].isin(main_models)]
        .groupby("model")
        .agg(
            mean_delta_r1=("delta_r@1", "mean"),
            mean_delta_r5=("delta_r@5", "mean"),
            mean_delta_r10=("delta_r@10", "mean"),
            r1_wins=("delta_r@1", lambda values: int((values > 0).sum())),
        )
        .sort_values("mean_delta_r1", ascending=False)
        .reset_index()
    )
    main_summary["model"] = main_summary["model"].map(MODEL_LABELS)

    adr_detail = merged[merged["model"] == "adr"][
        [
            "dataset",
            "direction",
            "mix_weight",
            "r@1_clip",
            "r@1",
            "delta_r@1",
            "r@5_clip",
            "r@5",
            "delta_r@5",
            "r@10_clip",
            "r@10",
            "delta_r@10",
        ]
    ].copy()
    ablation = (
        models[
            models["model"].isin(
                [
                    "adr",
                    "adr_without_advection",
                    "adr_without_diffusion",
                    "adr_without_reaction",
                    "parameter_matched_mlp",
                ]
            )
        ]
        .groupby("model")[["r@1", "r@5", "r@10"]]
        .mean()
        .reset_index()
    )
    ablation["model"] = ablation["model"].map(MODEL_LABELS)

    split_audit = pd.read_csv(results / "protocol" / "split_manifest_audit.csv")
    overlap_audit = pd.read_csv(results / "protocol" / "split_overlap_audit.csv")
    natural_domain = cross_domain[cross_domain["tested_on"] != "pokemon"]
    cross_summary = (
        natural_domain.groupby("model")[["r@1", "r@5", "r@10"]]
        .mean()
        .sort_values("r@1", ascending=False)
        .reset_index()
    )
    cross_summary["model"] = cross_summary["model"].map(MODEL_LABELS)
    noise_summary = (
        noise.groupby(["model", "severity"])[["r@1", "r@1_degradation"]]
        .mean()
        .reset_index()
    )
    noise_summary = noise_summary[
        noise_summary["model"].isin(
            ["clip_anchor", "adr", "bidirectional_cross_attention", "parameter_matched_mlp"]
        )
    ]
    noise_summary["model"] = noise_summary["model"].map(MODEL_LABELS)
    mismatch_summary = mismatch[mismatch["mismatch_ratio"] == 0.5][
        ["model", "mismatch_auc", "balanced_accuracy", "rejection_rate_on_mismatch"]
    ].sort_values("mismatch_auc", ascending=False)
    mismatch_summary["model"] = mismatch_summary["model"].map(MODEL_LABELS)

    significant_vs_clip = int(adr_stats_frame["significant_0.05"].sum())
    comparison_frame["significant_adr_win"] = (
        comparison_frame["significant_0.05"] & (comparison_frame["delta"] > 0)
    )
    comparison_frame["significant_adr_loss"] = (
        comparison_frame["significant_0.05"] & (comparison_frame["delta"] < 0)
    )
    comparison_summary = (
        comparison_frame.groupby("baseline")
        .agg(
            mean_delta=("delta", "mean"),
            significant_adr_wins=("significant_adr_win", "sum"),
            significant_adr_losses=("significant_adr_loss", "sum"),
        )
        .reset_index()
    )
    comparison_summary["baseline"] = comparison_summary["baseline"].map(MODEL_LABELS)
    efficiency_selected = efficiency[
        efficiency["model"].isin(
            [
                "adr",
                "bidirectional_cross_attention",
                "parameter_matched_mlp",
                "low_rank_bilinear",
            ]
        )
    ][
        [
            "model",
            "trainable_parameters",
            "estimated_flops_per_pair",
            "peak_inference_vram_mb_topk",
            "peak_training_vram_mb_512_pairs",
            "single_query_topk_latency_median_ms",
            "single_query_topk_latency_p95_ms",
            "training_seconds",
        ]
    ].copy()
    efficiency_selected["model"] = efficiency_selected["model"].map(MODEL_LABELS)

    lines = [
        "# Q1 补充实验：统一二阶段重排、统计检验与鲁棒性报告",
        "",
        "## 1. 可执行协议",
        "",
        "- Observation：冻结 CLIP ViT-B/32 的图像/文本嵌入，形状分别为 `[B, 512]`；第一阶段仅用 cosine similarity 产生 top-K。",
        "- State：ADR 将每个模态投影为 `[B, 128, 4, 4]` 浓度场，显式 Euler 迭代 4 步，`dt=0.2`。",
        "- Action/输出：本任务不是 RL；模型输出候选图文对的标量重排分数。",
        "- Loss：双向 listwise cross-entropy + `0.05` anchor geometry loss + `0.001` residual-scale penalty。",
        "- 候选协议：训练集每个查询 12 个冻结 CLIP hard negatives；测试阶段 top-200，Pokémon 因库大小使用 top-80。",
        "- 训练与选择：固定 seed 13，3 epochs；每个 epoch 仅在 validation 上选择 checkpoint 与 `lambda in [0,2]`（步长 0.05），`lambda=0` 是严格 CLIP 回退。",
        "- 测试隔离：test 只在 checkpoint 与 lambda 固定后读取；逐查询结果用于 paired bootstrap 与 exact McNemar，六项指标内使用 Holm correction。",
        "",
        "### 固定划分",
        "",
        markdown_table(split_audit[["dataset", "split", "samples", "unique_images", "manifest_sha256", "protocol"]], 0),
        "",
        f"跨 split 图像重叠与 source-ID 重叠均为 0：`{bool((overlap_audit[['image_overlap', 'source_id_overlap']] == 0).all().all())}`。",
        "",
        "## 2. 二阶段交互基线",
        "",
        "新增六个真实 top-K 二阶段头：symmetric MLP、low-rank bilinear、gated multimodal fusion、bidirectional cross-attention、late interaction、Transformer cross-encoder。它们均只重排相同的第一阶段候选，并使用相同训练损失与验证选择。这里是机制级公平实现，不声称逐层复现 SCAN、VSE++ 等原论文。",
        "",
        markdown_table(main_summary, 4),
        "",
        "## 3. ADR 主结果",
        "",
        markdown_table(adr_detail, 4),
        "",
        f"ADR 相对 CLIP 的 30 项成对比较中，Holm 校正后显著项为 `{significant_vs_clip}/30`。平均绝对 R@1 提升为 `{float(main_summary.loc[main_summary['model'] == 'ADR FluidMix', 'mean_delta_r1'].iloc[0]) * 100:.2f}` 个百分点。",
        "",
        "### 与验证集选出的强交互基线和等参数基线比较",
        "",
        markdown_table(comparison_summary, 4),
        "",
        "## 4. ADR 三算子消融与参数量匹配",
        "",
        markdown_table(ablation, 4),
        "",
        "完整 ADR 优于等参数 MLP，但三项逐算子消融差异很小；去除 reaction 的平均 R@1 反而略高。当前证据支持 ADR 场结构与参数量无关的收益，但不支持“三个算子都不可替代”的强结论。该负结果应在论文中如实保留。",
        "",
        "## 5. 效率",
        "",
        markdown_table(efficiency_selected, 4),
        "",
        "FLOPs 为模块钩子覆盖的 Linear、Conv、Attention 与显式 ADR 有限差分运算估计；显存、训练时间和 top-200 延迟为 RTX 5070 Ti 实测。训练时间包含三轮训练与 validation 选模，不包含一次性冻结 CLIP 特征编码。",
        "",
        "## 6. 鲁棒性",
        "",
        "### 跨域",
        "",
        markdown_table(cross_summary, 4),
        "",
        "自然图像目标域上 ADR 与 low-rank bilinear 并列最高；在 Pokémon 极端卡通域偏移上 ADR 低于 CLIP，因此不能宣称任意域都占优。",
        "",
        "### 噪声文本",
        "",
        markdown_table(noise_summary, 4),
        "",
        "噪声由 word deletion、adjacent swap 与 internal-character transposition 组成。ADR 在三个强度下的绝对 R@1 均高于 CLIP，但相对 clean 的降幅不是所有模型中最小，合理结论是“噪声下仍保持重排增益”。",
        "",
        "### 错配描述",
        "",
        markdown_table(mismatch_summary, 4),
        "",
        "错配采用固定循环置换（offset 997），validation 选择阈值，test 报告 AUC、balanced accuracy 与 50% 错配时 rejection rate。ADR 的 AUC 与去对流版本并列最高。",
        "",
        "## 7. 结论边界",
        "",
        "1. 已证实：统一候选与统一选择协议下，ADR 在五数据集的 10 个方向上均提高 R@1，且大多数提升具有成对统计显著性。",
        "2. 已证实：ADR 的主结果高于等参数 MLP，不能仅用参数量增加解释。",
        "3. 部分证实：ADR 在自然图像跨域、噪声文本和可控错配下保持强表现；极端 Pokémon 域偏移失败。",
        "4. 未证实：reaction 算子在当前超参数下不可替代。消融结果提示下一轮应重新研究 reaction 的稳定化形式或系数约束。",
        "",
        "## 8. 最终文件",
        "",
        "- `aggregate_metrics.csv`：五数据集所有测试结果。",
        "- `statistics_adr_vs_selected_baselines.csv`：ADR 对 cross-attention 与等参数 MLP 的 paired bootstrap/McNemar/Holm。",
        "- `statistics_adr_vs_ablations.csv`：ADR 三算子逐项比较。",
        "- `efficiency.csv`：参数、FLOPs、显存、时间与延迟。",
        "- `robustness/`：跨域、噪声文本、错配描述结果与协议。",
        "- `protocol/`：固定划分 hash、重叠审计与候选缓存。",
        "- `figures/`：最终论文级结果图。",
        "",
        "## 参考机制",
        "",
        "- Faghri et al., VSE++: Improving Visual-Semantic Embeddings with Hard Negatives, 2018: https://arxiv.org/abs/1707.05612",
        "- Lee et al., Stacked Cross Attention for Image-Text Matching, ECCV 2018: https://openaccess.thecvf.com/content_ECCV_2018/html/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.html",
        "- Zhang et al., Negative-Aware Attention Framework for Image-Text Matching, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Negative-Aware_Attention_Framework_for_Image-Text_Matching_CVPR_2022_paper.html",
    ]
    report_path = results / "Q1_extended_experiment_report_CN.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
