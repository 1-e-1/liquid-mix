from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = (
    "test_i2t_r@1",
    "test_i2t_r@5",
    "test_i2t_r@10",
    "test_t2i_r@1",
    "test_t2i_r@5",
    "test_t2i_r@10",
)
METRIC_ORDER = {metric: idx for idx, metric in enumerate(METRIC_COLUMNS)}


def resolve_root_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def infer_seed(path: Path, metadata: dict) -> int:
    if "seed" in metadata:
        return int(metadata["seed"])
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else 13


def read_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_result(path: Path) -> list[dict]:
    summary_path = path / "clip_lora_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing CLIP-LoRA summary: {summary_path}")

    metadata = read_metadata(path)
    summary = pd.read_csv(summary_path)
    if set(summary["model"]) < {"clip_zero_shot_openai", "clip_lora_finetuned"}:
        raise ValueError(f"summary must contain zero-shot and LoRA rows: {summary_path}")

    clip = summary[summary["model"] == "clip_zero_shot_openai"].iloc[0]
    lora = summary[summary["model"] == "clip_lora_finetuned"].iloc[0]
    dataset = metadata.get("dataset", path.name)
    seed = infer_seed(path, metadata)
    rows = []
    for metric in METRIC_COLUMNS:
        clip_value = float(clip[metric])
        lora_value = float(lora[metric])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "metric": metric.replace("test_", ""),
                "clip": clip_value,
                "lora": lora_value,
                "delta": lora_value - clip_value,
                "trainable_params": int(lora["trainable_params"]),
                "total_params": int(lora["total_params"]),
                "epochs": int(lora["epochs"]),
                "source_dir": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            }
        )
    return rows


def add_metric_rank(df: pd.DataFrame) -> pd.DataFrame:
    order = {metric.replace("test_", ""): rank for metric, rank in METRIC_ORDER.items()}
    out = df.copy()
    out["_metric_rank"] = out["metric"].map(order).fillna(len(order)).astype(int)
    return out


def format_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, per_seed: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    positive = int((per_seed["delta"] > 0).sum())
    total = int(len(per_seed))
    compact = aggregate[
        [
            "dataset",
            "seeds",
            "metric",
            "clip_mean",
            "lora_mean",
            "delta_mean",
            "delta_std",
            "delta_min",
            "delta_max",
        ]
    ]
    lines = [
        "# CLIP-LoRA Multi-seed Baseline Summary",
        "",
        "This report aggregates matched PEFT baselines trained with the same lightweight two-epoch LoRA budget.",
        "Deltas are LoRA minus zero-shot CLIP on the same test split.",
        "",
        f"Positive per-seed metric deltas: `{positive}/{total}`.",
        "",
        "## Mean and Range",
        "",
        format_table(compact, list(compact.columns)),
        "",
        "## Per-seed Results",
        "",
        format_table(per_seed, list(per_seed.columns)),
    ]
    (out_dir / "clip_lora_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/q1_clip_lora_coco1k_seed7",
            "results/q1_clip_lora_coco1k_seed13",
            "results/q1_clip_lora_coco1k_seed21",
            "results/q1_clip_lora_flickr8k_full_seed7",
            "results/q1_clip_lora_flickr8k_full_seed13",
            "results/q1_clip_lora_flickr8k_full_seed21",
            "results/q1_clip_lora_flickr30k_large_seed7",
            "results/q1_clip_lora_flickr30k_large_seed13",
            "results/q1_clip_lora_flickr30k_large_seed21",
        ],
    )
    parser.add_argument("--output", default="results/q1_clip_lora_multiseed")
    args = parser.parse_args()

    out_dir = resolve_root_path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw_path in args.inputs:
        rows.extend(load_result(resolve_root_path(raw_path)))

    per_seed = add_metric_rank(pd.DataFrame(rows)).sort_values(["dataset", "seed", "_metric_rank"])
    aggregate = (
        per_seed.groupby(["dataset", "metric"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            clip_mean=("clip", "mean"),
            lora_mean=("lora", "mean"),
            lora_std=("lora", "std"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            delta_min=("delta", "min"),
            delta_max=("delta", "max"),
            trainable_params=("trainable_params", "max"),
            total_params=("total_params", "max"),
            epochs=("epochs", "max"),
        )
    )
    aggregate = add_metric_rank(aggregate).sort_values(["dataset", "_metric_rank"])

    per_seed.drop(columns=["_metric_rank"]).to_csv(out_dir / "per_seed_results.csv", index=False)
    aggregate.drop(columns=["_metric_rank"]).to_csv(out_dir / "aggregate_results.csv", index=False)
    write_report(
        out_dir,
        per_seed.drop(columns=["_metric_rank"]),
        aggregate.drop(columns=["_metric_rank"]),
    )
    print(aggregate.drop(columns=["_metric_rank"]).to_string(index=False))
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
