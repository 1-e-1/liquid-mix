from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import yaml

from liquid_mix.rerankers import (
    RerankerConfig,
    build_reranker,
    closest_parameter_matched_mlp,
    trainable_parameter_count,
)
from liquid_mix.reranking_protocol import (
    build_candidate_cache,
    build_hard_candidates,
    evaluate_candidates,
    set_deterministic_seed,
    train_candidate_reranker,
)
from liquid_mix.statistics import (
    exact_paired_mcnemar,
    holm_adjust,
    paired_bootstrap_metric_delta,
)
from run_clip_vlm_experiment import load_or_create_features, resolve_device


BASELINE_MODELS = (
    "symmetric_mlp",
    "low_rank_bilinear",
    "gated_multimodal",
    "bidirectional_cross_attention",
    "late_interaction",
    "transformer_cross_encoder",
    "adr",
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_names(value: str | None, defaults: list[str]) -> list[str]:
    if not value:
        return defaults
    return [item.strip() for item in value.split(",") if item.strip()]


def load_dataset_features(
    dataset_spec: dict,
    device: torch.device,
) -> tuple[dict, dict, Path]:
    config_path = resolve(dataset_spec["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = resolve(config["dataset"]["processed_dir"])
    feature_dir = resolve(dataset_spec["feature_cache"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    features, train, val, test = load_or_create_features(config, data_dir, feature_dir, device)
    rows = {"train": train, "val": val, "test": test}
    return features, rows, config_path


def candidate_cache_for_dataset(
    dataset_name: str,
    features: dict,
    output_dir: Path,
    top_k: int,
    hard_candidates: int,
    device: torch.device,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    cache_path = output_dir / "protocol" / f"{dataset_name}_candidate_cache.pt"
    expected = {
        "sizes": {split: int(features[split]["image"].shape[0]) for split in ("train", "val", "test")},
        "top_k": int(top_k),
        "hard_candidates": int(hard_candidates),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") == expected:
            return (
                payload["val_selection"],
                payload["test"],
                payload["hard_text"],
                payload["hard_image"],
            )

    val_selection = build_candidate_cache(
        features["val"],
        top_k=top_k,
        device=device,
        query_count=min(1000, features["val"]["image"].shape[0]),
    )
    test = build_candidate_cache(features["test"], top_k=top_k, device=device)
    hard_text, hard_image = build_hard_candidates(
        features["train"]["image"],
        features["train"]["text"],
        hard_candidates,
        device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": expected,
            "val_selection": val_selection,
            "test": test,
            "hard_text": hard_text,
            "hard_image": hard_image,
        },
        cache_path,
    )
    return val_selection, test, hard_text, hard_image


def make_model(
    model_name: str,
    protocol: dict,
    clip_dim: int,
    ablations: dict,
) -> tuple[torch.nn.Module, dict]:
    base = {
        "name": model_name,
        "clip_dim": int(clip_dim),
        "hidden_dim": int(protocol["hidden_dim"]),
        "token_count": int(protocol["token_count"]),
        "num_heads": int(protocol["num_heads"]),
        "field_grid": int(protocol["field_grid"]),
        "steps": int(protocol["steps"]),
        "dt": float(protocol["dt"]),
        "diffusion": float(protocol["diffusion"]),
        "reaction": float(protocol["reaction"]),
    }
    if model_name in ablations:
        base["name"] = "adr"
        base.update(ablations[model_name])
    if model_name == "parameter_matched_mlp":
        adr = build_reranker(RerankerConfig(**{**base, "name": "adr"}))
        target = trainable_parameter_count(adr)
        model, width, relative_error = closest_parameter_matched_mlp(target, clip_dim)
        return model, {
            "name": model_name,
            "target_parameters": target,
            "matched_width": width,
            "relative_parameter_error": relative_error,
        }
    config = RerankerConfig(**base)
    return build_reranker(config), asdict(config)


def metric_rows(
    dataset: str,
    model_name: str,
    rows: list[dict],
    split: str,
    parameters: int,
    mix_weight: float | None = None,
) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "model": model_name,
            "split": split,
            "direction": row["direction"],
            "r@1": row["r@1"],
            "r@5": row["r@5"],
            "r@10": row["r@10"],
            "candidate_recall": row["candidate_recall"],
            "trainable_parameters": parameters,
            "mix_weight": mix_weight,
        }
        for row in rows
    ]


def statistical_comparison(
    baseline: dict[str, torch.Tensor],
    improved: dict[str, torch.Tensor],
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    rows = []
    for key in sorted(baseline):
        bootstrap = paired_bootstrap_metric_delta(
            baseline[key],
            improved[key],
            n_boot=bootstrap_samples,
            seed=seed,
        )
        test = exact_paired_mcnemar(baseline[key], improved[key])
        direction, metric = key.split("_", maxsplit=1)
        rows.append({"direction": direction, "metric": metric, **bootstrap, **test})
    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
        row["significant_0.05"] = bool(value < 0.05)
    return rows


def save_indicators(path: Path, indicators: dict[str, torch.Tensor]) -> None:
    np.savez_compressed(
        path,
        **{name.replace("@", "at"): values.detach().cpu().numpy() for name, values in indicators.items()},
    )


def run_one(
    dataset_name: str,
    model_name: str,
    features: dict,
    val_candidates: dict,
    test_candidates: dict,
    hard_text: torch.Tensor,
    hard_image: torch.Tensor,
    protocol: dict,
    ablations: dict,
    output_dir: Path,
    device: torch.device,
    resume: bool,
) -> list[dict]:
    run_dir = output_dir / "runs" / dataset_name / model_name
    completed = run_dir / "completed.json"
    metrics_file = run_dir / "metrics.csv"
    if resume and completed.exists() and metrics_file.exists():
        return pd.read_csv(metrics_file).to_dict("records")

    run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(int(protocol["seed"]))
    model, model_config = make_model(
        model_name,
        protocol,
        int(features["train"]["image"].shape[1]),
        ablations,
    )
    parameters = trainable_parameter_count(model)
    mix_weights = np.arange(
        float(protocol["mix_weight_min"]),
        float(protocol["mix_weight_max"]) + 0.5 * float(protocol["mix_weight_step"]),
        float(protocol["mix_weight_step"]),
    ).round(10).tolist()
    model, curves, training_seconds, selected_mix_weight = train_candidate_reranker(
        model,
        features["train"],
        features["val"],
        val_candidates,
        hard_text,
        hard_image,
        device,
        seed=int(protocol["seed"]),
        epochs=int(protocol["epochs"]),
        query_batch_size=int(protocol["query_batch_size"]),
        score_batch_size=int(protocol["score_batch_size"]),
        temperature=float(protocol["temperature"]),
        learning_rate=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        gradient_clip=float(protocol["gradient_clip"]),
        geometry_weight=float(protocol["geometry_weight"]),
        scale_weight=float(protocol["scale_weight"]),
        mix_weights=mix_weights,
    )
    val_rows, _ = evaluate_candidates(
        model,
        features["val"],
        val_candidates,
        device,
        int(protocol["score_batch_size"]),
        mix_weight=selected_mix_weight,
    )
    test_rows, test_indicators = evaluate_candidates(
        model,
        features["test"],
        test_candidates,
        device,
        int(protocol["score_batch_size"]),
        mix_weight=selected_mix_weight,
    )
    clip_rows, clip_indicators = evaluate_candidates(
        None,
        features["test"],
        test_candidates,
        device,
        int(protocol["score_batch_size"]),
    )
    stats = statistical_comparison(
        clip_indicators,
        test_indicators,
        int(protocol["bootstrap_samples"]),
        int(protocol["seed"]),
    )
    rows = (
        metric_rows(dataset_name, "clip_anchor", clip_rows, "test", 0, 0.0)
        + metric_rows(
            dataset_name,
            model_name,
            val_rows,
            "val_selection",
            parameters,
            selected_mix_weight,
        )
        + metric_rows(
            dataset_name,
            model_name,
            test_rows,
            "test",
            parameters,
            selected_mix_weight,
        )
    )
    pd.DataFrame(rows).to_csv(metrics_file, index=False)
    pd.DataFrame(stats).to_csv(run_dir / "statistics_vs_clip.csv", index=False)
    (run_dir / "curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    save_indicators(run_dir / "test_indicators.npz", test_indicators)
    torch.save({"model": model.state_dict(), "config": model_config}, run_dir / "checkpoint.pt")
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "model_config": model_config,
        "trainable_parameters": parameters,
        "training_seconds": training_seconds,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": int(protocol["seed"]),
        "test_read_after_selection": True,
        "selected_mix_weight": selected_mix_weight,
    }
    completed.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows


def write_split_audit(dataset_specs: list[dict], output_dir: Path) -> None:
    rows = []
    overlap_rows = []
    for spec in dataset_specs:
        config = yaml.safe_load(resolve(spec["config"]).read_text(encoding="utf-8"))
        data_dir = resolve(config["dataset"]["processed_dir"])
        split_images = {}
        split_ids = {}
        for split in ("train", "val", "test"):
            manifest = data_dir / f"{split}.jsonl"
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            split_images[split] = {Path(row["image_path"]).name.lower() for row in records}
            split_ids[split] = {str(row.get("source_id", "")) for row in records}
            rows.append(
                {
                    "dataset": spec["name"],
                    "split": split,
                    "samples": len(records),
                    "unique_images": len(split_images[split]),
                    "unique_source_ids": len(split_ids[split]),
                    "manifest_sha256": file_sha256(manifest),
                    "protocol": spec["split_protocol"],
                }
            )
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap_rows.append(
                {
                    "dataset": spec["name"],
                    "split_pair": f"{first}-{second}",
                    "image_overlap": len(split_images[first] & split_images[second]),
                    "source_id_overlap": len(split_ids[first] & split_ids[second]),
                }
            )
    protocol_dir = output_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(protocol_dir / "split_manifest_audit.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(protocol_dir / "split_overlap_audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Q1 two-stage reranking experiments")
    parser.add_argument("--config", default="configs/extended_q1_experiments.yaml")
    parser.add_argument("--output", default="results/q1_extended")
    parser.add_argument("--stage", choices=("baselines", "ablations", "all"), default="all")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if args.epochs is not None:
        protocol["epochs"] = int(args.epochs)
    output_dir = resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    selected_datasets = set(parse_names(args.datasets, [spec["name"] for spec in config["datasets"]]))
    dataset_specs = [spec for spec in config["datasets"] if spec["name"] in selected_datasets]
    write_split_audit(dataset_specs, output_dir)

    if args.stage == "baselines":
        defaults = list(BASELINE_MODELS)
    elif args.stage == "ablations":
        defaults = list(config["ablations"]) + ["parameter_matched_mlp"]
    else:
        defaults = list(dict.fromkeys(list(BASELINE_MODELS) + list(config["ablations"]) + ["parameter_matched_mlp"]))
    selected_models = parse_names(args.models, defaults)

    all_rows = []
    for dataset_spec in dataset_specs:
        dataset_name = dataset_spec["name"]
        features, _, config_path = load_dataset_features(dataset_spec, device)
        top_k = int(dataset_spec.get("top_k", protocol["top_k"]))
        val_candidates, test_candidates, hard_text, hard_image = candidate_cache_for_dataset(
            dataset_name,
            features,
            output_dir,
            top_k,
            int(protocol["hard_candidates"]),
            device,
        )
        dataset_record = {
            "dataset": dataset_name,
            "config": str(config_path.relative_to(ROOT)),
            "feature_cache": dataset_spec["feature_cache"],
            "top_k": top_k,
            "sizes": {split: int(features[split]["image"].shape[0]) for split in ("train", "val", "test")},
        }
        dataset_dir = output_dir / "protocol"
        (dataset_dir / f"{dataset_name}_protocol.json").write_text(
            json.dumps(dataset_record, indent=2),
            encoding="utf-8",
        )
        for model_name in selected_models:
            rows = run_one(
                dataset_name,
                model_name,
                features,
                val_candidates,
                test_candidates,
                hard_text,
                hard_image,
                protocol,
                config["ablations"],
                output_dir,
                device,
                args.resume,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)

    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": str(resolve(args.config)),
                "datasets": [spec["name"] for spec in dataset_specs],
                "models": selected_models,
                "device": str(device),
                "protocol": protocol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
