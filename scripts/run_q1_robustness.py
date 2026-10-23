from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import yaml

from liquid_mix.reranking_protocol import (
    build_candidate_cache,
    evaluate_candidates,
)
from run_clip_vlm_experiment import load_clip, resolve_device
from run_extended_q1_experiments import load_dataset_features, make_model, resolve


def load_trained_model(
    model_name: str,
    dataset_name: str,
    features: dict,
    config: dict,
    results_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, float]:
    model, _ = make_model(
        model_name,
        config["protocol"],
        int(features["train"]["image"].shape[1]),
        config["ablations"],
    )
    run_dir = results_dir / "runs" / dataset_name / model_name
    checkpoint = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    completed = json.loads((run_dir / "completed.json").read_text(encoding="utf-8"))
    return model.to(device).eval(), float(completed["selected_mix_weight"])


def perturb_caption(caption: str, severity: float, seed: int) -> str:
    rng = random.Random(int(seed))
    words = caption.split()
    kept = [word for word in words if rng.random() >= float(severity)]
    if not kept:
        kept = words[:1] or [caption]
    index = 0
    while index + 1 < len(kept):
        if rng.random() < float(severity) * 0.5:
            kept[index], kept[index + 1] = kept[index + 1], kept[index]
            index += 2
        else:
            index += 1
    for index, word in enumerate(kept):
        if len(word) > 3 and rng.random() < float(severity) * 0.5:
            position = rng.randrange(1, len(word) - 1)
            chars = list(word)
            chars[position - 1], chars[position] = chars[position], chars[position - 1]
            kept[index] = "".join(chars)
    return " ".join(kept)


def encode_texts(
    model,
    processor,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for start in tqdm(range(0, len(texts), int(batch_size)), desc="encode_noisy_text"):
        inputs = processor(
            text=texts[start : start + int(batch_size)],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_out = model.text_model(**inputs)
            outputs.append(model.text_projection(text_out.pooler_output).detach().cpu().float())
    return torch.cat(outputs)


def score_pairs(
    model: torch.nn.Module | None,
    image: torch.Tensor,
    text: torch.Tensor,
    image_index: torch.Tensor,
    text_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
    mix_weight: float = 1.0,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(image_index), int(batch_size)):
            ii = image_index[start : start + int(batch_size)]
            ti = text_index[start : start + int(batch_size)]
            pair_image = image[ii].to(device)
            pair_text = text[ti].to(device)
            anchor = F.cosine_similarity(pair_image, pair_text, dim=-1)
            values = (
                anchor
                if model is None
                else (1.0 - float(mix_weight)) * anchor + float(mix_weight) * model(pair_image, pair_text)
            )
            outputs.append(values.detach().cpu())
    return torch.cat(outputs)


def best_balanced_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    quantiles = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    best_score = -1.0
    best = float(quantiles[0])
    for threshold in quantiles:
        value = balanced_accuracy_score(labels, scores >= threshold)
        if value > best_score:
            best_score = float(value)
            best = float(threshold)
    return best


def controlled_mismatch_scores(
    model: torch.nn.Module | None,
    features: dict,
    device: torch.device,
    batch_size: int,
    mix_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    count = features["image"].shape[0]
    image_index = torch.arange(count)
    positive_index = torch.arange(count)
    offset = 997 % count
    if offset == 0:
        offset = 1
    negative_index = (positive_index + offset) % count
    positive = score_pairs(
        model,
        features["image"],
        features["text"],
        image_index,
        positive_index,
        device,
        batch_size,
        mix_weight,
    )
    negative = score_pairs(
        model,
        features["image"],
        features["text"],
        image_index,
        negative_index,
        device,
        batch_size,
        mix_weight,
    )
    scores = torch.cat([positive, negative]).numpy()
    labels = np.concatenate([np.ones(count), np.zeros(count)])
    return scores, labels, negative_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extended_q1_experiments.yaml")
    parser.add_argument("--results", default="results/q1_extended")
    parser.add_argument("--source", default="coco_karpathy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    results_dir = resolve(args.results)
    robustness_dir = results_dir / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    source_spec = next(spec for spec in config["datasets"] if spec["name"] == args.source)
    source_features, source_rows, source_config_path = load_dataset_features(source_spec, device)
    defaults = list(
        dict.fromkeys(
            config["models"] + list(config["ablations"]) + ["parameter_matched_mlp"]
        )
    )
    model_names = defaults if args.models is None else [name.strip() for name in args.models.split(",")]
    model_names = [
        name
        for name in model_names
        if (results_dir / "runs" / args.source / name / "checkpoint.pt").exists()
    ]
    models = {
        name: load_trained_model(
            name,
            args.source,
            source_features,
            config,
            results_dir,
            device,
        )
        for name in model_names
    }

    cross_domain_rows = []
    for target_spec in config["datasets"]:
        target_name = target_spec["name"]
        if target_name == args.source:
            continue
        target_features, _, _ = load_dataset_features(target_spec, device)
        top_k = int(target_spec.get("top_k", config["protocol"]["top_k"]))
        candidates = build_candidate_cache(target_features["test"], top_k, device)
        clip_rows, _ = evaluate_candidates(
            None,
            target_features["test"],
            candidates,
            device,
            int(config["protocol"]["score_batch_size"]),
        )
        for model_name, model_payload in [("clip_anchor", (None, 0.0)), *models.items()]:
            model, mix_weight = model_payload
            rows = (
                clip_rows
                if model is None
                else evaluate_candidates(
                    model,
                    target_features["test"],
                    candidates,
                    device,
                    int(config["protocol"]["score_batch_size"]),
                    mix_weight=mix_weight,
                )[0]
            )
            for row in rows:
                cross_domain_rows.append(
                    {
                        "trained_on": args.source,
                        "tested_on": target_name,
                        "model": model_name,
                        **row,
                    }
                )
    pd.DataFrame(cross_domain_rows).to_csv(robustness_dir / "cross_domain.csv", index=False)

    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    clip_model, processor = load_clip(
        source_config["clip"]["model_name"],
        bool(source_config["clip"].get("local_files_only", True)),
        device,
    )
    noisy_rows = []
    clean_top_k = int(source_spec.get("top_k", config["protocol"]["top_k"]))
    clean_candidates = build_candidate_cache(source_features["test"], clean_top_k, device)
    clean_model_rows = {}
    for model_name, model_payload in [("clip_anchor", (None, 0.0)), *models.items()]:
        model, mix_weight = model_payload
        clean_model_rows[model_name] = evaluate_candidates(
            model,
            source_features["test"],
            clean_candidates,
            device,
            int(config["protocol"]["score_batch_size"]),
            mix_weight=mix_weight,
        )[0]
    for severity in (0.1, 0.2, 0.3):
        cache_path = robustness_dir / f"noisy_text_{severity:.1f}.pt"
        if cache_path.exists():
            noisy_text = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            captions = [
                perturb_caption(sample.caption, severity, int(config["protocol"]["seed"]) + index)
                for index, sample in enumerate(source_rows["test"])
            ]
            noisy_text = encode_texts(
                clip_model,
                processor,
                captions,
                int(source_config["clip"]["batch_size"]),
                device,
            )
            torch.save(noisy_text, cache_path)
        noisy_features = {
            "image": source_features["test"]["image"],
            "text": noisy_text,
        }
        noisy_candidates = build_candidate_cache(noisy_features, clean_top_k, device)
        for model_name, model_payload in [("clip_anchor", (None, 0.0)), *models.items()]:
            model, mix_weight = model_payload
            rows, _ = evaluate_candidates(
                model,
                noisy_features,
                noisy_candidates,
                device,
                int(config["protocol"]["score_batch_size"]),
                mix_weight=mix_weight,
            )
            clean_lookup = {row["direction"]: row for row in clean_model_rows[model_name]}
            for row in rows:
                clean = clean_lookup[row["direction"]]
                noisy_rows.append(
                    {
                        "dataset": args.source,
                        "severity": severity,
                        "model": model_name,
                        **row,
                        "r@1_degradation": float(clean["r@1"]) - float(row["r@1"]),
                        "r@5_degradation": float(clean["r@5"]) - float(row["r@5"]),
                        "r@10_degradation": float(clean["r@10"]) - float(row["r@10"]),
                    }
                )
    pd.DataFrame(noisy_rows).to_csv(robustness_dir / "noisy_text.csv", index=False)
    del clip_model

    mismatch_rows = []
    for model_name, model_payload in [("clip_anchor", (None, 0.0)), *models.items()]:
        model, mix_weight = model_payload
        val_scores, val_labels, _ = controlled_mismatch_scores(
            model,
            source_features["val"],
            device,
            int(config["protocol"]["score_batch_size"]),
            mix_weight,
        )
        threshold = best_balanced_threshold(val_scores, val_labels)
        test_scores, test_labels, mismatch_index = controlled_mismatch_scores(
            model,
            source_features["test"],
            device,
            int(config["protocol"]["score_batch_size"]),
            mix_weight,
        )
        count = source_features["test"]["image"].shape[0]
        positive_scores = test_scores[:count]
        negative_scores = test_scores[count:]
        base = {
            "dataset": args.source,
            "model": model_name,
            "mismatch_auc": float(roc_auc_score(test_labels, test_scores)),
            "mean_positive_negative_margin": float(np.mean(positive_scores - negative_scores)),
            "validation_threshold": threshold,
        }
        rng = np.random.default_rng(int(config["protocol"]["seed"]))
        order = rng.permutation(count)
        for ratio in (0.1, 0.25, 0.5):
            corrupt_count = int(round(count * ratio))
            corrupt = np.zeros(count, dtype=bool)
            corrupt[order[:corrupt_count]] = True
            mixed_scores = np.where(corrupt, negative_scores, positive_scores)
            labels = np.logical_not(corrupt)
            mismatch_rows.append(
                {
                    **base,
                    "mismatch_ratio": ratio,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels, mixed_scores >= threshold)
                    ),
                    "rejection_rate_on_mismatch": float(
                        (mixed_scores[corrupt] < threshold).mean()
                    ),
                }
            )
    pd.DataFrame(mismatch_rows).to_csv(robustness_dir / "mismatched_descriptions.csv", index=False)
    (robustness_dir / "protocol.json").write_text(
        json.dumps(
            {
                "source_training_domain": args.source,
                "cross_domain_target_validation_used": False,
                "noise": "word deletion + adjacent swap + internal-character transposition",
                "noise_severities": [0.1, 0.2, 0.3],
                "mismatch": "fixed cyclic caption derangement with offset 997; no image is paired with its own caption",
                "mismatch_threshold_selected_on": "source validation split only",
                "models": model_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
