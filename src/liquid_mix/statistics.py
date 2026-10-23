from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
import torch


def paired_retrieval_indicator(score_matrix: torch.Tensor, k: int) -> torch.Tensor:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("score_matrix must be square [num_images, num_texts]")
    kk = min(int(k), score_matrix.shape[1])
    order = torch.argsort(score_matrix, dim=1, descending=True)
    target = torch.arange(score_matrix.shape[0], device=score_matrix.device).view(-1, 1)
    return (order[:, :kk] == target).any(dim=1).to(torch.float32).cpu()


def paired_bootstrap_metric_delta(
    baseline_values: torch.Tensor,
    improved_values: torch.Tensor,
    n_boot: int = 10000,
    seed: int = 13,
    ci: float = 0.95,
) -> dict[str, float]:
    baseline = baseline_values.detach().cpu().numpy().astype(np.float64).reshape(-1)
    improved = improved_values.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if baseline.shape != improved.shape:
        raise ValueError("baseline_values and improved_values must have the same shape")
    if baseline.size == 0:
        raise ValueError("metric arrays must be non-empty")
    rng = np.random.default_rng(int(seed))
    paired_delta = improved - baseline
    delta_samples = np.empty(int(n_boot), dtype=np.float64)
    bootstrap_chunk = 256
    for start in range(0, int(n_boot), bootstrap_chunk):
        count = min(bootstrap_chunk, int(n_boot) - start)
        indices = rng.integers(0, baseline.size, size=(count, baseline.size))
        delta_samples[start : start + count] = paired_delta[indices].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    return {
        "baseline_mean": float(baseline.mean()),
        "improved_mean": float(improved.mean()),
        "delta": float(improved.mean() - baseline.mean()),
        "ci_low": float(np.quantile(delta_samples, alpha)),
        "ci_high": float(np.quantile(delta_samples, 1.0 - alpha)),
        "prob_delta_gt_0": float((delta_samples > 0.0).mean()),
        "n": int(baseline.size),
        "n_boot": int(n_boot),
    }


def exact_paired_mcnemar(
    baseline_values: torch.Tensor,
    improved_values: torch.Tensor,
) -> dict[str, float | int]:
    baseline = baseline_values.detach().cpu().numpy().astype(bool).reshape(-1)
    improved = improved_values.detach().cpu().numpy().astype(bool).reshape(-1)
    if baseline.shape != improved.shape:
        raise ValueError("baseline_values and improved_values must have the same shape")
    baseline_only = int(np.logical_and(baseline, np.logical_not(improved)).sum())
    improved_only = int(np.logical_and(np.logical_not(baseline), improved).sum())
    discordant = baseline_only + improved_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(improved_only, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "baseline_only": baseline_only,
        "improved_only": improved_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()
