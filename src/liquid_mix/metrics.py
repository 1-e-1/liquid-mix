from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits_np = logits.detach().float().cpu().numpy().reshape(-1)
    labels_np = labels.detach().float().cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float((preds == labels_np).mean()) if labels_np.size else math.nan
    if len(set(labels_np.tolist())) < 2:
        auc = math.nan
    else:
        auc = float(roc_auc_score(labels_np, probs))
    return {"accuracy": accuracy, "auc": auc}


def retrieval_recall_at_k(scores: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [num_images, num_captions] matrix")
    order = torch.argsort(scores, dim=1, descending=True)
    target = torch.arange(scores.shape[0], device=scores.device).view(-1, 1)
    recalls = {}
    for k in ks:
        k = min(int(k), scores.shape[1])
        hit_count = int((order[:, :k] == target).any(dim=1).sum().item())
        recalls[f"r@{k}"] = hit_count / int(scores.shape[0])
    return recalls
