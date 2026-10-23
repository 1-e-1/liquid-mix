from __future__ import annotations

from copy import deepcopy
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


METRIC_KS = (1, 5, 10)
DIRECTIONS = ("i2t", "t2i")


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosine_topk(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    top_k: int,
    direction: str,
    device: torch.device,
    query_count: int | None = None,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_emb.ndim != 2 or text_emb.ndim != 2:
        raise ValueError("embeddings must be rank-2 tensors")
    if image_emb.shape[0] != text_emb.shape[0]:
        raise ValueError("paired retrieval expects the same number of images and texts")
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction}")
    queries = image_emb if direction == "i2t" else text_emb
    references = text_emb if direction == "i2t" else image_emb
    count = len(queries) if query_count is None else min(int(query_count), len(queries))
    k = min(int(top_k), len(references))
    references_device = F.normalize(references.to(device), dim=-1)
    all_indices = []
    all_values = []
    with torch.no_grad():
        for start in range(0, count, int(chunk_size)):
            query = F.normalize(queries[start : start + int(chunk_size)].to(device), dim=-1)
            values, indices = torch.topk(query @ references_device.T, k=k, dim=1)
            all_indices.append(indices.cpu())
            all_values.append(values.cpu())
    return torch.cat(all_indices), torch.cat(all_values)


def build_candidate_cache(
    features: dict[str, torch.Tensor],
    top_k: int,
    device: torch.device,
    query_count: int | None = None,
) -> dict[str, dict[str, torch.Tensor | float]]:
    payload: dict[str, dict[str, torch.Tensor | float]] = {}
    for direction in DIRECTIONS:
        indices, scores = cosine_topk(
            features["image"],
            features["text"],
            top_k,
            direction,
            device,
            query_count=query_count,
        )
        target = torch.arange(indices.shape[0]).view(-1, 1)
        payload[direction] = {
            "indices": indices,
            "clip_scores": scores,
            "candidate_recall": float((indices == target).any(dim=1).float().mean()),
        }
    return payload


def build_hard_candidates(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    requested = min(int(count) + 1, image_emb.shape[0])
    hard_text, _ = cosine_topk(image_emb, text_emb, requested, "i2t", device)
    hard_image, _ = cosine_topk(image_emb, text_emb, requested, "t2i", device)
    target = torch.arange(image_emb.shape[0]).view(-1, 1)

    def remove_target(values: torch.Tensor) -> torch.Tensor:
        rows = []
        for index in range(values.shape[0]):
            filtered = values[index][values[index] != target[index, 0]]
            rows.append(filtered[: int(count)])
        return torch.stack(rows)

    return remove_target(hard_text), remove_target(hard_image)


def score_candidate_pairs(
    model: nn.Module,
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    candidate_indices: torch.Tensor,
    direction: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    query_count, candidate_count = candidate_indices.shape
    if direction == "i2t":
        image_index = torch.arange(query_count).repeat_interleave(candidate_count)
        text_index = candidate_indices.reshape(-1)
    elif direction == "t2i":
        image_index = candidate_indices.reshape(-1)
        text_index = torch.arange(query_count).repeat_interleave(candidate_count)
    else:
        raise ValueError(f"unknown direction: {direction}")
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(image_index), int(batch_size)):
            ii = image_index[start : start + int(batch_size)]
            ti = text_index[start : start + int(batch_size)]
            outputs.append(model(image_emb[ii].to(device), text_emb[ti].to(device)).detach().cpu())
    return torch.cat(outputs).view(query_count, candidate_count)


def retrieval_indicators(
    candidate_indices: torch.Tensor,
    scores: torch.Tensor,
    ks: tuple[int, ...] = METRIC_KS,
) -> dict[str, torch.Tensor]:
    if candidate_indices.shape != scores.shape:
        raise ValueError("candidate_indices and scores must have the same shape")
    order = torch.argsort(scores, dim=1, descending=True)
    ranked = candidate_indices.gather(1, order)
    target = torch.arange(ranked.shape[0]).view(-1, 1)
    return {
        f"r@{k}": (ranked[:, : min(int(k), ranked.shape[1])] == target).any(dim=1).float()
        for k in ks
    }


def evaluate_candidates(
    model: nn.Module | None,
    features: dict[str, torch.Tensor],
    candidate_cache: dict[str, dict[str, torch.Tensor | float]],
    device: torch.device,
    batch_size: int,
    mix_weight: float | None = None,
) -> tuple[list[dict[str, float | str]], dict[str, torch.Tensor]]:
    rows = []
    indicators: dict[str, torch.Tensor] = {}
    for direction in DIRECTIONS:
        payload = candidate_cache[direction]
        indices = payload["indices"]
        clip_scores = payload["clip_scores"]
        if not isinstance(indices, torch.Tensor) or not isinstance(clip_scores, torch.Tensor):
            raise TypeError("candidate cache must contain tensors")
        if model is None:
            scores = clip_scores
        else:
            model_scores = score_candidate_pairs(
                model,
                features["image"],
                features["text"],
                indices,
                direction,
                device,
                batch_size,
            )
            scores = (
                model_scores
                if mix_weight is None
                else (1.0 - float(mix_weight)) * clip_scores + float(mix_weight) * model_scores
            )
        direction_indicators = retrieval_indicators(indices, scores)
        for metric, values in direction_indicators.items():
            indicators[f"{direction}_{metric}"] = values
        rows.append(
            {
                "direction": direction,
                **{metric: float(values.mean()) for metric, values in direction_indicators.items()},
                "candidate_recall": float(payload["candidate_recall"]),
            }
        )
    return rows, indicators


def select_mix_weight(
    model: nn.Module,
    features: dict[str, torch.Tensor],
    candidate_cache: dict[str, dict[str, torch.Tensor | float]],
    device: torch.device,
    batch_size: int,
    weights: list[float],
) -> tuple[float, float, list[dict[str, float | str]]]:
    model_scores = {}
    for direction in DIRECTIONS:
        payload = candidate_cache[direction]
        indices = payload["indices"]
        if not isinstance(indices, torch.Tensor):
            raise TypeError("candidate indices must be tensors")
        model_scores[direction] = score_candidate_pairs(
            model,
            features["image"],
            features["text"],
            indices,
            direction,
            device,
            batch_size,
        )
    best_weight = 0.0
    best_selection = -float("inf")
    best_rows: list[dict[str, float | str]] = []
    for weight in weights:
        rows = []
        for direction in DIRECTIONS:
            payload = candidate_cache[direction]
            indices = payload["indices"]
            clip_scores = payload["clip_scores"]
            if not isinstance(indices, torch.Tensor) or not isinstance(clip_scores, torch.Tensor):
                raise TypeError("candidate cache must contain tensors")
            scores = (1.0 - float(weight)) * clip_scores + float(weight) * model_scores[direction]
            direction_indicators = retrieval_indicators(indices, scores)
            rows.append(
                {
                    "direction": direction,
                    **{metric: float(values.mean()) for metric, values in direction_indicators.items()},
                    "candidate_recall": float(payload["candidate_recall"]),
                }
            )
        selection = sum(float(row[f"r@{k}"]) for row in rows for k in METRIC_KS)
        if selection > best_selection:
            best_weight = float(weight)
            best_selection = float(selection)
            best_rows = rows
    return best_weight, best_selection, best_rows


def _training_scores(
    model: nn.Module,
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    query_index: torch.Tensor,
    negative_index: torch.Tensor,
    direction: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.cat([query_index.view(-1, 1), negative_index[query_index]], dim=1)
    batch, count = candidates.shape
    if direction == "i2t":
        image = image_emb[query_index].repeat_interleave(count, dim=0)
        text = text_emb[candidates.reshape(-1)]
    elif direction == "t2i":
        image = image_emb[candidates.reshape(-1)]
        text = text_emb[query_index].repeat_interleave(count, dim=0)
    else:
        raise ValueError(f"unknown direction: {direction}")
    image = image.to(device)
    text = text.to(device)
    scores = model(image, text).view(batch, count)
    anchors = F.cosine_similarity(image, text, dim=-1).view(batch, count)
    return scores, anchors


def train_candidate_reranker(
    model: nn.Module,
    train_features: dict[str, torch.Tensor],
    val_features: dict[str, torch.Tensor],
    val_candidates: dict[str, dict[str, torch.Tensor | float]],
    hard_text: torch.Tensor,
    hard_image: torch.Tensor,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    query_batch_size: int,
    score_batch_size: int,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    geometry_weight: float,
    scale_weight: float,
    mix_weights: list[float],
) -> tuple[nn.Module, list[dict[str, float]], float, float]:
    set_deterministic_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    indices = torch.arange(train_features["image"].shape[0])
    loader = DataLoader(
        TensorDataset(indices),
        batch_size=int(query_batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    best_score = -float("inf")
    best_state = deepcopy(model.state_dict())
    best_mix_weight = 0.0
    curves: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        ranking_losses = []
        geometry_losses = []
        for (query_index,) in loader:
            optimizer.zero_grad(set_to_none=True)
            i2t_scores, i2t_anchor = _training_scores(
                model,
                train_features["image"],
                train_features["text"],
                query_index,
                hard_text,
                "i2t",
                device,
            )
            t2i_scores, t2i_anchor = _training_scores(
                model,
                train_features["image"],
                train_features["text"],
                query_index,
                hard_image,
                "t2i",
                device,
            )
            target = torch.zeros(len(query_index), dtype=torch.long, device=device)
            ranking_loss = 0.5 * (
                F.cross_entropy(i2t_scores / float(temperature), target)
                + F.cross_entropy(t2i_scores / float(temperature), target)
            )
            geometry_loss = 0.5 * (
                F.mse_loss(i2t_scores, i2t_anchor) + F.mse_loss(t2i_scores, t2i_anchor)
            )
            residual_scale = getattr(model, "residual_scale", torch.zeros((), device=device))
            loss = (
                ranking_loss
                + float(geometry_weight) * geometry_loss
                + float(scale_weight) * residual_scale.pow(2)
            )
            loss.backward()
            if float(gradient_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ranking_losses.append(float(ranking_loss.detach().cpu()))
            geometry_losses.append(float(geometry_loss.detach().cpu()))
        epoch_mix_weight, selection, val_rows = select_mix_weight(
            model,
            val_features,
            val_candidates,
            device,
            score_batch_size,
            mix_weights,
        )
        curves.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "ranking_loss": float(np.mean(ranking_losses)),
                "geometry_loss": float(np.mean(geometry_losses)),
                "val_selection": float(selection),
                "val_mix_weight": float(epoch_mix_weight),
                "residual_scale": float(getattr(model, "residual_scale").detach().cpu()),
            }
        )
        if selection > best_score:
            best_score = float(selection)
            best_mix_weight = float(epoch_mix_weight)
            best_state = deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, curves, float(time.perf_counter() - started), best_mix_weight
