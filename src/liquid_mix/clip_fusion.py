from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F

from .models import FusionConfig, build_fusion


@dataclass
class CLIPAdapterConfig:
    method: str = "fluidmix"
    clip_dim: int = 512
    hidden_dim: int = 128
    field_grid: int = 4
    steps: int = 4
    dt: float = 0.2
    diffusion: float = 0.03
    reaction: float = 0.5
    num_heads: int = 4
    use_advection: bool = True
    use_diffusion: bool = True
    use_reaction: bool = True


class CLIPFusionAdapter(nn.Module):
    """Trainable fusion head on top of frozen CLIP image/text embeddings."""

    def __init__(self, config: CLIPAdapterConfig):
        super().__init__()
        self.config = config
        self.image_proj = nn.Sequential(
            nn.Linear(config.clip_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.clip_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        field_dim = config.hidden_dim * config.field_grid * config.field_grid
        self.image_field = nn.Linear(config.clip_dim, field_dim)
        self.text_field = nn.Linear(config.clip_dim, field_dim)
        self.fusion = build_fusion(
            FusionConfig(
                name=config.method,
                hidden_dim=config.hidden_dim,
                field_grid=config.field_grid,
                steps=config.steps,
                dt=config.dt,
                diffusion=config.diffusion,
                reaction=config.reaction,
                num_heads=config.num_heads,
                use_advection=config.use_advection,
                use_diffusion=config.use_diffusion,
                use_reaction=config.use_reaction,
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        image_vec = self.image_proj(image_emb)
        text_vec = self.text_proj(text_emb)
        batch = image_emb.shape[0]
        image_field = torch.tanh(self.image_field(image_emb)).view(
            batch, self.config.hidden_dim, self.config.field_grid, self.config.field_grid
        )
        text_field = torch.tanh(self.text_field(text_emb)).view(
            batch, self.config.hidden_dim, self.config.field_grid, self.config.field_grid
        )
        fused = self.fusion(image_vec, text_vec, image_field, text_field)
        return self.classifier(fused).squeeze(-1)


class ResidualCLIPFusionScorer(nn.Module):
    """CLIP cosine score plus a trainable fusion residual.

    The residual scale is initialized to zero, so the initial model exactly
    preserves the frozen CLIP ranking before adapter training.
    """

    def __init__(self, config: CLIPAdapterConfig, clip_scale: float = 1.0):
        super().__init__()
        self.adapter = CLIPFusionAdapter(config)
        self.clip_scale = float(clip_scale)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        clip_score = F.cosine_similarity(image_emb, text_emb, dim=-1) * self.clip_scale
        residual = self.adapter(image_emb, text_emb)
        return clip_score + self.residual_scale * residual


def make_pair_indices(num_samples: int, negative_offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_indices = []
    text_indices = []
    labels = []
    for idx in range(num_samples):
        image_indices.append(idx)
        text_indices.append(idx)
        labels.append(1.0)
        neg_idx = (idx + negative_offset) % num_samples
        if neg_idx == idx:
            neg_idx = (idx + 1) % num_samples
        image_indices.append(idx)
        text_indices.append(neg_idx)
        labels.append(0.0)
    return (
        torch.tensor(image_indices, dtype=torch.long),
        torch.tensor(text_indices, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def make_hard_negative_pair_indices(
    score_matrix: torch.Tensor,
    negatives_per_positive: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("score_matrix must be square [num_images, num_texts]")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be >= 1")
    n = int(score_matrix.shape[0])
    k = min(int(negatives_per_positive), n - 1)
    masked = score_matrix.detach().clone()
    masked[torch.arange(n), torch.arange(n)] = -torch.inf
    hard_text_indices = torch.topk(masked, k=k, dim=1).indices

    image_indices = []
    text_indices = []
    labels = []
    for idx in range(n):
        image_indices.append(idx)
        text_indices.append(idx)
        labels.append(1.0)
        for neg_idx in hard_text_indices[idx].tolist():
            image_indices.append(idx)
            text_indices.append(int(neg_idx))
            labels.append(0.0)
    return (
        torch.tensor(image_indices, dtype=torch.long),
        torch.tensor(text_indices, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def best_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores_np = scores.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    candidates = np.unique(scores_np)
    if candidates.size == 0:
        return 0.0
    best_acc = -1.0
    best_t = float(candidates[0])
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
    accuracy = float((preds == labels_np).mean())
    auc = math.nan if len(set(labels_np.tolist())) < 2 else float(roc_auc_score(labels_np, scores_np))
    return {"accuracy": accuracy, "auc": auc}


def recall_at_k(score_matrix: torch.Tensor, ks: Sequence[int] = (1, 5)) -> dict[str, float]:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("score_matrix must be square [num_images, num_texts]")
    order = torch.argsort(score_matrix, dim=1, descending=True)
    target = torch.arange(score_matrix.shape[0], device=score_matrix.device).view(-1, 1)
    out = {}
    for k in ks:
        kk = min(int(k), score_matrix.shape[1])
        out[f"r@{kk}"] = int((order[:, :kk] == target).any(dim=1).sum().item()) / int(score_matrix.shape[0])
    return out


def clip_cosine_scores(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    image_norm = F.normalize(image_emb, dim=-1)
    text_norm = F.normalize(text_emb, dim=-1)
    return image_norm @ text_norm.T


def symmetric_contrastive_loss(score_matrix: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("score_matrix must be square [batch, batch]")
    labels = torch.arange(score_matrix.shape[0], device=score_matrix.device)
    logits = score_matrix / float(temperature)
    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.T, labels)
    return 0.5 * (image_to_text + text_to_image)
