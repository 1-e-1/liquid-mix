from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .clip_fusion import CLIPAdapterConfig, ResidualCLIPFusionScorer


@dataclass
class RerankerConfig:
    name: str
    clip_dim: int = 512
    hidden_dim: int = 128
    token_count: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    field_grid: int = 4
    steps: int = 4
    dt: float = 0.2
    diffusion: float = 0.03
    reaction: float = 0.5
    use_advection: bool = True
    use_diffusion: bool = True
    use_reaction: bool = True
    mlp_width: int | None = None


class AnchoredPairReranker(nn.Module):
    """Frozen cosine anchor plus a trainable pairwise second-stage score."""

    def __init__(self, interaction: nn.Module, scale_init: float = 0.1):
        super().__init__()
        self.interaction = interaction
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        anchor = F.cosine_similarity(image_emb, text_emb, dim=-1)
        return anchor + self.residual_scale * self.interaction(image_emb, text_emb)


class SymmetricMLPInteraction(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        image = F.normalize(image_emb, dim=-1)
        text = F.normalize(text_emb, dim=-1)
        cosine = (image * text).sum(dim=-1, keepdim=True)
        features = torch.cat([image, text, (image - text).abs(), image * text, cosine], dim=-1)
        return self.net(features).squeeze(-1)


class LowRankBilinearInteraction(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.image_proj = nn.Linear(dim, hidden, bias=False)
        self.text_proj = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        interaction = self.image_proj(image_emb) * self.text_proj(text_emb)
        return self.out(interaction).squeeze(-1)


class GatedMultimodalInteraction(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.image_proj = nn.Linear(dim, hidden)
        self.text_proj = nn.Linear(dim, hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        image = torch.tanh(self.image_proj(image_emb))
        text = torch.tanh(self.text_proj(text_emb))
        gate = torch.sigmoid(self.gate(torch.cat([image, text], dim=-1)))
        fused = gate * image + (1.0 - gate) * text
        return self.out(torch.cat([fused, (image - text).abs(), image * text], dim=-1)).squeeze(-1)


class BidirectionalCrossAttentionInteraction(nn.Module):
    def __init__(self, dim: int, hidden: int, token_count: int, heads: int, dropout: float):
        super().__init__()
        self.hidden = hidden
        self.token_count = token_count
        heads = heads if hidden % heads == 0 else 1
        self.image_tokens = nn.Linear(dim, hidden * token_count)
        self.text_tokens = nn.Linear(dim, hidden * token_count)
        self.image_to_text = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.text_to_image = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch = image_emb.shape[0]
        image = self.image_tokens(image_emb).view(batch, self.token_count, self.hidden)
        text = self.text_tokens(text_emb).view(batch, self.token_count, self.hidden)
        image_context, _ = self.image_to_text(image, text, text, need_weights=False)
        text_context, _ = self.text_to_image(text, image, image, need_weights=False)
        image_pair = torch.cat([image.mean(1), image_context.mean(1)], dim=-1)
        text_pair = torch.cat([text.mean(1), text_context.mean(1)], dim=-1)
        return self.out(torch.cat([image_pair, text_pair], dim=-1)).squeeze(-1)


class LateInteractionReranker(nn.Module):
    def __init__(self, dim: int, hidden: int, token_count: int):
        super().__init__()
        self.hidden = hidden
        self.token_count = token_count
        self.image_tokens = nn.Linear(dim, hidden * token_count)
        self.text_tokens = nn.Linear(dim, hidden * token_count)
        self.calibrator = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch = image_emb.shape[0]
        image = self.image_tokens(image_emb).view(batch, self.token_count, self.hidden)
        text = self.text_tokens(text_emb).view(batch, self.token_count, self.hidden)
        image = F.normalize(image, dim=-1)
        text = F.normalize(text, dim=-1)
        similarity = image @ text.transpose(1, 2)
        image_max = similarity.max(dim=2).values
        text_max = similarity.max(dim=1).values
        features = torch.stack(
            [
                image_max.mean(1),
                text_max.mean(1),
                similarity.mean(dim=(1, 2)),
                similarity.amax(dim=(1, 2)),
                similarity.std(dim=(1, 2), unbiased=False),
            ],
            dim=-1,
        )
        return self.calibrator(features).squeeze(-1)


class TransformerCrossEncoderInteraction(nn.Module):
    def __init__(self, dim: int, hidden: int, token_count: int, heads: int, dropout: float):
        super().__init__()
        self.hidden = hidden
        self.token_count = token_count
        heads = heads if hidden % heads == 0 else 1
        self.image_tokens = nn.Linear(dim, hidden * token_count)
        self.text_tokens = nn.Linear(dim, hidden * token_count)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.modality = nn.Parameter(torch.zeros(1, token_count * 2 + 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.modality, std=0.02)

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch = image_emb.shape[0]
        image = self.image_tokens(image_emb).view(batch, self.token_count, self.hidden)
        text = self.text_tokens(text_emb).view(batch, self.token_count, self.hidden)
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, image, text], dim=1) + self.modality
        return self.out(self.encoder(tokens)[:, 0]).squeeze(-1)


def build_reranker(config: RerankerConfig, scale_init: float = 0.1) -> nn.Module:
    name = config.name.lower()
    hidden = int(config.hidden_dim)
    if name == "adr":
        model = ResidualCLIPFusionScorer(
            CLIPAdapterConfig(
                method="fluidmix",
                clip_dim=config.clip_dim,
                hidden_dim=hidden,
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
        model.residual_scale.data.fill_(float(scale_init))
        return model
    if name == "symmetric_mlp":
        interaction = SymmetricMLPInteraction(
            config.clip_dim,
            int(config.mlp_width or hidden),
            config.dropout,
        )
    elif name == "low_rank_bilinear":
        interaction = LowRankBilinearInteraction(config.clip_dim, hidden, config.dropout)
    elif name == "gated_multimodal":
        interaction = GatedMultimodalInteraction(config.clip_dim, hidden, config.dropout)
    elif name == "bidirectional_cross_attention":
        interaction = BidirectionalCrossAttentionInteraction(
            config.clip_dim,
            hidden,
            config.token_count,
            config.num_heads,
            config.dropout,
        )
    elif name == "late_interaction":
        interaction = LateInteractionReranker(
            config.clip_dim,
            hidden,
            config.token_count,
        )
    elif name == "transformer_cross_encoder":
        interaction = TransformerCrossEncoderInteraction(
            config.clip_dim,
            hidden,
            config.token_count,
            config.num_heads,
            config.dropout,
        )
    else:
        raise ValueError(f"Unknown reranker: {config.name}")
    return AnchoredPairReranker(interaction, scale_init=scale_init)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def closest_parameter_matched_mlp(
    target_parameters: int,
    clip_dim: int,
    dropout: float = 0.1,
    min_width: int = 16,
    max_width: int = 4096,
) -> tuple[nn.Module, int, float]:
    linear_input = int(clip_dim) * 4 + 1
    estimate = int(
        round(
            (
                -(linear_input + 3)
                + math.sqrt((linear_input + 3) ** 2 + 4 * max(int(target_parameters) - 2, 0))
            )
            / 2
        )
    )
    candidates = range(
        max(int(min_width), estimate - 8),
        min(int(max_width), estimate + 8) + 1,
    )
    best: tuple[int, int] | None = None
    for width in candidates:
        model = build_reranker(
            RerankerConfig(
                name="symmetric_mlp",
                clip_dim=clip_dim,
                hidden_dim=width,
                mlp_width=width,
                dropout=dropout,
            )
        )
        count = trainable_parameter_count(model)
        error = abs(count - int(target_parameters))
        if best is None or error < best[1]:
            best = (width, error)
    if best is None:
        raise RuntimeError("could not construct parameter-matched MLP")
    width = best[0]
    model = build_reranker(
        RerankerConfig(
            name="symmetric_mlp",
            clip_dim=clip_dim,
            hidden_dim=width,
            mlp_width=width,
            dropout=dropout,
        )
    )
    count = trainable_parameter_count(model)
    relative_error = abs(count - target_parameters) / max(int(target_parameters), 1)
    return model, width, float(relative_error)
