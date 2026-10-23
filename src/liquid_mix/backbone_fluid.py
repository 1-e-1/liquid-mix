from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .models import FusionConfig, build_fusion


@dataclass
class BackboneFluidConfig:
    image_dim: int
    text_dim: int
    hidden_dim: int = 128
    field_grid: int = 4
    steps: int = 4
    dt: float = 0.2
    diffusion: float = 0.03
    reaction: float = 0.5
    num_heads: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class BackboneFluidMixScorer(nn.Module):
    """FluidMix scorer for frozen non-CLIP image/text backbone features."""

    def __init__(self, config: BackboneFluidConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.image_vec_proj = nn.Sequential(nn.Linear(config.image_dim, dim), nn.LayerNorm(dim))
        self.text_vec_proj = nn.Sequential(nn.Linear(config.text_dim, dim), nn.LayerNorm(dim))
        self.image_field_proj = nn.Conv2d(config.image_dim, dim, kernel_size=1)
        self.text_field_proj = nn.Conv2d(config.text_dim, dim, kernel_size=1)
        self.fusion = build_fusion(
            FusionConfig(
                name="fluidmix",
                hidden_dim=dim,
                field_grid=config.field_grid,
                steps=config.steps,
                dt=config.dt,
                diffusion=config.diffusion,
                reaction=config.reaction,
                num_heads=config.num_heads,
            )
        )
        self.image_retrieval = nn.Linear(dim, dim)
        self.text_retrieval = nn.Linear(dim, dim)
        self.fluid_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.fluid_scale = nn.Parameter(torch.tensor(1.0))

    def project(
        self,
        image_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_field: torch.Tensor,
        text_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_vec = self.image_vec_proj(image_vec)
        text_vec = self.text_vec_proj(text_vec)
        image_field = torch.tanh(self.image_field_proj(image_field))
        text_field = torch.tanh(self.text_field_proj(text_field))
        return image_vec, text_vec, image_field, text_field

    def score_pairs(
        self,
        image_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_field: torch.Tensor,
        text_field: torch.Tensor,
    ) -> torch.Tensor:
        image_vec, text_vec, image_field, text_field = self.project(image_vec, text_vec, image_field, text_field)
        image_z = F.normalize(self.image_retrieval(image_vec), dim=-1)
        text_z = F.normalize(self.text_retrieval(text_vec), dim=-1)
        retrieval_logit = (image_z * text_z).sum(dim=-1) * self.logit_scale.exp().clamp(max=100.0)
        fused = self.fusion(image_vec, text_vec, image_field, text_field)
        fluid_logit = self.fluid_head(fused).squeeze(-1)
        return retrieval_logit + self.fluid_scale * fluid_logit

    def score_matrix(
        self,
        image_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_field: torch.Tensor,
        text_field: torch.Tensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        num_images = image_vec.shape[0]
        num_texts = text_vec.shape[0]
        total = num_images * num_texts
        flat = torch.arange(total, device=image_vec.device)
        scores = []
        for start in range(0, total, int(chunk_size)):
            ids = flat[start : start + int(chunk_size)]
            image_idx = torch.div(ids, num_texts, rounding_mode="floor")
            text_idx = ids.remainder(num_texts)
            scores.append(
                self.score_pairs(
                    image_vec[image_idx],
                    text_vec[text_idx],
                    image_field[image_idx],
                    text_field[text_idx],
                )
            )
        return torch.cat(scores, dim=0).view(num_images, num_texts)

    def forward(
        self,
        image_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_field: torch.Tensor,
        text_field: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_pairs(image_vec, text_vec, image_field, text_field)
