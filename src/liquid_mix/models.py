from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    name: str = "fluidmix"
    hidden_dim: int = 128
    field_grid: int = 4
    steps: int = 4
    dt: float = 0.2
    diffusion: float = 0.08
    reaction: float = 0.5
    num_heads: int = 2
    use_advection: bool = True
    use_diffusion: bool = True
    use_reaction: bool = True


class ConcatFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_vec, text_vec, image_field, text_field):
        return self.net(torch.cat([image_vec, text_vec], dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.out = nn.LayerNorm(hidden_dim)

    def forward(self, image_vec, text_vec, image_field, text_field):
        gate = self.gate(torch.cat([image_vec, text_vec], dim=-1))
        return self.out(gate * image_vec + (1.0 - gate) * text_vec)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        heads = num_heads if hidden_dim % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_vec, text_vec, image_field, text_field):
        tokens = torch.stack([image_vec, text_vec], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = attended.mean(dim=1)
        return fused + self.ffn(fused)


class LiquidFusion(nn.Module):
    """Liquid time-constant fusion baseline over global modality features."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        dim = config.hidden_dim
        self.steps = config.steps
        self.dt = config.dt
        self.input_proj = nn.Linear(dim * 2, dim)
        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.tau = nn.Parameter(torch.ones(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, image_vec, text_vec, image_field, text_field):
        drive = self.input_proj(torch.cat([image_vec, text_vec], dim=-1))
        state = 0.5 * (image_vec + text_vec)
        tau = F.softplus(self.tau).view(1, -1) + 1e-3
        for _ in range(self.steps):
            target = torch.tanh(self.state_proj(state) + drive)
            dstate = (-state + target) / tau
            state = state + self.dt * dstate
        return self.norm(state)


class FluidMixFusion(nn.Module):
    """Advection-diffusion-reaction fusion over modality concentration fields."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        dim = config.hidden_dim
        self.field_grid = int(config.field_grid)
        self.steps = config.steps
        self.dt = config.dt
        self.diffusion = config.diffusion
        self.reaction = config.reaction
        self.use_advection = bool(config.use_advection)
        self.use_diffusion = bool(config.use_diffusion)
        self.use_reaction = bool(config.use_reaction)
        self.modality_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.velocity = nn.Linear(dim * 2, dim * 2)
        self.reaction_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("laplacian_kernel", kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        return F.conv2d(field, self.laplacian_kernel, padding=1, groups=field.shape[1])

    @staticmethod
    def _gradients(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_x = 0.5 * (torch.roll(field, shifts=-1, dims=3) - torch.roll(field, shifts=1, dims=3))
        grad_y = 0.5 * (torch.roll(field, shifts=-1, dims=2) - torch.roll(field, shifts=1, dims=2))
        return grad_x, grad_y

    def forward(self, image_vec, text_vec, image_field, text_field):
        gate = self.modality_gate(torch.cat([image_vec, text_vec], dim=-1)).view(
            image_vec.shape[0], image_vec.shape[1], 1, 1
        )
        concentration = gate * image_field + (1.0 - gate) * text_field
        velocity = torch.tanh(self.velocity(torch.cat([image_vec, text_vec], dim=-1)))
        vx, vy = velocity.chunk(2, dim=-1)
        vx = vx.view(vx.shape[0], vx.shape[1], 1, 1)
        vy = vy.view(vy.shape[0], vy.shape[1], 1, 1)

        for _ in range(self.steps):
            if self.use_advection:
                grad_x, grad_y = self._gradients(concentration)
                advection = -(vx * grad_x + vy * grad_y)
            else:
                advection = torch.zeros_like(concentration)
            diffusion = (
                self.diffusion * self._laplacian(concentration)
                if self.use_diffusion
                else torch.zeros_like(concentration)
            )
            reaction = (
                self.reaction * torch.tanh(self.reaction_conv(concentration))
                if self.use_reaction
                else torch.zeros_like(concentration)
            )
            concentration = concentration + self.dt * (advection + diffusion + reaction)

        vector = concentration.mean(dim=(2, 3))
        return self.norm(vector)


def build_fusion(config: FusionConfig) -> nn.Module:
    name = config.name.lower()
    if name == "concat":
        return ConcatFusion(config.hidden_dim)
    if name == "gated":
        return GatedFusion(config.hidden_dim)
    if name == "cross_attention":
        return CrossAttentionFusion(config.hidden_dim, config.num_heads)
    if name == "liquid":
        return LiquidFusion(config)
    if name == "fluidmix":
        return FluidMixFusion(config)
    raise ValueError(f"Unknown fusion method: {config.name}")
