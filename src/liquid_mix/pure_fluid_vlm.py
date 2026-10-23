from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .models import FusionConfig, build_fusion


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class SimpleCaptionTokenizer:
    def __init__(self, token_to_id: dict[str, int], max_len: int):
        if "[PAD]" not in token_to_id or token_to_id["[PAD]"] != 0:
            raise ValueError("token_to_id must map [PAD] to 0")
        if "[UNK]" not in token_to_id or token_to_id["[UNK]"] != 1:
            raise ValueError("token_to_id must map [UNK] to 1")
        self.token_to_id = dict(token_to_id)
        self.max_len = int(max_len)

    @classmethod
    def fit(
        cls,
        captions: Iterable[str],
        max_vocab: int,
        max_len: int,
        min_freq: int = 1,
    ) -> "SimpleCaptionTokenizer":
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(cls.tokenize(caption))
        words = [
            token
            for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_freq
        ]
        token_to_id = {"[PAD]": 0, "[UNK]": 1}
        for token in words[: max(0, int(max_vocab) - len(token_to_id))]:
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id, max_len=max_len)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> torch.Tensor:
        ids = [self.token_to_id.get(token, 1) for token in self.tokenize(text)]
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.stack([self.encode(text) for text in texts], dim=0)

    def state_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "max_len": self.max_len}

    @classmethod
    def from_state_dict(cls, state: dict) -> "SimpleCaptionTokenizer":
        return cls(token_to_id=dict(state["token_to_id"]), max_len=int(state["max_len"]))


@dataclass
class PureFluidConfig:
    vocab_size: int
    max_len: int = 32
    image_size: int = 96
    hidden_dim: int = 128
    field_grid: int = 4
    text_layers: int = 2
    text_heads: int = 4
    dropout: float = 0.1
    steps: int = 4
    dt: float = 0.2
    diffusion: float = 0.03
    reaction: float = 0.5
    num_heads: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ImagePatchEncoder(nn.Module):
    def __init__(self, config: PureFluidConfig):
        super().__init__()
        dim = config.hidden_dim
        c1 = max(16, dim // 4)
        c2 = max(32, dim // 2)
        self.field_grid = config.field_grid
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_count(c1), c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c2), c2),
            nn.GELU(),
            nn.Conv2d(c2, dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_count(dim), dim),
            nn.GELU(),
        )
        self.field_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.vec_norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(images)
        field = F.adaptive_avg_pool2d(features, (self.field_grid, self.field_grid))
        field = torch.tanh(self.field_proj(field))
        vector = self.vec_norm(field.mean(dim=(2, 3)))
        return vector, field


class TextConcentrationEncoder(nn.Module):
    def __init__(self, config: PureFluidConfig):
        super().__init__()
        dim = config.hidden_dim
        heads = config.text_heads if dim % config.text_heads == 0 else 1
        self.max_len = config.max_len
        self.field_grid = config.field_grid
        self.token_embed = nn.Embedding(config.vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.text_layers)
        self.field_proj = nn.Linear(dim, dim)
        self.vec_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] != self.max_len:
            raise ValueError(f"Expected token length {self.max_len}, got {tokens.shape[1]}")
        pad_mask = tokens.eq(0)
        x = self.token_embed(tokens) + self.pos_embed[:, : tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
        lengths = keep.sum(dim=1).clamp_min(1.0)
        vector = self.vec_norm((x * keep).sum(dim=1) / lengths)

        cells = self.field_grid * self.field_grid
        field_tokens = F.adaptive_avg_pool1d((x * keep).transpose(1, 2), cells).transpose(1, 2)
        field_tokens = torch.tanh(self.field_proj(field_tokens))
        field = field_tokens.transpose(1, 2).reshape(tokens.shape[0], -1, self.field_grid, self.field_grid)
        return vector, field


class PureFluidVLM(nn.Module):
    """End-to-end image-text matcher using only project-local encoders."""

    def __init__(self, config: PureFluidConfig):
        super().__init__()
        self.config = config
        self.image_encoder = ImagePatchEncoder(config)
        self.text_encoder = TextConcentrationEncoder(config)
        self.fusion = build_fusion(
            FusionConfig(
                name="fluidmix",
                hidden_dim=config.hidden_dim,
                field_grid=config.field_grid,
                steps=config.steps,
                dt=config.dt,
                diffusion=config.diffusion,
                reaction=config.reaction,
                num_heads=config.num_heads,
            )
        )
        dim = config.hidden_dim
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

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image_encoder(images)

    def encode_text(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_encoder(tokens)

    def score_pairs_from_encoded(
        self,
        image_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_field: torch.Tensor,
        text_field: torch.Tensor,
    ) -> torch.Tensor:
        image_z = F.normalize(self.image_retrieval(image_vec), dim=-1)
        text_z = F.normalize(self.text_retrieval(text_vec), dim=-1)
        retrieval_logit = (image_z * text_z).sum(dim=-1) * self.logit_scale.exp().clamp(max=100.0)
        fused = self.fusion(image_vec, text_vec, image_field, text_field)
        fluid_logit = self.fluid_head(fused).squeeze(-1)
        return retrieval_logit + self.fluid_scale * fluid_logit

    def score_pairs(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        image_vec, image_field = self.encode_image(images)
        text_vec, text_field = self.encode_text(tokens)
        return self.score_pairs_from_encoded(image_vec, text_vec, image_field, text_field)

    def score_matrix(self, images: torch.Tensor, tokens: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        image_vec, image_field = self.encode_image(images)
        text_vec, text_field = self.encode_text(tokens)
        num_images = image_vec.shape[0]
        num_texts = text_vec.shape[0]
        total = num_images * num_texts
        scores = []
        flat = torch.arange(total, device=images.device)
        for start in range(0, total, int(chunk_size)):
            ids = flat[start : start + int(chunk_size)]
            image_idx = torch.div(ids, num_texts, rounding_mode="floor")
            text_idx = ids.remainder(num_texts)
            scores.append(
                self.score_pairs_from_encoded(
                    image_vec[image_idx],
                    text_vec[text_idx],
                    image_field[image_idx],
                    text_field[text_idx],
                )
            )
        return torch.cat(scores, dim=0).view(num_images, num_texts)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(images, tokens)
