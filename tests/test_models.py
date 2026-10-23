import torch

from liquid_mix.models import FusionConfig, build_fusion


def test_each_fusion_module_returns_fixed_width_features():
    batch = 3
    dim = 16
    grid = 4
    image_vec = torch.randn(batch, dim)
    text_vec = torch.randn(batch, dim)
    image_field = torch.randn(batch, dim, grid, grid)
    text_field = torch.randn(batch, dim, grid, grid)

    for name in ["concat", "gated", "cross_attention", "liquid", "fluidmix"]:
        fusion = build_fusion(FusionConfig(name=name, hidden_dim=dim, field_grid=grid, steps=3))
        out = fusion(image_vec, text_vec, image_field, text_field)
        assert out.shape == (batch, dim)
        assert torch.isfinite(out).all()
