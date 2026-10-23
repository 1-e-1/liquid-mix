import torch

from liquid_mix.backbone_fluid import BackboneFluidConfig, BackboneFluidMixScorer


def test_backbone_fluid_scorer_accepts_precomputed_features():
    config = BackboneFluidConfig(image_dim=32, text_dim=24, hidden_dim=16, field_grid=2, steps=1)
    model = BackboneFluidMixScorer(config)

    image_vec = torch.randn(3, 32)
    text_vec = torch.randn(3, 24)
    image_field = torch.randn(3, 32, 2, 2)
    text_field = torch.randn(3, 24, 2, 2)

    pair_scores = model(image_vec, text_vec, image_field, text_field)
    score_matrix = model.score_matrix(image_vec, text_vec, image_field, text_field, chunk_size=4)

    assert pair_scores.shape == (3,)
    assert score_matrix.shape == (3, 3)
    assert torch.isfinite(score_matrix).all()
