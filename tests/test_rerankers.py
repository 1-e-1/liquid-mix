import torch

from liquid_mix.rerankers import (
    RerankerConfig,
    build_reranker,
    closest_parameter_matched_mlp,
    trainable_parameter_count,
)


def test_all_second_stage_rerankers_return_pair_scores():
    image = torch.randn(3, 32)
    text = torch.randn(3, 32)
    names = (
        "symmetric_mlp",
        "low_rank_bilinear",
        "gated_multimodal",
        "bidirectional_cross_attention",
        "late_interaction",
        "transformer_cross_encoder",
        "adr",
    )
    for name in names:
        model = build_reranker(
            RerankerConfig(
                name=name,
                clip_dim=32,
                hidden_dim=16,
                token_count=2,
                num_heads=2,
                field_grid=2,
                steps=1,
            )
        )
        scores = model(image, text)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()


def test_adr_operator_flags_change_the_dynamics():
    image = torch.randn(4, 32)
    text = torch.randn(4, 32)
    full = build_reranker(
        RerankerConfig(
            name="adr",
            clip_dim=32,
            hidden_dim=16,
            field_grid=2,
            steps=2,
        )
    )
    without_reaction = build_reranker(
        RerankerConfig(
            name="adr",
            clip_dim=32,
            hidden_dim=16,
            field_grid=2,
            steps=2,
            use_reaction=False,
        )
    )
    without_reaction.load_state_dict(full.state_dict())
    assert not torch.allclose(full(image, text), without_reaction(image, text))


def test_parameter_matched_mlp_is_within_one_percent():
    adr = build_reranker(
        RerankerConfig(
            name="adr",
            clip_dim=64,
            hidden_dim=32,
            field_grid=2,
            steps=2,
        )
    )
    target = trainable_parameter_count(adr)
    matched, _, relative_error = closest_parameter_matched_mlp(target, clip_dim=64)
    assert relative_error < 0.01
    assert abs(trainable_parameter_count(matched) - target) / target < 0.01
