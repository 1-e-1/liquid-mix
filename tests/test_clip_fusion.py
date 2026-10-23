import torch

from liquid_mix.clip_fusion import (
    CLIPAdapterConfig,
    CLIPFusionAdapter,
    ResidualCLIPFusionScorer,
    best_threshold,
    binary_scores_at_threshold,
    clip_cosine_scores,
    make_hard_negative_pair_indices,
    make_pair_indices,
    recall_at_k,
    symmetric_contrastive_loss,
)


def test_clip_fusion_adapter_outputs_one_logit_per_pair():
    image_emb = torch.randn(4, 32)
    text_emb = torch.randn(4, 32)

    for method in ["concat", "gated", "cross_attention", "liquid", "fluidmix"]:
        model = CLIPFusionAdapter(
            CLIPAdapterConfig(
                method=method,
                clip_dim=32,
                hidden_dim=16,
                field_grid=2,
                steps=2,
                num_heads=2,
            )
        )
        logits = model(image_emb, text_emb)
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()


def test_pair_indices_include_positive_and_shifted_negative():
    image_idx, text_idx, labels = make_pair_indices(num_samples=4, negative_offset=1)

    assert image_idx.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert text_idx.tolist() == [0, 1, 1, 2, 2, 3, 3, 0]
    assert labels.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_hard_negative_pair_indices_use_top_non_diagonal_scores():
    scores = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.3, 1.0, 0.8, 0.2],
            [0.7, 0.4, 1.0, 0.6],
            [0.2, 0.5, 0.9, 1.0],
        ]
    )

    image_idx, text_idx, labels = make_hard_negative_pair_indices(scores, negatives_per_positive=2)

    assert image_idx.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert text_idx.tolist() == [0, 1, 3, 1, 2, 0, 2, 0, 3, 3, 2, 1]
    assert labels.tolist() == [1.0, 0.0, 0.0] * 4


def test_threshold_metrics_and_recall():
    scores = torch.tensor([0.9, 0.1, 0.8, 0.2])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    threshold = best_threshold(scores, labels)
    metrics = binary_scores_at_threshold(scores, labels, threshold)

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0

    matrix = torch.eye(3)
    assert recall_at_k(matrix, ks=(1, 2)) == {"r@1": 1.0, "r@2": 1.0}


def test_clip_cosine_scores_prefers_matching_vectors():
    eye = torch.eye(3)
    scores = clip_cosine_scores(eye, eye)

    assert scores.shape == (3, 3)
    assert recall_at_k(scores, ks=(1,))["r@1"] == 1.0


def test_residual_scorer_preserves_clip_cosine_at_initialization():
    image_emb = torch.eye(4)
    text_emb = torch.eye(4)
    scorer = ResidualCLIPFusionScorer(
        CLIPAdapterConfig(method="fluidmix", clip_dim=4, hidden_dim=8, field_grid=2, steps=1)
    )

    scores = scorer(image_emb, text_emb)

    assert torch.allclose(scores, torch.ones(4), atol=1e-6)


def test_symmetric_contrastive_loss_rewards_diagonal_scores():
    good = torch.eye(4) * 5.0
    bad = torch.zeros(4, 4)

    assert symmetric_contrastive_loss(good) < symmetric_contrastive_loss(bad)
