import torch

from liquid_mix.statistics import (
    exact_paired_mcnemar,
    holm_adjust,
    paired_bootstrap_metric_delta,
    paired_retrieval_indicator,
)


def test_paired_retrieval_indicator_marks_correct_top1():
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.1, 0.0],
            [0.0, 0.1, 0.8],
        ]
    )

    indicator = paired_retrieval_indicator(scores, k=1)

    assert indicator.tolist() == [1.0, 0.0, 1.0]


def test_paired_bootstrap_metric_delta_reports_positive_shift():
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0])
    improved = torch.tensor([1.0, 1.0, 1.0, 1.0])

    stats = paired_bootstrap_metric_delta(baseline, improved, n_boot=200, seed=7)

    assert stats["baseline_mean"] == 0.0
    assert stats["improved_mean"] == 1.0
    assert stats["delta"] == 1.0
    assert stats["prob_delta_gt_0"] == 1.0
    assert stats["ci_low"] == 1.0


def test_exact_mcnemar_and_holm_adjustment():
    baseline = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
    improved = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.float32)
    result = exact_paired_mcnemar(baseline, improved)
    assert result["improved_only"] == 3
    assert result["baseline_only"] == 0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert len(adjusted) == 3
    assert all(0.0 <= value <= 1.0 for value in adjusted)
