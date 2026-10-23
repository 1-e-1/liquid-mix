import torch

from liquid_mix.metrics import binary_metrics, retrieval_recall_at_k


def test_binary_metrics_reports_accuracy_and_auc():
    logits = torch.tensor([-3.0, 2.0, -1.0, 4.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    metrics = binary_metrics(logits, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_retrieval_recall_at_k_uses_diagonal_as_ground_truth():
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.2],
            [0.8, 0.7, 0.1],
            [0.1, 0.2, 0.6],
        ]
    )

    recalls = retrieval_recall_at_k(scores, ks=(1, 2))

    assert recalls["r@1"] == 2 / 3
    assert recalls["r@2"] == 1.0
