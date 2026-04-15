from src.evaluation import compute_metrics


def test_compute_metrics_returns_expected_keys():
    metrics = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
