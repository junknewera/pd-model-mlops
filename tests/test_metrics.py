import numpy as np
import pytest

from src.models.metrics import compute_metrics, plot_roc


def test_perfect_predictions():
    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == {"roc_auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_known_values():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, proba)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_threshold():
    y = np.array([0, 1, 1])
    proba = np.array([0.1, 0.3, 0.9])
    assert compute_metrics(y, proba)["recall"] == 0.5
    assert compute_metrics(y, proba, threshold=0.2)["recall"] == 1.0


def test_plot_roc_saves_file(tmp_path):
    path = tmp_path / "roc.png"
    plot_roc(np.array([0, 1, 0, 1]), {"m": np.array([0.2, 0.7, 0.4, 0.6])}, path)
    assert path.stat().st_size > 0
