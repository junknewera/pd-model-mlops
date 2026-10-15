import numpy as np

from src.monitoring.drift import psi, shift_data, verdict


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.1


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(loc=1, size=5000)) > 0.25


def test_psi_discrete_feature():
    ref = np.array([-1, 0, 0, 0, 1, 2] * 100)
    assert psi(ref, ref) == 0
    assert psi(ref, ref + 2) > 0.25


def test_verdict():
    assert verdict(0.05) == "stable"
    assert verdict(0.2) == "moderate"
    assert verdict(0.3) == "significant"


def test_shift_data_changes_key_features(clients):
    shifted = shift_data(clients)
    assert (shifted["LIMIT_BAL"] < clients["LIMIT_BAL"]).all()
    assert shifted["PAY_1"].mean() > clients["PAY_1"].mean()
