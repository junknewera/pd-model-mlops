import numpy as np

from src.features.build_features import CATEGORICAL, NUMERIC, add_features


def test_adds_columns_and_keeps_input(clients):
    before = clients.copy()
    out = add_features(clients)
    assert set(NUMERIC + CATEGORICAL) <= set(out.columns)
    assert len(out) == len(clients)
    assert clients.equals(before)


def test_education_others_collapsed(clients):
    assert set(add_features(clients)["EDUCATION"]) <= {1, 2, 3, 4}


def test_age_bins(clients):
    df = clients.head(4).copy()
    df["AGE"] = [24, 25, 44, 60]
    assert add_features(df)["AGE_BIN"].tolist() == ["<25", "25-34", "35-44", "55+"]


def test_pay_delay(clients):
    df = clients.head(1).copy()
    df[["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = [-1, 2, 0, 3, -2, 0]
    out = add_features(df)
    assert out["PAY_DELAY_MAX"].iloc[0] == 3
    assert out["PAY_DELAY_COUNT"].iloc[0] == 2


def test_pay_ratio_without_bill_is_nan(clients):
    df = clients.head(2).copy()
    df["BILL_AMT2"] = [0, 1000]
    df["PAY_AMT1"] = [500, 500]
    out = add_features(df)
    assert np.isnan(out["PAY_RATIO"].iloc[0])
    assert out["PAY_RATIO"].iloc[1] == 0.5
    assert not np.isinf(add_features(clients)[NUMERIC]).any().any()
