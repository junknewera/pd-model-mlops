import numpy as np
import pandas as pd
import pytest

from src.data.validation import TARGET_RAW
from src.features.build_features import BILL_COLS, PAY_AMT_COLS, PAY_COLS


@pytest.fixture
def clients():
    n = 200
    rng = np.random.default_rng(0)
    data = {
        "LIMIT_BAL": rng.integers(1, 50, n) * 10_000,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),
        "MARRIAGE": rng.integers(0, 4, n),
        "AGE": rng.integers(21, 75, n),
    }
    data.update({c: rng.integers(-2, 9, n) for c in PAY_COLS})
    data.update({c: rng.integers(-1_000, 100_000, n) for c in BILL_COLS})
    data.update({c: rng.integers(0, 20_000, n) for c in PAY_AMT_COLS})
    return pd.DataFrame(data)


@pytest.fixture
def raw_clients(clients):
    df = clients.rename(columns={"PAY_1": "PAY_0"})
    df.insert(0, "ID", range(1, len(df) + 1))
    df[TARGET_RAW] = (np.random.default_rng(1).random(len(df)) < 0.22).astype(int)
    return df
