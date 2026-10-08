import numpy as np
import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BIN"]
NEW_NUMERIC = [
    "PAY_DELAY_MAX",
    "PAY_DELAY_COUNT",
    "BILL_AMT_MEAN",
    "PAY_AMT_MEAN",
    "UTILIZATION",
    "PAY_RATIO",
]
NUMERIC = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS + NEW_NUMERIC


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # по ответу автора датасета EDUCATION 0, 4, 5, 6 - это others
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["AGE_BIN"] = pd.cut(
        df["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)

    pay = df[PAY_COLS]
    df["PAY_DELAY_MAX"] = pay.max(axis=1)
    df["PAY_DELAY_COUNT"] = (pay > 0).sum(axis=1)
    df["BILL_AMT_MEAN"] = df[BILL_COLS].mean(axis=1)
    df["PAY_AMT_MEAN"] = df[PAY_AMT_COLS].mean(axis=1)
    df["UTILIZATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    # платёж в сентябре гасит счёт за август (BILL_AMT2);
    # если счёта не было, доля не определена, NaN потом заполнит Imputer
    bill = df["BILL_AMT2"].where(df["BILL_AMT2"] > 0, np.nan)
    df["PAY_RATIO"] = (df["PAY_AMT1"] / bill).clip(upper=2)
    return df
