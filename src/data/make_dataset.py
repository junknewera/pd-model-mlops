import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.download import RAW_PATH
from src.data.validation import TARGET_RAW, failed_expectations, validate

TARGET = "default"
PROCESSED_DIR = Path("data/processed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # PAY_0 в исходнике - это сентябрь, остальные колонки нумеруются с 2
    return (
        df.drop(columns="ID")
        .rename(columns={"PAY_0": "PAY_1", TARGET_RAW: TARGET})
        .drop_duplicates()
        .reset_index(drop=True)
    )


def main():
    logging.disable(logging.WARNING)
    params = yaml.safe_load(open("params.yaml"))
    raw = pd.read_csv(RAW_PATH)

    result = validate(raw)
    if not result.success:
        raise SystemExit(f"raw data failed validation: {failed_expectations(result)}")

    df = clean(raw)
    train, test = train_test_split(
        df,
        test_size=params["split"]["test_size"],
        stratify=df[TARGET],
        random_state=params["seed"],
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(
        f"raw {raw.shape} -> clean {df.shape}; train {train.shape}, test {test.shape}"
    )


if __name__ == "__main__":
    main()
