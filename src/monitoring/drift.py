import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src.data.make_dataset import PROCESSED_DIR, TARGET

KEY_FEATURES = ["LIMIT_BAL", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1"]


def psi(expected, actual, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def verdict(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def shift_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LIMIT_BAL"] = (df["LIMIT_BAL"] * 0.6).round()
    df["PAY_1"] = (df["PAY_1"] + 1).clip(upper=8)
    df["AGE"] = (df["AGE"] - 5).clip(lower=21)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = joblib.load("models/model.joblib")
    X_train = pd.read_csv(PROCESSED_DIR / "train.csv").drop(columns=TARGET)
    test = pd.read_csv(PROCESSED_DIR / "test.csv").drop(columns=TARGET)
    batch = test.sample(args.n, random_state=args.seed)
    if args.shift:
        batch = shift_data(batch)

    with requests.Session() as s:
        scores = [
            s.post(f"{args.url}/predict", json=row, timeout=10).json()["probability"]
            for row in batch.to_dict(orient="records")
        ]
    ref_scores = model.predict_proba(X_train)[:, 1]

    report = {"score": psi(ref_scores, scores)}
    report.update({f: psi(X_train[f], batch[f]) for f in KEY_FEATURES})

    print(f"sent {len(scores)} requests to {args.url}, shift={args.shift}")
    print(f"mean score: train {ref_scores.mean():.3f}, new {np.mean(scores):.3f}")
    for name, value in report.items():
        print(f"PSI {name:<10} {value:.4f}  {verdict(value)}")

    out = Path("reports") / ("drift_shift.json" if args.shift else "drift.json")
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
