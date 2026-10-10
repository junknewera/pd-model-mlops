import json
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.data.make_dataset import PROCESSED_DIR, TARGET
from src.models.metrics import compute_metrics, plot_roc
from src.models.pipeline import build_pipeline

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")
FIG_DIR = Path("reports/figures")


def make_search(name, cfg, params):
    pipe = build_pipeline(name, params["seed"])
    common = dict(
        scoring=params["train"]["scoring"], cv=params["train"]["cv"], n_jobs=-1
    )
    if cfg["search"] == "grid":
        return GridSearchCV(pipe, cfg["params"], **common)
    return RandomizedSearchCV(
        pipe,
        cfg["params"],
        n_iter=cfg["n_iter"],
        random_state=params["seed"],
        **common,
    )


def log_candidates(name, search):
    res = search.cv_results_
    for i, cand in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}-{i}", nested=True):
            mlflow.log_params(cand)
            mlflow.log_metric("cv_roc_auc", res["mean_test_score"][i])
            mlflow.log_metric("cv_roc_auc_std", res["std_test_score"][i])


def main():
    params = yaml.safe_load(open("params.yaml"))
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("pd-model")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, cfg in params["train"]["models"].items():
        search = make_search(name, cfg, params)
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            proba = search.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, proba)

            mlflow.log_params({"model": name, "search": cfg["search"]})
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

            with tempfile.TemporaryDirectory() as tmp:
                roc_path = Path(tmp) / f"roc_{name}.png"
                plot_roc(y_test, {name: proba}, roc_path)
                mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(
                search.best_estimator_,
                name="model",
                input_example=X_train.head(3),
                code_paths=["src"],
                # skops не сохраняет эти типы без явного разрешения, модель своя
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    "src.features.build_features.add_features",
                ],
            )
            log_candidates(name, search)

        print(f"{name}: cv_auc={search.best_score_:.4f} test={metrics}")
        results[name] = {
            "cv_auc": search.best_score_,
            "model": search.best_estimator_,
            "proba": proba,
            "metrics": metrics,
        }

    best = max(results, key=lambda n: results[n]["cv_auc"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best]["model"], MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "best_model": best,
                "cv_roc_auc": results[best]["cv_auc"],
                **results[best]["metrics"],
            },
            indent=2,
        )
    )
    probas = {n: r["proba"] for n, r in results.items()}
    plot_roc(y_test, probas, FIG_DIR / "roc_curve.png")
    print(f"best model: {best}, saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
