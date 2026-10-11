from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.build_features import CATEGORICAL, NUMERIC, add_features


def make_model(name: str, seed: int = 42):
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss")
    if name == "catboost":
        return CatBoostClassifier(
            random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False
        )
    raise ValueError(f"unknown model: {name}")


def build_pipeline(model_name: str, seed: int = 42) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", preprocess),
            ("model", make_model(model_name, seed)),
        ]
    )
