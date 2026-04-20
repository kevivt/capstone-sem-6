import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


MISSING_TOKENS = ["?", "\t?", " ?", "\t?\t", "", " "]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CKD preprocessing and modeling pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to local CKD dataset file (.csv/.data/.arff)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="class",
        help="Target column name (default: class)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_from_arff(file_path: Path) -> pd.DataFrame:
    data, _ = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )
    return df


def try_load_local_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".arff":
        return load_from_arff(file_path)
    if suffix in {".data", ".txt"}:
        return pd.read_csv(file_path, header=None)
    raise ValueError(f"Unsupported file type: {suffix}")


def fetch_ckd_from_ucirepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ckd = fetch_ucirepo(id=336)
    X = ckd.data.features.copy()
    y = ckd.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        target_col = y.columns[0]
        X[target_col] = y[target_col]
    else:
        X["class"] = y
    return X


def resolve_dataset(path_arg: Optional[str]) -> pd.DataFrame:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return try_load_local_file(path)

    search_roots = [
        Path("medical datasets/raw/chronic+kidney+disease"),
        Path("Sem_6_Capstone/medical datasets/raw/chronic+kidney+disease"),
        Path("."),
    ]

    for root in search_roots:
        if root.exists() and root.is_dir():
            candidates = list(root.rglob("*.csv")) + list(root.rglob("*.arff"))
            if candidates:
                return try_load_local_file(candidates[0])

    print("No local CKD data file found. Falling back to ucimlrepo(id=336).")
    return fetch_ckd_from_ucirepo()


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace(MISSING_TOKENS, np.nan)
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    return cleaned


def infer_numeric_columns(X: pd.DataFrame, threshold: float = 0.75) -> Tuple[list, list, pd.DataFrame]:
    X2 = X.copy()
    numeric_cols = []
    categorical_cols = []

    for col in X2.columns:
        numeric_version = pd.to_numeric(X2[col], errors="coerce")
        non_null = X2[col].notna().sum()
        numeric_non_null = numeric_version.notna().sum()

        if non_null == 0:
            categorical_cols.append(col)
            continue

        if numeric_non_null / non_null >= threshold:
            X2[col] = numeric_version
            numeric_cols.append(col)
        else:
            X2[col] = X2[col].astype(str).str.strip()
            X2[col] = X2[col].replace("nan", np.nan)
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, X2


def encode_target(y: pd.Series) -> pd.Series:
    y_clean = y.astype(str).str.strip().str.lower()
    mapping = {
        "ckd": 1,
        "notckd": 0,
        "not ckd": 0,
        "1": 1,
        "0": 0,
    }
    y_encoded = y_clean.map(mapping)
    if y_encoded.isna().any():
        unknown = sorted(y_clean[y_encoded.isna()].dropna().unique())
        raise ValueError(f"Unknown target labels found: {unknown}")
    return y_encoded.astype(int)


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )


def train_and_evaluate_models(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series) -> None:
    models = {
        "SVM_linear": SVC(C=0.241, kernel="linear", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=1, weights="distance", algorithm="kd_tree"),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=6,
            reg_alpha=60.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        print("xgboost not installed. Skipping XGBoost model.")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = None

        macro_f1 = f1_score(y_test, preds, average="macro")
        print("\n" + "=" * 60)
        print(f"Model: {name}")
        print(f"Macro F1: {macro_f1:.4f}")

        if probs is not None:
            auc = roc_auc_score(y_test, probs)
            print(f"ROC AUC: {auc:.4f}")

        print(classification_report(y_test, preds, digits=4))


def main() -> None:
    args = parse_args()

    df = resolve_dataset(args.input)
    df = normalize_missing_values(df)

    if args.target not in df.columns:
        if "class" in df.columns:
            args.target = "class"
        else:
            raise ValueError(
                f"Target column '{args.target}' not found. Available columns: {list(df.columns)}"
            )

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=[args.target])
    y = encode_target(df[args.target])

    num_cols, cat_cols, X = infer_numeric_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    smote = SMOTE(random_state=args.random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_prepared, y_train)

    print("Preprocessing complete.")
    print(f"Train shape before SMOTE: {X_train_prepared.shape}")
    print(f"Train shape after SMOTE: {X_train_balanced.shape}")
    print(f"Test shape: {X_test_prepared.shape}")

    train_and_evaluate_models(X_train_balanced, y_train_balanced, X_test_prepared, y_test)


if __name__ == "__main__":
    main()
