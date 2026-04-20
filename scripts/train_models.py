import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas.api.types import is_numeric_dtype
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


MISSING_TOKENS = ["?", "\t?", " ?", "\t?\t", "", " "]


def stratified_sample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int | None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y

    sampled_idx = (
        y.groupby(y, group_keys=False)
        .apply(lambda s: s.sample(n=max(1, int(round(max_rows * (len(s) / len(y))))), random_state=random_state))
        .index
    )

    if len(sampled_idx) > max_rows:
        sampled_idx = pd.Index(sampled_idx).to_series().sample(n=max_rows, random_state=random_state).index

    Xs = X.loc[sampled_idx].reset_index(drop=True)
    ys = y.loc[sampled_idx].reset_index(drop=True)
    return Xs, ys


def load_ckd(ckd_path: Optional[Path]) -> pd.DataFrame:
    if ckd_path and ckd_path.exists():
        if ckd_path.suffix.lower() == ".arff":
            data, _ = arff.loadarff(str(ckd_path))
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(
                        lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
                    )
            return df
        return pd.read_csv(ckd_path)

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


def infer_numeric_categorical(X: pd.DataFrame, threshold: float = 0.75) -> Tuple[list, list, pd.DataFrame]:
    X2 = X.copy()
    num_cols = []
    cat_cols = []

    for col in X2.columns:
        numeric_candidate = pd.to_numeric(X2[col], errors="coerce")
        non_null_count = X2[col].notna().sum()
        numeric_non_null = numeric_candidate.notna().sum()

        if non_null_count == 0:
            cat_cols.append(col)
            continue

        if (numeric_non_null / non_null_count) >= threshold:
            X2[col] = numeric_candidate
            num_cols.append(col)
        else:
            X2[col] = X2[col].astype(str).str.strip().replace("nan", np.nan)
            cat_cols.append(col)

    return num_cols, cat_cols, X2


def preprocess_and_split(df, target_col, num_cols, cat_cols, is_diabetes=False):
    df = df.copy().replace(MISSING_TOKENS, np.nan)
    df.columns = [str(c).strip() for c in df.columns]

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if is_diabetes:
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        existing_zero_cols = [col for col in zero_cols if col in df.columns]
        if existing_zero_cols:
            df[existing_zero_cols] = df[existing_zero_cols].replace(0, np.nan)

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    if is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y_text = y_raw.astype(str).str.strip().str.lower()
        target_map = {"ckd": 1, "notckd": 0, "not ckd": 0, "1": 1, "0": 0}
        if set(y_text.dropna().unique()).issubset(set(target_map.keys())):
            y = y_text.map(target_map)
        else:
            y = pd.Series(LabelEncoder().fit_transform(y_text))

    y = pd.Series(y)
    if y.isna().any():
        raise ValueError(f"Target column '{target_col}' contains nulls after encoding.")
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_num = pd.DataFrame(index=X_train.index)
    X_test_num = pd.DataFrame(index=X_test.index)
    if num_cols:
        num_imputer = KNNImputer(n_neighbors=5)
        X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
        X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

        scaler = MinMaxScaler()
        X_train_num = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols, index=X_train.index)
        X_test_num = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols, index=X_test.index)
    else:
        num_imputer = None
        scaler = None

    X_train_cat = pd.DataFrame(index=X_train.index)
    X_test_cat = pd.DataFrame(index=X_test.index)
    encoders = {}
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
        X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns=cat_cols, index=X_test.index)

        for col in cat_cols:
            train_vals = X_train_cat[col].astype(str)
            test_vals = X_test_cat[col].astype(str)
            classes = sorted(train_vals.unique())
            mapping = {label: idx for idx, label in enumerate(classes)}
            X_train_cat[col] = train_vals.map(mapping).astype(int)
            X_test_cat[col] = test_vals.map(mapping).fillna(-1).astype(int)
            encoders[col] = mapping
    else:
        cat_imputer = None

    X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

    return X_train_processed, X_test_processed, y_train, y_test, num_imputer, scaler, cat_imputer, encoders


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def train_models(X_train_dict, y_train_dict):
    models = {
        "ckd": DecisionTreeClassifier(random_state=42),
        "hypertension": LogisticRegression(max_iter=1000, random_state=42),
        "diabetes": GaussianNB(),
    }

    fitted_models = {}
    for name, model in models.items():
        fitted_models[name] = model.fit(X_train_dict[name], y_train_dict[name])

    return fitted_models


def evaluate_models(fitted_models, X_test_dict, y_test_dict):
    metrics = {}
    for name, model in fitted_models.items():
        X_test = X_test_dict[name]
        y_test = y_test_dict[name]

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        macro_f1 = f1_score(y_test, preds, average="macro")
        roc_auc = roc_auc_score(y_test, probs)
        metrics[name] = {"Macro_F1": float(macro_f1), "ROC_AUC": float(roc_auc)}

        print("\n" + "=" * 70)
        print(f"Model: {name.upper()}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test, preds, digits=4))

    return metrics


def save_artifacts(models, feature_names, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, path / f"{name}_model.joblib")
        (path / f"{name}_features.json").write_text(json.dumps(feature_names[name]), encoding="utf-8")


def save_processed_splits(prefix, target_col, X_train_smote, y_train_smote, X_test, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = pd.concat([X_train_smote, pd.Series(y_train_smote, name=target_col)], axis=1)
    test_out = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True).rename(target_col)], axis=1)

    train_out.to_csv(out_dir / f"{prefix}_train_smote.csv", index=False)
    test_out.to_csv(out_dir / f"{prefix}_test.csv", index=False)


def load_preprocessed_splits(prefix: str, processed_dir: Path):
    train_file = processed_dir / f"{prefix}_train_smote.csv"
    test_file = processed_dir / f"{prefix}_test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Missing preprocessed files for '{prefix}'. Expected: {train_file} and {test_file}"
        )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    target_col = train_df.columns[-1]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    test_target_col = test_df.columns[-1]
    X_test = test_df.drop(columns=[test_target_col])
    y_test = test_df[test_target_col].astype(int)

    return X_train, X_test, y_train, y_test


def prepare_dataset(df: pd.DataFrame, target_col: str, prefix: str, out_dir: Path, is_diabetes: bool = False):
    raw_X = df.drop(columns=[target_col]).copy()
    num_cols, cat_cols, _ = infer_numeric_categorical(raw_X)

    X_train, X_test, y_train, y_test, *_ = preprocess_and_split(
        df=df,
        target_col=target_col,
        num_cols=num_cols,
        cat_cols=cat_cols,
        is_diabetes=is_diabetes,
    )

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    save_processed_splits(prefix, target_col, X_train_smote, y_train_smote, X_test, y_test, out_dir)

    return X_train_smote, X_test, y_train_smote, y_test


def main():
    parser = argparse.ArgumentParser(description="Train disease models.")
    parser.add_argument(
        "--data-source",
        choices=["auto", "preprocessed", "raw"],
        default="auto",
        help="Training input source. 'preprocessed' uses *_train_smote/*_test files; 'raw' rebuilds from datasets.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional stratified cap per disease for training rows. Default 0 uses the full available training data.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    data_root = project_dir / "medical datasets"
    processed_dir = project_dir / "preprocessed_outputs"
    artifacts_dir = project_dir / "artifacts"

    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}

    preprocessed_ready = all(
        (processed_dir / f"{name}_train_smote.csv").exists() and (processed_dir / f"{name}_test.csv").exists()
        for name in ["ckd", "hypertension", "diabetes"]
    )

    use_preprocessed = args.data_source == "preprocessed" or (args.data_source == "auto" and preprocessed_ready)

    if use_preprocessed:
        X_train_dict["ckd"], X_test_dict["ckd"], y_train_dict["ckd"], y_test_dict["ckd"] = load_preprocessed_splits(
            "ckd", processed_dir
        )
        X_train_dict["hypertension"], X_test_dict["hypertension"], y_train_dict["hypertension"], y_test_dict[
            "hypertension"
        ] = load_preprocessed_splits("hypertension", processed_dir)
        X_train_dict["diabetes"], X_test_dict["diabetes"], y_train_dict["diabetes"], y_test_dict["diabetes"] = (
            load_preprocessed_splits("diabetes", processed_dir)
        )
    else:
        raw_root = data_root / "raw"
        ckd_folder = raw_root / "chronic+kidney+disease"
        ckd_file_candidates = list(ckd_folder.rglob("*.csv")) + list(ckd_folder.rglob("*.arff"))
        ckd_file = ckd_file_candidates[0] if ckd_file_candidates else None
        ckd_df = load_ckd(ckd_file)

        htn_file = raw_root / "hypertension" / "framingham_heart_study.csv"
        if not htn_file.exists():
            raise FileNotFoundError(f"Hypertension file not found: {htn_file}")
        htn_df = pd.read_csv(htn_file)

        diabetes_file = raw_root / "diabetes" / "diabetes.csv"
        if not diabetes_file.exists():
            raise FileNotFoundError(f"Diabetes file not found: {diabetes_file}")
        diabetes_df = pd.read_csv(diabetes_file)

        X_train_dict["ckd"], X_test_dict["ckd"], y_train_dict["ckd"], y_test_dict["ckd"] = prepare_dataset(
            ckd_df, "class", "ckd", processed_dir
        )
        X_train_dict["hypertension"], X_test_dict["hypertension"], y_train_dict[
            "hypertension"
        ], y_test_dict["hypertension"] = prepare_dataset(htn_df, "prevalentHyp", "hypertension", processed_dir)
        X_train_dict["diabetes"], X_test_dict["diabetes"], y_train_dict["diabetes"], y_test_dict["diabetes"] = (
            prepare_dataset(diabetes_df, "Outcome", "diabetes", processed_dir, is_diabetes=True)
        )

    cap = None if args.max_train_rows == 0 else args.max_train_rows
    if cap is not None:
        for disease in ["ckd", "hypertension", "diabetes"]:
            X_train_dict[disease], y_train_dict[disease] = stratified_sample_xy(
                X_train_dict[disease], y_train_dict[disease], cap, random_state=42
            )
            print(f"Training rows used for {disease}: {len(X_train_dict[disease]):,}")

    models = train_models(X_train_dict, y_train_dict)
    metrics = evaluate_models(models, X_test_dict, y_test_dict)

    feature_names = {name: list(X_train_dict[name].columns) for name in X_train_dict}
    save_artifacts(models, feature_names, artifacts_dir)

    summary_rows = []
    label_map = {
        "ckd": "CKD_DT_default",
        "hypertension": "HTN_LOGREG_1000",
        "diabetes": "DIAB_GNB_default",
    }
    for disease, vals in metrics.items():
        summary_rows.append(
            {
                "model": label_map[disease],
                "macro_f1": vals["Macro_F1"],
                "roc_auc": vals["ROC_AUC"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_file = project_dir / "training_results_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 70)
    print("Unified training completed.")
    print(f"Saved summary: {summary_file}")
    print(f"Saved artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
