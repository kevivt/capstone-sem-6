from pathlib import Path
from typing import Optional, Tuple
import argparse

import numpy as np
import pandas as pd
from scipy.io import arff
from pandas.api.types import is_numeric_dtype
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE


MISSING_TOKENS = ["?", "\t?", " ?", "\t?\t", "", " "]
KNN_MAX_ROWS = 50000


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

    # Fallback if local CKD archive is not extracted.
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


def preprocess_mixed_dataframe(df: pd.DataFrame, target_col: str, output_prefix: str, out_dir: Path) -> None:
    df = df.copy()
    df = df.replace(MISSING_TOKENS, np.nan)
    df.columns = [str(c).strip() for c in df.columns]

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col].copy()

    if is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y_text = y_raw.astype(str).str.strip().str.lower()
        target_map = {"ckd": 1, "notckd": 0, "not ckd": 0, "1": 1, "0": 0}
        if set(y_text.dropna().unique()).issubset(set(target_map.keys())):
            y = y_text.map(target_map)
        else:
            y = pd.Series(LabelEncoder().fit_transform(y_text))

    if pd.Series(y).isna().any():
        raise ValueError(f"Target column '{target_col}' contains unmapped/null values after encoding.")

    y = pd.Series(y).astype(int)

    num_cols, cat_cols, X = infer_numeric_categorical(X)

    if num_cols:
        # KNNImputer is expensive at very large row counts; use median imputation for scale.
        if len(X) > KNN_MAX_ROWS:
            X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
        else:
            X[num_cols] = KNNImputer(n_neighbors=5).fit_transform(X[num_cols])
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    for col in cat_cols:
        # Some survey-derived columns can be entirely missing for a disease subset.
        # Use a stable placeholder instead of fitting an imputer on an empty feature.
        if X[col].notna().sum() == 0:
            X[col] = "missing"
        else:
            X[col] = SimpleImputer(strategy="most_frequent").fit_transform(X[[col]])[:, 0]
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    preprocessed = X.copy()
    preprocessed[target_col] = y.values

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{output_prefix}_preprocessed.csv"
    preprocessed.to_csv(full_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    pd.concat([X_train_balanced, y_train_balanced.rename(target_col)], axis=1).to_csv(
        out_dir / f"{output_prefix}_train_smote.csv", index=False
    )
    pd.concat([X_test, y_test.rename(target_col)], axis=1).to_csv(
        out_dir / f"{output_prefix}_test.csv", index=False
    )

    print(f"\n[{output_prefix.upper()}] Preprocessing complete")
    print(f"Target column: {target_col}")
    print(f"Rows, columns: {preprocessed.shape}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
    print(f"Saved: {full_path}")
    print(f"Saved: {out_dir / f'{output_prefix}_train_smote.csv'}")
    print(f"Saved: {out_dir / f'{output_prefix}_test.csv'}")


def preprocess_diabetes(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    # In Pima diabetes dataset, zeros in these fields are often placeholders for missing.
    zero_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_missing_cols:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    preprocess_mixed_dataframe(df, target_col="Outcome", output_prefix="diabetes", out_dir=out_dir)


def preprocess_large_datasets(
    base: Path,
    out_dir: Path,
    equalized: bool = False,
    vicinity: bool = False,
    targeted_500k: bool = False,
) -> None:
    large_dir = base / "medical datasets" / "large"
    if equalized:
        large_dir = large_dir / "equalized"
    if vicinity:
        large_dir = large_dir / "vicinity"

    if targeted_500k:
        ckd_file = base / "medical datasets" / "large" / "ckd_large.csv"
        htn_file = base / "medical datasets" / "large" / "hypertension_large_500k.csv"
        diabetes_file = base / "medical datasets" / "large" / "diabetes_large_500k.csv"
    elif equalized:
        ckd_file = large_dir / "ckd_large_equalized.csv"
        htn_file = large_dir / "hypertension_large_equalized.csv"
        diabetes_file = large_dir / "diabetes_large_equalized.csv"
    elif vicinity:
        ckd_file = large_dir / "ckd_large_vicinity.csv"
        htn_file = large_dir / "hypertension_large_vicinity.csv"
        diabetes_file = large_dir / "diabetes_large_vicinity.csv"
    else:
        ckd_file = large_dir / "ckd_large.csv"
        htn_file = large_dir / "hypertension_large.csv"
        diabetes_file = large_dir / "diabetes_large.csv"

    required = [ckd_file, htn_file, diabetes_file]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Large dataset files missing. Build large datasets first, or run balance_large_disease_datasets.py for equalized inputs. "
            f"Missing: {missing}"
        )

    ckd_df = pd.read_csv(ckd_file)
    preprocess_mixed_dataframe(ckd_df, target_col="ckd_label", output_prefix="ckd", out_dir=out_dir)

    htn_df = pd.read_csv(htn_file)
    preprocess_mixed_dataframe(htn_df, target_col="prevalentHyp", output_prefix="hypertension", out_dir=out_dir)

    diabetes_df = pd.read_csv(diabetes_file)
    preprocess_mixed_dataframe(diabetes_df, target_col="diabetes", output_prefix="diabetes", out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess disease datasets for training.")
    parser.add_argument(
        "--source",
        choices=["auto", "large", "large_equalized", "large_vicinity", "large_500k", "legacy"],
        default="auto",
        help="Dataset source: large, large_equalized, large_vicinity, large_500k, legacy (original small files), or auto.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_root = base / "medical datasets"
    out_dir = base / "preprocessed_outputs"

    large_dir = data_root / "large"
    large_ready = all(
        (large_dir / name).exists()
        for name in ["ckd_large.csv", "hypertension_large.csv", "diabetes_large.csv"]
    )

    equalized_ready = all(
        (large_dir / "equalized" / name).exists()
        for name in ["ckd_large_equalized.csv", "hypertension_large_equalized.csv", "diabetes_large_equalized.csv"]
    )

    vicinity_ready = all(
        (large_dir / "vicinity" / name).exists()
        for name in ["ckd_large_vicinity.csv", "hypertension_large_vicinity.csv", "diabetes_large_vicinity.csv"]
    )

    targeted_500k_ready = all(
        (large_dir / name).exists()
        for name in ["ckd_large.csv", "hypertension_large_500k.csv", "diabetes_large_500k.csv"]
    )

    if args.source == "large_500k":
        preprocess_large_datasets(base, out_dir, targeted_500k=True)
        return

    if args.source == "large_vicinity":
        preprocess_large_datasets(base, out_dir, vicinity=True)
        return

    if args.source == "large_equalized":
        preprocess_large_datasets(base, out_dir, equalized=True)
        return

    if args.source == "large" or (args.source == "auto" and large_ready and not equalized_ready and not vicinity_ready):
        preprocess_large_datasets(base, out_dir, equalized=False)
        return

    if args.source == "auto" and vicinity_ready:
        preprocess_large_datasets(base, out_dir, vicinity=True)
        return

    if args.source == "auto" and targeted_500k_ready:
        preprocess_large_datasets(base, out_dir, targeted_500k=True)
        return

    if args.source == "auto" and equalized_ready:
        preprocess_large_datasets(base, out_dir, equalized=True)
        return

    raw_root = data_root / "raw"
    ckd_folder = raw_root / "chronic+kidney+disease"
    htn_file = raw_root / "hypertension" / "framingham_heart_study.csv"
    diabetes_file = raw_root / "diabetes" / "diabetes.csv"

    ckd_file_candidates = list(ckd_folder.rglob("*.csv")) + list(ckd_folder.rglob("*.arff"))
    ckd_file = ckd_file_candidates[0] if ckd_file_candidates else None

    ckd_df = load_ckd(ckd_file)
    preprocess_mixed_dataframe(ckd_df, target_col="class", output_prefix="ckd", out_dir=out_dir)

    if not htn_file.exists():
        raise FileNotFoundError(f"Hypertension file not found: {htn_file}")
    htn_df = pd.read_csv(htn_file)
    preprocess_mixed_dataframe(htn_df, target_col="prevalentHyp", output_prefix="hypertension", out_dir=out_dir)

    if not diabetes_file.exists():
        raise FileNotFoundError(f"Diabetes file not found: {diabetes_file}")
    diabetes_df = pd.read_csv(diabetes_file)
    preprocess_diabetes(diabetes_df, out_dir)


if __name__ == "__main__":
    main()
