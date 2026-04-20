"""
Fit the same preprocessing stack used in train_models (KNN impute + MinMax + categorical maps)
on the large unified disease CSVs and export deterministic bounds/fills for raw-input mapping.

Output: artifacts/model_input_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import train_models as tm  # noqa: E402


def _sample_df(df: pd.DataFrame, target_col: str, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    y = df[target_col]
    idx = (
        y.groupby(y, group_keys=False)
        .apply(lambda s: s.sample(n=max(1, int(round(max_rows * (len(s) / len(y))))), random_state=random_state))
        .index
    )
    if len(idx) > max_rows:
        idx = pd.Index(idx).to_series().sample(n=max_rows, random_state=random_state).index
    return df.loc[idx].reset_index(drop=True)


def export_disease(
    df: pd.DataFrame,
    target_col: str,
    disease: str,
    is_diabetes: bool,
    max_rows: int,
    random_state: int,
) -> dict:
    df = _sample_df(df, target_col, max_rows, random_state)
    raw_X = df.drop(columns=[target_col]).copy()
    num_cols, cat_cols, _ = tm.infer_numeric_categorical(raw_X)

    X_train, _, y_train, _, num_imputer, scaler, _, encoders = tm.preprocess_and_split(
        df=df,
        target_col=target_col,
        num_cols=num_cols,
        cat_cols=cat_cols,
        is_diabetes=is_diabetes,
    )

    # Rebuild imputed (pre-scale) training numerics for medians
    X_raw_train = df.drop(columns=[target_col]).loc[X_train.index]
    X_num_imp = pd.DataFrame(
        num_imputer.transform(X_raw_train[num_cols]), columns=num_cols, index=X_raw_train.index
    )
    fills = {c: float(np.nanmedian(X_num_imp[c].astype(float))) for c in num_cols}

    scale: dict = {}
    if scaler is not None:
        for j, c in enumerate(num_cols):
            scale[c] = {
                "min": float(scaler.data_min_[j]),
                "max": float(scaler.data_max_[j]),
                "fill": fills.get(c, float(np.nanmedian(X_num_imp[c].astype(float)))),
            }

    categories: dict = {}
    for col, mapping in (encoders or {}).items():
        categories[col] = {str(k): int(v) for k, v in mapping.items()}

    cat_fills: dict = {}
    if cat_cols:
        for col in cat_cols:
            mode = X_raw_train[col].mode(dropna=True)
            cat_fills[col] = str(mode.iloc[0]) if len(mode) else ""

    feature_order = list(X_train.columns)
    return {
        "disease": disease,
        "target_col": target_col,
        "feature_order": feature_order,
        "numeric_columns": list(num_cols),
        "categorical_columns": list(cat_cols),
        "scale": scale,
        "categories": categories,
        "categorical_defaults": cat_fills,
        "train_rows_used": int(len(df)),
        "positive_rate": float(y_train.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model input calibration JSON for raw mapping.")
    parser.add_argument("--max-rows", type=int, default=40000, help="Stratified cap per disease (0 = full).")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "model_input_calibration.json",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_root = PROJECT_DIR / "medical datasets" / "large"
    paths = {
        "ckd": (data_root / "ckd_large.csv", "ckd_label", False),
        "hypertension": (data_root / "hypertension_large_500k.csv", "prevalentHyp", False),
        "diabetes": (data_root / "diabetes_large_500k.csv", "diabetes", False),
    }

    payload = {
        "generated_by": "scripts/export_raw_mapping_calibration.py",
        "assumptions": [
            "Scaling matches MinMaxScaler fit on KNN-imputed training numerics (same random_state=42 split as training).",
            "Single-row inference uses training medians (post-imputation) for missing numeric fields and modal category strings.",
            "Categorical keys are case-sensitive string labels as they appeared in the training CSV after cleaning.",
        ],
        "diseases": {},
    }

    cap = None if args.max_rows == 0 else args.max_rows
    for disease, (path, target, is_dm) in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset for {disease}: {path}")
        df = pd.read_csv(path)
        payload["diseases"][disease] = export_disease(
            df, target, disease, is_dm, cap or 0, args.random_state
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
