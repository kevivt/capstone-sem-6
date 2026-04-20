from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def summarize_frame(path: Path, target_col: str | None = None, max_unique_preview: int = 12) -> dict:
    df = pd.read_csv(path)
    summary: dict[str, object] = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }

    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False).to_dict()
        summary["target_distribution"] = {str(k): int(v) for k, v in vc.items()}

    per_col = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        unique_non_null = int(series.nunique(dropna=True))
        entry: dict[str, object] = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "missing": missing,
            "missing_pct": round((missing / len(df) * 100.0) if len(df) else 0.0, 4),
            "unique_non_null": unique_non_null,
            "constant_non_null": unique_non_null <= 1,
        }

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_non_null = numeric.notna().sum()
        if numeric_non_null > 0 and numeric_non_null >= max(1, non_null * 0.8):
            entry["numeric_min"] = None if numeric_non_null == 0 else float(numeric.min())
            entry["numeric_max"] = None if numeric_non_null == 0 else float(numeric.max())
            entry["numeric_mean"] = None if numeric_non_null == 0 else round(float(numeric.mean()), 6)
        elif 0 < unique_non_null <= max_unique_preview:
            values = series.dropna().astype(str).unique().tolist()
            entry["preview_values"] = sorted(values)[:max_unique_preview]

        per_col.append(entry)

    summary["columns_detail"] = per_col
    summary["problem_columns"] = {
        "all_missing": [c["column"] for c in per_col if c["non_null"] == 0],
        "constant": [c["column"] for c in per_col if c["constant_non_null"]],
        "mostly_missing_over_50pct": [c["column"] for c in per_col if c["missing_pct"] > 50.0],
    }
    return summary


def disease_inputs() -> list[tuple[str, Path, str]]:
    base = Path(__file__).resolve().parent.parent
    return [
        ("large_ckd", base / "medical datasets" / "large" / "ckd_large.csv", "ckd_label"),
        ("large_hypertension_500k", base / "medical datasets" / "large" / "hypertension_large_500k.csv", "prevalentHyp"),
        ("large_diabetes_500k", base / "medical datasets" / "large" / "diabetes_large_500k.csv", "diabetes"),
        ("preprocessed_ckd", base / "preprocessed_outputs" / "ckd_preprocessed.csv", "ckd_label"),
        ("preprocessed_hypertension", base / "preprocessed_outputs" / "hypertension_preprocessed.csv", "prevalentHyp"),
        ("preprocessed_diabetes", base / "preprocessed_outputs" / "diabetes_preprocessed.csv", "diabetes"),
        ("raw_hypertension_framingham", base / "medical datasets" / "raw" / "hypertension" / "framingham_heart_study.csv", "prevalentHyp"),
        ("raw_diabetes_pima", base / "medical datasets" / "raw" / "diabetes" / "diabetes.csv", "Outcome"),
        ("raw_nhanes_merged", base / "medical datasets" / "raw" / "NHANES_2017_2018" / "NHANES_2017_2018_MERGED.csv", None),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit disease datasets for missingness and feature viability.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports" / "current" / "data_audit_summary.json",
        help="Where to write the audit JSON report.",
    )
    args = parser.parse_args()

    results: dict[str, object] = {}
    for label, path, target in disease_inputs():
        if not path.exists():
            results[label] = {"path": str(path), "missing_file": True}
            continue
        results[label] = summarize_frame(path, target_col=target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote audit report to {args.output}")


if __name__ == "__main__":
    main()
