from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_SPECS = {
    "ckd": {"file": "ckd_large.csv", "target": "ckd_label"},
    "hypertension": {"file": "hypertension_large.csv", "target": "prevalentHyp"},
    "diabetes": {"file": "diabetes_large.csv", "target": "diabetes"},
}


def _stratified_resample(df: pd.DataFrame, target_col: str, n_rows: int, random_state: int) -> pd.DataFrame:
    groups = []
    ratios = []
    for _, g in df.groupby(target_col, dropna=False):
        groups.append(g)
        ratios.append(len(g) / len(df))

    raw_counts = np.asarray(ratios, dtype=float) * n_rows
    base_counts = np.floor(raw_counts).astype(int)
    remainder = int(n_rows - int(base_counts.sum()))
    if remainder > 0:
        order = np.argsort(-(raw_counts - base_counts))
        for i in order[:remainder]:
            base_counts[i] += 1

    sampled_parts = []
    for idx, group in enumerate(groups):
        take = int(base_counts[idx])
        if take <= 0:
            continue
        replace = len(group) < take
        sampled_parts.append(group.sample(n=take, replace=replace, random_state=random_state + idx))

    out = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def balance_large_datasets(
    large_dir: Path,
    output_dir: Path,
    target_rows: int | None,
    random_state: int,
    mode: str,
    low_rows: int,
    high_rows: int,
) -> dict:
    frames: dict[str, pd.DataFrame] = {}
    original_counts: dict[str, int] = {}

    for disease, spec in DATASET_SPECS.items():
        path = large_dir / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_csv(path)
        frames[disease] = df
        original_counts[disease] = int(len(df))

    if target_rows is None or target_rows <= 0:
        target_rows = min(original_counts.values())

    output_dir.mkdir(parents=True, exist_ok=True)

    balanced_counts: dict[str, int] = {}
    output_files: dict[str, str] = {}
    target_per_disease: dict[str, int] = {}

    for disease, spec in DATASET_SPECS.items():
        df = frames[disease]
        if spec["target"] not in df.columns:
            raise ValueError(f"Target column '{spec['target']}' missing in {spec['file']}")

        if mode == "vicinity":
            disease_target = int(min(max(len(df), low_rows), high_rows))
        else:
            disease_target = int(target_rows)

        target_per_disease[disease] = disease_target

        balanced_df = _stratified_resample(df, spec["target"], disease_target, random_state)
        suffix = "vicinity" if mode == "vicinity" else "equalized"
        out_path = output_dir / f"{disease}_large_{suffix}.csv"
        balanced_df.to_csv(out_path, index=False)

        balanced_counts[disease] = int(len(balanced_df))
        output_files[disease] = str(out_path)

    summary = {
        "mode": mode,
        "target_rows": int(target_rows),
        "target_rows_per_disease": target_per_disease,
        "vicinity_bounds": {"low_rows": int(low_rows), "high_rows": int(high_rows)},
        "original_counts": original_counts,
        "equalized_counts": balanced_counts,
        "output_files": output_files,
    }

    summary_name = "vicinity_dataset_summary.json" if mode == "vicinity" else "equalized_dataset_summary.json"
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create equal-sized disease datasets from existing large CSV files.")
    parser.add_argument(
        "--large-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "medical datasets" / "large",
        help="Directory containing ckd_large.csv, hypertension_large.csv, diabetes_large.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "medical datasets" / "large" / "equalized",
        help="Directory to write balanced CSV files",
    )
    parser.add_argument(
        "--mode",
        choices=["equalized", "vicinity"],
        default="equalized",
        help="equalized: exact same rows for all diseases, vicinity: row counts in same range (bounded).",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=0,
        help="Rows per disease. Use 0 to use the minimum disease row count.",
    )
    parser.add_argument(
        "--low-rows",
        type=int,
        default=10000,
        help="Lower row bound for vicinity mode.",
    )
    parser.add_argument(
        "--high-rows",
        type=int,
        default=20000,
        help="Upper row bound for vicinity mode.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.low_rows > args.high_rows:
        raise ValueError("--low-rows must be less than or equal to --high-rows.")

    if args.mode == "vicinity" and args.output_dir == Path(__file__).resolve().parent.parent / "medical datasets" / "large" / "equalized":
        args.output_dir = Path(__file__).resolve().parent.parent / "medical datasets" / "large" / "vicinity"

    summary = balance_large_datasets(
        large_dir=args.large_dir,
        output_dir=args.output_dir,
        target_rows=args.target_rows,
        random_state=args.random_state,
        mode=args.mode,
        low_rows=args.low_rows,
        high_rows=args.high_rows,
    )

    print("Balanced disease dataset generation complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
