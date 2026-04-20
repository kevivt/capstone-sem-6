from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_large_medical_datasets import build_from_brfss, _normalize_for_training


def _expand_paths(items: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if any(ch in item for ch in ["*", "?"]):
            paths.extend(sorted(Path().glob(item)))
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
            paths.extend(sorted(p.glob("*.xpt")))
            paths.extend(sorted(p.glob("*.sas7bdat")))
        else:
            paths.append(p)
    unique = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _sample_exact(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(
            f"Not enough rows: requested {n}, available {len(df)}. Add more BRFSS years/files and retry."
        )
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 500k-scale diabetes and hypertension datasets from BRFSS files (download-first, no synthetic rows)."
    )
    parser.add_argument(
        "--brfss-files",
        nargs="+",
        required=True,
        help="BRFSS file paths, directory paths, or glob patterns (csv/xpt/sas7bdat).",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=500000,
        help="Rows per disease table to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "medical datasets" / "large",
        help="Output directory for generated disease tables.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    brfss_paths = _expand_paths(args.brfss_files)
    if not brfss_paths:
        raise FileNotFoundError("No BRFSS files found from --brfss-files inputs.")

    missing = [str(p) for p in brfss_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Some BRFSS files do not exist: {missing}")

    frames = []
    for path in brfss_paths:
        print(f"Reading BRFSS: {path}")
        frames.append(build_from_brfss(path))

    unified = pd.concat(frames, ignore_index=True)
    unified = _normalize_for_training(unified)

    htn_cols = [
        "male",
        "age",
        "education",
        "currentSmoker",
        "cigsPerDay",
        "prevalentStroke",
        "BMI",
        "sysBP",
        "diaBP",
        "glucose",
        "diabetes",
        "prevalentHyp",
        "source",
    ]
    diab_cols = [
        "age",
        "male",
        "BMI",
        "sysBP",
        "diaBP",
        "glucose",
        "prevalentHyp",
        "currentSmoker",
        "diabetes",
        "source",
    ]

    htn_raw = unified[htn_cols].dropna(subset=["prevalentHyp"]).copy()
    diab_raw = unified[diab_cols].dropna(subset=["diabetes"]).copy()

    htn_out = _sample_exact(htn_raw, args.target_rows, args.random_state)
    diab_out = _sample_exact(diab_raw, args.target_rows, args.random_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    htn_path = args.output_dir / "hypertension_large_500k.csv"
    diab_path = args.output_dir / "diabetes_large_500k.csv"
    htn_out.to_csv(htn_path, index=False)
    diab_out.to_csv(diab_path, index=False)

    summary = {
        "target_rows": args.target_rows,
        "inputs": [str(p) for p in brfss_paths],
        "unified_rows_after_cleaning": int(len(unified)),
        "hypertension_rows_available": int(len(htn_raw)),
        "diabetes_rows_available": int(len(diab_raw)),
        "outputs": {
            "hypertension": str(htn_path),
            "diabetes": str(diab_path),
        },
    }
    summary_path = args.output_dir / "diabetes_hypertension_500k_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("500k diabetes and hypertension datasets generated.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
