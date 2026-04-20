from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    upper_map = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in upper_map:
            return upper_map[name.upper()]
    return None


def _numeric(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    col = _pick_column(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _brfss_binary(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    col = _pick_column(df, candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    raw = pd.to_numeric(df[col], errors="coerce")
    # BRFSS convention: 1 = Yes, 2 = No, 7/9 = missing/refused.
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[raw == 1] = 1.0
    out.loc[raw == 2] = 0.0
    return out


def _nhanes_binary(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    col = _pick_column(df, candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    raw = pd.to_numeric(df[col], errors="coerce")
    # NHANES convention: 1 = Yes, 2 = No, 7/9 = missing.
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[raw == 1] = 1.0
    out.loc[raw == 2] = 0.0
    return out


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xpt", ".sas7bdat"}:
        return pd.read_sas(path)
    raise ValueError(f"Unsupported file type: {path}")


def build_from_brfss(brfss_path: Path) -> pd.DataFrame:
    df = _read_dataset(brfss_path)

    age = _numeric(df, ["_AGE80", "_AGEG5YR", "X_AGE80", "AGE"])
    male_raw = _numeric(df, ["SEXVAR", "SEX", "_SEX"])
    male = pd.Series(np.nan, index=df.index, dtype="float64")
    male.loc[male_raw == 1] = 1.0
    male.loc[male_raw == 2] = 0.0

    bmi = _numeric(df, ["_BMI5", "BMI5", "X_BMI5"]) / 100.0

    # BRFSS blood pressure values are usually not provided as direct continuous SBP/DBP.
    # We keep NaN here and let downstream merge with NHANES fill continuous BP where available.
    sys_bp = pd.Series(np.nan, index=df.index, dtype="float64")
    dia_bp = pd.Series(np.nan, index=df.index, dtype="float64")
    glucose = pd.Series(np.nan, index=df.index, dtype="float64")

    diabetes = _brfss_binary(df, ["DIABETE4", "DIABETE3", "DIABETE2"])
    htn = _brfss_binary(df, ["BPHIGH6", "BPHIGH4"])
    ckd = _brfss_binary(df, ["CHCKDNY2", "CHCKDNY1"])

    smoker = _brfss_binary(df, ["SMOKE100"])
    cigs_per_day = _numeric(df, ["AVEDRNK3", "_RFSMOK3"], default=np.nan)
    education = _numeric(df, ["EDUCA"])
    prevalent_stroke = _brfss_binary(df, ["CVDSTRK3"])

    out = pd.DataFrame(
        {
            "source": "brfss",
            "age": age,
            "male": male,
            "BMI": bmi,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "glucose": glucose,
            "diabetes": diabetes,
            "prevalentHyp": htn,
            "ckd_label": ckd,
            "currentSmoker": smoker,
            "cigsPerDay": cigs_per_day,
            "education": education,
            "prevalentStroke": prevalent_stroke,
        }
    )
    return out


def build_from_nhanes(nhanes_path: Path) -> pd.DataFrame:
    df = _read_dataset(nhanes_path)

    age = _numeric(df, ["RIDAGEYR"])
    sex = _numeric(df, ["RIAGENDR"])
    male = pd.Series(np.nan, index=df.index, dtype="float64")
    male.loc[sex == 1] = 1.0
    male.loc[sex == 2] = 0.0

    bmi = _numeric(df, ["BMXBMI"])

    sbp_cols = [
        _pick_column(df, ["BPXSY1"]),
        _pick_column(df, ["BPXSY2"]),
        _pick_column(df, ["BPXSY3"]),
        _pick_column(df, ["BPXSY4"]),
    ]
    sbp_cols = [c for c in sbp_cols if c is not None]
    sys_bp = (
        df[sbp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        if sbp_cols
        else pd.Series(np.nan, index=df.index)
    )

    dbp_cols = [
        _pick_column(df, ["BPXDI1"]),
        _pick_column(df, ["BPXDI2"]),
        _pick_column(df, ["BPXDI3"]),
        _pick_column(df, ["BPXDI4"]),
    ]
    dbp_cols = [c for c in dbp_cols if c is not None]
    dia_bp = (
        df[dbp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        if dbp_cols
        else pd.Series(np.nan, index=df.index)
    )

    glucose = _numeric(df, ["LBXGLU", "LBDGLUSI", "LBXGH"])

    diabetes = _nhanes_binary(df, ["DIQ010"])
    htn_question = _nhanes_binary(df, ["BPQ020"])
    htn_rule = ((sys_bp >= 140) | (dia_bp >= 90)).astype("float64")
    prevalent_hyp = htn_question.copy()
    prevalent_hyp.loc[prevalent_hyp.isna()] = htn_rule.loc[prevalent_hyp.isna()]

    ckd = _nhanes_binary(df, ["KIQ022", "KIQ021"])

    smoker = _nhanes_binary(df, ["SMQ020"])
    cigs_per_day = _numeric(df, ["SMD641", "SMD650"], default=np.nan)
    education = _numeric(df, ["DMDEDUC2", "DMDEDUC3"])
    prevalent_stroke = _nhanes_binary(df, ["MCQ160F"])

    out = pd.DataFrame(
        {
            "source": "nhanes",
            "age": age,
            "male": male,
            "BMI": bmi,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "glucose": glucose,
            "diabetes": diabetes,
            "prevalentHyp": prevalent_hyp,
            "ckd_label": ckd,
            "currentSmoker": smoker,
            "cigsPerDay": cigs_per_day,
            "education": education,
            "prevalentStroke": prevalent_stroke,
        }
    )
    return out


def _normalize_for_training(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # Keep only plausible physiology values to reduce obvious survey artifacts.
    data = data[(data["age"].isna()) | ((data["age"] >= 18) & (data["age"] <= 100))]
    data = data[(data["BMI"].isna()) | ((data["BMI"] >= 10) & (data["BMI"] <= 80))]
    data = data[(data["sysBP"].isna()) | ((data["sysBP"] >= 70) & (data["sysBP"] <= 300))]
    data = data[(data["diaBP"].isna()) | ((data["diaBP"] >= 30) & (data["diaBP"] <= 200))]
    data = data[(data["glucose"].isna()) | ((data["glucose"] >= 20) & (data["glucose"] <= 700))]

    return data.reset_index(drop=True)


def export_disease_tables(unified: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

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
    htn_df = unified[htn_cols].dropna(subset=["prevalentHyp"]).copy()
    htn_path = out_dir / "hypertension_large.csv"
    htn_df.to_csv(htn_path, index=False)

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
    diab_df = unified[diab_cols].dropna(subset=["diabetes"]).copy()
    diab_path = out_dir / "diabetes_large.csv"
    diab_df.to_csv(diab_path, index=False)

    ckd_cols = [
        "age",
        "male",
        "BMI",
        "sysBP",
        "diaBP",
        "glucose",
        "diabetes",
        "prevalentHyp",
        "ckd_label",
        "source",
    ]
    ckd_df = unified[ckd_cols].dropna(subset=["ckd_label"]).copy()
    ckd_path = out_dir / "ckd_large.csv"
    ckd_df.to_csv(ckd_path, index=False)

    return {
        "hypertension_large": int(len(htn_df)),
        "diabetes_large": int(len(diab_df)),
        "ckd_large": int(len(ckd_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build large disease datasets from BRFSS and NHANES files.")
    parser.add_argument("--brfss", type=Path, required=False, help="Path to BRFSS CSV/XPT file")
    parser.add_argument("--nhanes", type=Path, required=False, help="Path to NHANES CSV/XPT file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "medical datasets" / "large",
        help="Folder to write unified and disease-specific large datasets",
    )
    args = parser.parse_args()

    if not args.brfss and not args.nhanes:
        raise ValueError("Provide at least one source file using --brfss or --nhanes")

    frames: list[pd.DataFrame] = []
    if args.brfss:
        frames.append(build_from_brfss(args.brfss))
    if args.nhanes:
        frames.append(build_from_nhanes(args.nhanes))

    unified = pd.concat(frames, ignore_index=True)
    unified = _normalize_for_training(unified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unified_path = args.output_dir / "unified_large_health.csv"
    unified.to_csv(unified_path, index=False)

    counts = export_disease_tables(unified, args.output_dir)
    summary = {
        "rows_unified": int(len(unified)),
        "rows_by_source": {k: int(v) for k, v in unified["source"].value_counts(dropna=False).to_dict().items()},
        "rows_by_disease_table": counts,
        "columns": list(unified.columns),
    }

    summary_path = args.output_dir / "large_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Large dataset build complete")
    print(f"Unified output: {unified_path}")
    print(f"Summary output: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
