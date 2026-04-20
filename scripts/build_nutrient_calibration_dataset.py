from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_FOOD_KB = PROJECT_DIR / "medical datasets" / "raw" / "unified_food_kb_20260222_093221.csv"
OUT_DIR = PROJECT_DIR / "reports" / "current" / "nutrient_dataset"
ARTIFACT_RULES = PROJECT_DIR / "artifacts" / "nutrient_calibration_rules.json"


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _rank_score(series: pd.Series, prefer_low: bool) -> pd.Series:
    s = series.copy()
    med = float(s.dropna().median()) if s.notna().any() else 0.0
    s = s.fillna(med)
    rank = s.rank(pct=True, method="average")
    return 1.0 - rank if prefer_low else rank


def _tier_from_score(s: pd.Series) -> pd.Series:
    q1 = float(s.quantile(0.33))
    q2 = float(s.quantile(0.66))
    return pd.Series(
        np.where(s >= q2, "high", np.where(s >= q1, "medium", "low")),
        index=s.index,
    )


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "food_name",
        "food_category",
        "energy_kcal",
        "protein_g",
        "carbs_g",
        "sugars_free_g",
        "sodium_mg",
        "potassium_mg",
        "phosphorus_mg",
        "fiber_g",
        "data_completeness",
    ]
    available = [c for c in keep if c in df.columns]
    out = df[available].copy()

    numeric_cols = [
        "energy_kcal",
        "protein_g",
        "carbs_g",
        "sugars_free_g",
        "sodium_mg",
        "potassium_mg",
        "phosphorus_mg",
        "fiber_g",
        "data_completeness",
    ]
    out = _safe_numeric(out, numeric_cols)
    out = out.dropna(subset=["food_name"]).drop_duplicates(subset=["food_name"]).reset_index(drop=True)

    completeness_cols = [c for c in ["sodium_mg", "potassium_mg", "phosphorus_mg", "sugars_free_g", "fiber_g", "carbs_g"] if c in out.columns]
    if "data_completeness" not in out.columns:
        out["data_completeness"] = out[completeness_cols].notna().mean(axis=1)
    else:
        out["data_completeness"] = out["data_completeness"].fillna(out[completeness_cols].notna().mean(axis=1))

    return out


def _score_ckd(df: pd.DataFrame) -> pd.Series:
    sodium = _rank_score(df["sodium_mg"], prefer_low=True)
    potassium = _rank_score(df["potassium_mg"], prefer_low=True)
    phosphorus = _rank_score(df["phosphorus_mg"], prefer_low=True)
    completeness = _rank_score(df["data_completeness"], prefer_low=False)
    return 0.40 * sodium + 0.30 * potassium + 0.20 * phosphorus + 0.10 * completeness


def _score_hypertension(df: pd.DataFrame) -> pd.Series:
    sodium = _rank_score(df["sodium_mg"], prefer_low=True)
    fiber = _rank_score(df["fiber_g"], prefer_low=False)
    sugar = _rank_score(df["sugars_free_g"], prefer_low=True)
    completeness = _rank_score(df["data_completeness"], prefer_low=False)
    return 0.45 * sodium + 0.25 * fiber + 0.15 * sugar + 0.15 * completeness


def _score_diabetes(df: pd.DataFrame) -> pd.Series:
    sugar = _rank_score(df["sugars_free_g"], prefer_low=True)
    fiber = _rank_score(df["fiber_g"], prefer_low=False)
    carbs = _rank_score(df["carbs_g"], prefer_low=True)
    completeness = _rank_score(df["data_completeness"], prefer_low=False)
    return 0.40 * sugar + 0.25 * fiber + 0.20 * carbs + 0.15 * completeness


def _caps_from_top(top_df: pd.DataFrame, disease: str) -> dict:
    def q(col: str, p: float, default: float) -> float:
        if col not in top_df.columns or top_df[col].dropna().empty:
            return default
        return float(top_df[col].quantile(p))

    high = {
        "sodium_mg": q("sodium_mg", 0.25, 1500.0),
        "potassium_mg": q("potassium_mg", 0.25, 2000.0),
        "phosphorus_mg": q("phosphorus_mg", 0.25, 900.0),
        "sugars_free_g": q("sugars_free_g", 0.30, 20.0),
        "carbs_g": q("carbs_g", 0.35, 45.0),
        "fiber_target_g": q("fiber_g", 0.60, 28.0),
    }
    moderate = {
        "sodium_mg": q("sodium_mg", 0.45, 1800.0),
        "potassium_mg": q("potassium_mg", 0.45, 2600.0),
        "phosphorus_mg": q("phosphorus_mg", 0.45, 1000.0),
        "sugars_free_g": q("sugars_free_g", 0.50, 30.0),
        "carbs_g": q("carbs_g", 0.55, 55.0),
        "fiber_target_g": q("fiber_g", 0.50, 25.0),
    }
    low = {
        "sodium_mg": q("sodium_mg", 0.65, 2000.0),
        "potassium_mg": q("potassium_mg", 0.65, 3200.0),
        "phosphorus_mg": q("phosphorus_mg", 0.65, 1200.0),
        "sugars_free_g": q("sugars_free_g", 0.70, 40.0),
        "carbs_g": q("carbs_g", 0.70, 65.0),
        "fiber_target_g": q("fiber_g", 0.40, 22.0),
    }

    # Disease-aware adjustments to keep values realistic.
    if disease == "ckd":
        high["potassium_mg"] = min(high["potassium_mg"], 2200.0)
        high["phosphorus_mg"] = min(high["phosphorus_mg"], 900.0)
    if disease == "diabetes":
        high["sugars_free_g"] = min(high["sugars_free_g"], 20.0)

    return {
        "low": {k: round(v, 2) for k, v in low.items()},
        "moderate": {k: round(v, 2) for k, v in moderate.items()},
        "high": {k: round(v, 2) for k, v in high.items()},
    }


def run() -> None:
    if not RAW_FOOD_KB.exists():
        raise FileNotFoundError(f"Nutrient dataset not found: {RAW_FOOD_KB}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_FOOD_KB)
    base_df = _prepare_base(raw_df)

    disease_scoring = {
        "ckd": _score_ckd(base_df),
        "hypertension": _score_hypertension(base_df),
        "diabetes": _score_diabetes(base_df),
    }

    summary_rows = []
    rules = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(RAW_FOOD_KB.name),
        "diseases": {},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for idx, (disease, score) in enumerate(disease_scoring.items()):
        tmp = base_df.copy()
        tmp["disease"] = disease
        tmp["suitability_score"] = score.round(6)
        tmp["tier"] = _tier_from_score(tmp["suitability_score"])
        tmp = tmp.sort_values("suitability_score", ascending=False).reset_index(drop=True)

        out_csv = OUT_DIR / f"{disease}_nutrient_candidates.csv"
        tmp.to_csv(out_csv, index=False)

        top_df = tmp.head(min(500, len(tmp))).copy()
        top_csv = OUT_DIR / f"{disease}_top500_candidates.csv"
        top_df.to_csv(top_csv, index=False)

        caps = _caps_from_top(top_df, disease)

        if disease == "ckd":
            weights = {
                "sodium_low_better": 0.40,
                "potassium_low_better": 0.30,
                "phosphorus_low_better": 0.20,
                "completeness_high_better": 0.10,
            }
        elif disease == "hypertension":
            weights = {
                "sodium_low_better": 0.45,
                "fiber_high_better": 0.25,
                "sugar_low_better": 0.15,
                "completeness_high_better": 0.15,
            }
        else:
            weights = {
                "sugar_low_better": 0.40,
                "fiber_high_better": 0.25,
                "carbs_low_better": 0.20,
                "completeness_high_better": 0.15,
            }

        rules["diseases"][disease] = {
            "weights": weights,
            "caps": caps,
            "candidate_counts": {
                "all_rows": int(len(tmp)),
                "top500": int(len(top_df)),
            },
        }

        summary_rows.append(
            {
                "disease": disease,
                "total_candidates": int(len(tmp)),
                "top500_count": int(len(top_df)),
                "score_mean": round(float(tmp["suitability_score"].mean()), 6),
                "score_p90": round(float(tmp["suitability_score"].quantile(0.9)), 6),
                "score_max": round(float(tmp["suitability_score"].max()), 6),
                "high_tier_count": int((tmp["tier"] == "high").sum()),
                "moderate_tier_count": int((tmp["tier"] == "medium").sum()),
                "low_tier_count": int((tmp["tier"] == "low").sum()),
            }
        )

        axes[idx].hist(tmp["suitability_score"], bins=30, color="#4e79a7", alpha=0.85)
        axes[idx].set_title(f"{disease.upper()} score distribution")
        axes[idx].set_xlabel("Suitability score")
        axes[idx].grid(axis="y", linestyle="--", alpha=0.2)

    axes[0].set_ylabel("Food count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nutrient_suitability_distributions.png", dpi=220)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows).sort_values("disease")
    summary_df.to_csv(OUT_DIR / "nutrient_dataset_summary.csv", index=False)

    ARTIFACT_RULES.write_text(json.dumps(rules, indent=2), encoding="utf-8")

    md_lines = [
        "# Nutrient Dataset Calibration Outputs",
        "",
        "Generated disease-specific nutrient suitability datasets and runtime calibration rules from the unified food KB.",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        "- `{disease}_nutrient_candidates.csv` (all candidates with suitability score)",
        "- `{disease}_top500_candidates.csv` (top-ranked foods for recommendation pipeline)",
        "- `nutrient_dataset_summary.csv`",
        "- `nutrient_suitability_distributions.png`",
        "- `artifacts/nutrient_calibration_rules.json`",
    ]
    (OUT_DIR / "README_NUTRIENT_CALIBRATION.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved nutrient calibration outputs to: {OUT_DIR}")
    print(f"Saved runtime calibration rules to: {ARTIFACT_RULES}")


if __name__ == "__main__":
    run()
