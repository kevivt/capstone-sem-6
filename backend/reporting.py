from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FOOD_KB_PATH = BASE_DIR / "medical datasets" / "raw" / "unified_food_kb_20260222_093221.csv"


class FoodKB:
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    def load(self) -> None:
        if self._df is not None:
            return
        if not FOOD_KB_PATH.exists():
            self._df = pd.DataFrame()
            return

        df = pd.read_csv(FOOD_KB_PATH)
        keep_cols = [
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
        self._df = df[[c for c in keep_cols if c in df.columns]].copy()

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
        for col in numeric_cols:
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

        self._df = self._df.dropna(subset=["food_name"]).drop_duplicates(subset=["food_name"])

    @property
    def df(self) -> pd.DataFrame:
        self.load()
        return self._df if self._df is not None else pd.DataFrame()


food_kb = FoodKB()


def _score_foods(df: pd.DataFrame, disease: str, plan: Dict[str, Any]) -> pd.DataFrame:
    work = df.copy()

    sodium_limit = float(plan.get("sodium_limit_mg", 2000))
    potassium_limit = float(plan.get("potassium_limit_mg", 3500))
    calories_target = float(plan.get("calories_target", 2000))
    sugar_limit = float(plan.get("sugar_limit_g", 40) or 40)
    phosphorus_limit = float(plan.get("phosphorus_limit_mg", 1200) or 1200)
    fiber_target = float(plan.get("fiber_target_g", 25) or 25)

    work["score"] = 0.0

    if "sodium_mg" in work.columns:
        work["score"] += (1.0 - (work["sodium_mg"].fillna(sodium_limit) / max(sodium_limit, 1.0))).clip(0, 1) * 0.4

    if "potassium_mg" in work.columns:
        pot_weight = 0.35 if disease == "ckd" else 0.2
        work["score"] += (1.0 - (work["potassium_mg"].fillna(potassium_limit) / max(potassium_limit, 1.0))).clip(0, 1) * pot_weight

    if "energy_kcal" in work.columns:
        energy_target = max(80.0, calories_target / 8.0)
        work["score"] += (1.0 - (work["energy_kcal"].fillna(energy_target) / energy_target)).abs().clip(0, 1) * 0.1

    if disease == "diabetes" and "sugars_free_g" in work.columns:
        work["score"] += (1.0 - (work["sugars_free_g"].fillna(sugar_limit) / max(sugar_limit, 1.0))).clip(0, 1) * 0.30

    if disease == "hypertension" and "fiber_g" in work.columns:
        work["score"] += (work["fiber_g"].fillna(0.0) / max(fiber_target, 1.0)).clip(0, 1) * 0.20

    if disease == "ckd" and "phosphorus_mg" in work.columns:
        work["score"] += (1.0 - (work["phosphorus_mg"].fillna(phosphorus_limit) / max(phosphorus_limit, 1.0))).clip(0, 1) * 0.30

    if "data_completeness" in work.columns:
        work["score"] += work["data_completeness"].fillna(0.0).clip(0, 1) * 0.1

    return work


def recommend_foods(disease: str, risk_level: str, plan: Dict[str, Any], top_n: int = 8) -> List[Dict[str, Any]]:
    df = food_kb.df
    if df.empty:
        return []

    filtered = df.copy()

    if "data_completeness" in filtered.columns:
        filtered = filtered[(filtered["data_completeness"].isna()) | (filtered["data_completeness"] >= 0.7)]

    key_cols = [c for c in ["sodium_mg", "potassium_mg", "phosphorus_mg"] if c in filtered.columns]
    if key_cols:
        filtered = filtered[~((filtered[key_cols].fillna(0) <= 0).all(axis=1))]

    sodium_factor = {"low": 1.2, "moderate": 1.0, "high": 0.85}.get(risk_level, 1.0)
    potassium_factor = {"low": 1.2, "moderate": 1.0, "high": 0.85}.get(risk_level, 1.0)

    sodium_cap = float(plan.get("sodium_limit_mg", 2000)) * sodium_factor
    potassium_cap = float(plan.get("potassium_limit_mg", 3500)) * potassium_factor

    if "sodium_mg" in filtered.columns:
        filtered = filtered[(filtered["sodium_mg"].isna()) | (filtered["sodium_mg"] <= sodium_cap)]
    if "potassium_mg" in filtered.columns:
        filtered = filtered[(filtered["potassium_mg"].isna()) | (filtered["potassium_mg"] <= potassium_cap)]

    if disease == "diabetes" and "sugars_free_g" in filtered.columns:
        sugar_cap = float(plan.get("sugar_limit_g", 40) or 40)
        filtered = filtered[(filtered["sugars_free_g"].isna()) | (filtered["sugars_free_g"] <= sugar_cap)]

    if disease == "ckd" and "phosphorus_mg" in filtered.columns:
        phosphorus_cap = float(plan.get("phosphorus_limit_mg", 1200) or 1200)
        filtered = filtered[(filtered["phosphorus_mg"].isna()) | (filtered["phosphorus_mg"] <= phosphorus_cap)]

    if filtered.empty:
        filtered = df.copy()

    scored = _score_foods(filtered, disease, plan)
    top = scored.sort_values("score", ascending=False).head(max(1, top_n))

    records = []
    for row in top.to_dict(orient="records"):
        records.append(
            {
                "food_name": row.get("food_name"),
                "food_category": row.get("food_category"),
                "energy_kcal": row.get("energy_kcal"),
                "protein_g": row.get("protein_g"),
                "carbs_g": row.get("carbs_g"),
                "sugars_free_g": row.get("sugars_free_g"),
                "sodium_mg": row.get("sodium_mg"),
                "potassium_mg": row.get("potassium_mg"),
                "phosphorus_mg": row.get("phosphorus_mg"),
                "food_score": round(float(row.get("score", 0.0)), 6),
            }
        )

    return records


def build_report_text(disease: str, risk_score: float, risk_level: str, top_factors: List[Dict[str, Any]]) -> str:
    factor_list = ", ".join([str(item.get("feature")) for item in top_factors[:5]])
    disease_guidance = {
        "ckd": "Meal calibration emphasizes sodium, potassium, protein, and phosphorus control.",
        "hypertension": "Meal calibration emphasizes sodium reduction, calorie discipline, and fiber-rich choices.",
        "diabetes": "Meal calibration emphasizes carbohydrate quality, lower added sugar, and fiber support.",
    }.get(disease, "Meal calibration follows disease-specific dietary constraints.")
    return (
        f"{disease.title()} risk assessment completed. "
        f"Risk score is {risk_score:.4f} ({risk_level} risk band). "
        f"Most influential factors for this disease model include: {factor_list}. "
        f"{disease_guidance} "
        "Diet recommendations are filtered from the unified nutrition knowledge base "
        "to align with disease-specific metabolic constraints."
    )
