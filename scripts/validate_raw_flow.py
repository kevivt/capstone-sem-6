import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.model_registry import registry
from backend.raw_input import map_raw_payload
from backend.rules import build_calibration_context, build_plan


def _load_raw_sample(disease: str) -> dict:
    sample_path = PROJECT_DIR / "demo_inputs" / "raw_samples" / f"{disease}_raw_sample_1.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    return payload["raw_inputs"]


def validate_alignment() -> None:
    print("=== Raw-to-Feature Alignment Validation ===")
    registry.load()
    ok = True
    for disease in ["ckd", "hypertension", "diabetes"]:
        raw_inputs = _load_raw_sample(disease)
        mapped = map_raw_payload(disease, raw_inputs)
        expected = registry.features.get(disease, [])
        got = list(mapped.transformed_features.keys())
        is_match = expected == got
        print(f"{disease}: feature_order_match={is_match}, expected={len(expected)}, got={len(got)}")
        if not is_match:
            ok = False
            print(f"  expected[:5]={expected[:5]}")
            print(f"  got[:5]={got[:5]}")
        if mapped.validation_warnings:
            print(f"  warnings={len(mapped.validation_warnings)}")
    if not ok:
        raise SystemExit("Feature-order validation failed.")


def validate_plan_sensitivity() -> None:
    print("\n=== Diet Plan Sensitivity Validation ===")
    registry.load()
    for disease in ["ckd", "hypertension", "diabetes"]:
        profile_cfg = registry.get_risk_profile(disease)
        thresholds = profile_cfg.get("thresholds", {})
        top_factors = profile_cfg.get("top_factors", [])
        transformed = map_raw_payload(disease, _load_raw_sample(disease)).transformed_features
        base_profile = {"calories_target": 2000}

        plans = {}
        for score in [0.2, 0.6, 0.95]:
            context = build_calibration_context(
                disease=disease,
                risk_score=score,
                thresholds=thresholds,
                top_factors=top_factors,
                transformed_features=transformed,
            )
            plans[score] = build_plan(
                disease=disease,
                risk_score=score,
                profile=base_profile,
                thresholds=thresholds,
                calibration_context=context,
            )

        low, moderate, high = plans[0.2], plans[0.6], plans[0.95]
        print(
            f"{disease}: sodium(low/mod/high)="
            f"{low['sodium_limit_mg']}/{moderate['sodium_limit_mg']}/{high['sodium_limit_mg']}, "
            f"calories={low['calories_target']}/{moderate['calories_target']}/{high['calories_target']}"
        )


if __name__ == "__main__":
    validate_alignment()
    validate_plan_sensitivity()
    print("\nValidation completed successfully.")

