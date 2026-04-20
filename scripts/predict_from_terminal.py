import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.model_registry import registry
from backend.raw_input import get_raw_field_metadata, get_raw_template, map_raw_payload
from backend.rules import classify_risk_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one risk prediction from terminal.")
    parser.add_argument("--input", type=str, default=None, help="Path to JSON file with keys: disease, features")
    parser.add_argument("--raw-input", type=str, default=None, help="Path to JSON file with keys: disease, raw_inputs")
    parser.add_argument(
        "--disease",
        type=str,
        choices=["ckd", "hypertension", "diabetes"],
        default=None,
        help="Disease key when passing values inline",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Inline JSON object string for model-ready features.",
    )
    parser.add_argument(
        "--raw-features",
        type=str,
        default=None,
        help="Inline JSON object string for raw patient-friendly inputs.",
    )
    parser.add_argument("--interactive", action="store_true", help="Prompt for model-ready features (requires --disease).")
    parser.add_argument("--guided", action="store_true", help="Guided flow for model-ready features.")
    parser.add_argument("--guided-raw", action="store_true", help="Guided flow for patient-friendly raw inputs.")
    parser.add_argument("--show-template", action="store_true", help="Print required model feature columns and exit.")
    parser.add_argument("--show-raw-template", action="store_true", help="Print raw-input template and exit.")
    return parser.parse_args()


def prompt_disease() -> str:
    print("Select disease:")
    print("1) CKD")
    print("2) Hypertension")
    print("3) Diabetes")

    while True:
        raw = input("Enter 1/2/3 or disease name: ").strip().lower()
        if raw in {"1", "ckd"}:
            return "ckd"
        if raw in {"2", "hypertension", "htn"}:
            return "hypertension"
        if raw in {"3", "diabetes", "dm"}:
            return "diabetes"
        print("Invalid choice. Enter 1, 2, 3, ckd, hypertension, or diabetes.")


def build_payload(args: argparse.Namespace) -> tuple[str, dict]:
    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        disease = payload.get("disease")
        features = payload.get("features", {})
        if disease not in {"ckd", "hypertension", "diabetes"}:
            raise ValueError("Input JSON must contain disease as ckd/hypertension/diabetes.")
        if not isinstance(features, dict):
            raise ValueError("Input JSON must contain features as an object.")
        return disease, features

    if not args.disease or not args.features:
        raise ValueError("Provide either --input, or both --disease and --features.")

    features = json.loads(args.features)
    if not isinstance(features, dict):
        raise ValueError("--features must be a JSON object.")

    return args.disease, features


def build_raw_payload(args: argparse.Namespace) -> tuple[str, dict]:
    if args.raw_input:
        payload = json.loads(Path(args.raw_input).read_text(encoding="utf-8"))
        disease = payload.get("disease")
        raw_inputs = payload.get("raw_inputs", {})
        if disease not in {"ckd", "hypertension", "diabetes"}:
            raise ValueError("Raw input JSON must contain disease as ckd/hypertension/diabetes.")
        if not isinstance(raw_inputs, dict):
            raise ValueError("Raw input JSON must contain raw_inputs as an object.")
        return disease, raw_inputs

    if not args.disease or not args.raw_features:
        raise ValueError("Provide either --raw-input, or both --disease and --raw-features.")

    raw_features = json.loads(args.raw_features)
    if not isinstance(raw_features, dict):
        raise ValueError("--raw-features must be a JSON object.")

    return args.disease, raw_features


def parse_value(raw: str) -> int | float | str | bool:
    text = raw.strip()
    if text == "":
        raise ValueError("Feature value cannot be empty.")

    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"

    try:
        if "." in text or "e" in lower:
            return float(text)
        return int(text)
    except ValueError:
        return text


def prompt_features(disease: str, allow_empty_as_zero: bool = False) -> dict:
    feature_order = registry.features.get(disease)
    if not feature_order:
        raise ValueError(f"No feature schema found for disease '{disease}'.")

    print(f"Enter values for {disease.upper()} in this exact model feature order:")
    print(", ".join(feature_order))
    if allow_empty_as_zero:
        print("Tip: press Enter to use default value 0 for a field.")

    payload = {}
    for feature in feature_order:
        while True:
            raw = input(f"{feature}: ")
            try:
                if allow_empty_as_zero and raw.strip() == "":
                    payload[feature] = 0
                    break
                payload[feature] = parse_value(raw)
                break
            except ValueError as ex:
                print(f"Invalid value for {feature}: {ex}")
    return payload


def prompt_raw_inputs(disease: str) -> dict:
    fields = get_raw_field_metadata(disease)
    print(f"Enter patient-friendly raw inputs for {disease.upper()}:")
    payload = {}

    for field in fields:
        label = field["label"]
        required = field.get("required", False)
        while True:
            suffix = "" if required else " (optional, press Enter to skip)"
            raw = input(f"{label}{suffix}: ")
            try:
                if raw.strip() == "" and not required:
                    break
                payload[field["name"]] = parse_value(raw)
                break
            except ValueError as ex:
                print(f"Invalid value for {field['name']}: {ex}")
    return payload


def print_patient_summary(disease: str, result: dict) -> None:
    disease_name = {
        "ckd": "Chronic Kidney Disease",
        "hypertension": "Hypertension",
        "diabetes": "Diabetes",
    }.get(disease, disease.upper())

    advice = {
        "low": "Maintain healthy routines and continue regular checkups.",
        "moderate": "Consider a clinical follow-up and tighten lifestyle controls.",
        "high": "Strongly recommended to consult a clinician promptly for assessment.",
    }.get(result["risk_level"], "Follow up with a clinician for interpretation.")

    print("\n" + "=" * 64)
    print("PATIENT RISK SUMMARY")
    print("=" * 64)
    print(f"Disease: {disease_name}")
    print(f"Risk level: {result['risk_level_display']}")
    print(f"Risk score: {result['risk_score']}")
    print(f"Model: {result['model']}")
    print(f"Guidance: {advice}")
    print("=" * 64)


def main() -> None:
    args = parse_args()
    registry.load()

    if args.show_template:
        if not args.disease:
            raise ValueError("--show-template requires --disease.")
        feature_order = registry.features.get(args.disease)
        if not feature_order:
            raise ValueError(f"No model/features loaded for disease '{args.disease}'.")
        template = {feature: "<value>" for feature in feature_order}
        print(json.dumps({"disease": args.disease, "features": template}, indent=2))
        return

    if args.show_raw_template:
        if not args.disease:
            raise ValueError("--show-raw-template requires --disease.")
        print(json.dumps(get_raw_template(args.disease), indent=2))
        return

    transformed_features = None
    validation_warnings: list[str] = []

    if args.guided_raw:
        disease = args.disease if args.disease else prompt_disease()
        raw_inputs = prompt_raw_inputs(disease)
        mapped = map_raw_payload(disease, raw_inputs)
        disease = mapped.disease
        transformed_features = mapped.transformed_features
        validation_warnings = mapped.validation_warnings
    elif args.raw_input or args.raw_features:
        disease, raw_inputs = build_raw_payload(args)
        mapped = map_raw_payload(disease, raw_inputs)
        disease = mapped.disease
        transformed_features = mapped.transformed_features
        validation_warnings = mapped.validation_warnings
    elif args.guided:
        disease = args.disease if args.disease else prompt_disease()
        transformed_features = prompt_features(disease, allow_empty_as_zero=True)
    elif args.interactive:
        if not args.disease:
            raise ValueError("--interactive requires --disease.")
        disease = args.disease
        transformed_features = prompt_features(disease)
    else:
        disease, transformed_features = build_payload(args)

    pred, score, model_name = registry.predict(disease, transformed_features)
    thresholds = registry.get_risk_profile(disease).get("thresholds", {})
    risk_level = classify_risk_level(score, thresholds)
    moderate = float(thresholds.get("moderate", 0.5))
    predicted_class = int(1 if score >= moderate else 0)

    result = {
        "disease": disease,
        "model": model_name,
        "predicted_class": predicted_class,
        "raw_model_prediction": pred,
        "risk_score": round(float(score), 6),
        "risk_level": risk_level,
        "risk_level_display": risk_level.upper(),
        "thresholds": {
            "moderate": round(float(thresholds.get("moderate", 0.5)), 6),
            "high": round(float(thresholds.get("high", 0.75)), 6),
        },
    }
    if validation_warnings:
        result["validation_warnings"] = validation_warnings
    if args.raw_input or args.raw_features or args.guided_raw:
        result["transformed_features"] = transformed_features

    print(f"Risk level for {disease.upper()}: {risk_level.upper()} (score={result['risk_score']})")
    if validation_warnings:
        print("Warnings:")
        for warning in validation_warnings:
            print(f"- {warning}")
    if args.guided or args.guided_raw:
        print_patient_summary(disease, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
