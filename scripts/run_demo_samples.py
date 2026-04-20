import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.model_registry import registry
from backend.rules import classify_risk_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions for all demo sample JSON files.")
    parser.add_argument(
        "--folder",
        type=str,
        default="demo_inputs/terminal_samples",
        help="Folder containing sample JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Sample folder not found: {folder}")

    files = sorted(folder.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in: {folder}")

    registry.load()

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        disease = payload.get("disease")
        features = payload.get("features", {})

        pred, score, model_name = registry.predict(disease, features)
        thresholds = registry.get_risk_profile(disease).get("thresholds", {})
        risk_level = classify_risk_level(score, thresholds)
        moderate = float(thresholds.get("moderate", 0.5))

        result = {
            "disease": disease,
            "model": model_name,
            "predicted_class": int(1 if score >= moderate else 0),
            "raw_model_prediction": int(pred),
            "risk_score": round(float(score), 6),
            "risk_level": risk_level,
            "risk_level_display": risk_level.upper(),
            "thresholds": {
                "moderate": round(float(thresholds.get("moderate", 0.5)), 6),
                "high": round(float(thresholds.get("high", 0.75)), 6),
            },
        }

        print(f"\n=== {file_path.name} ===")
        print(f"Risk level for {disease.upper()}: {risk_level.upper()} (score={result['risk_score']})")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
