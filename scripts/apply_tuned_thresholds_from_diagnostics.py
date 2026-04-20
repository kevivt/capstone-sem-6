from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def run(project_dir: Path) -> None:
    diagnostics_csv = project_dir / "reports" / "current" / "model_diagnostics" / "threshold_diagnostics_summary.csv"
    risk_cfg_path = project_dir / "artifacts" / "risk_thresholds_and_factors.json"

    if not diagnostics_csv.exists():
        raise FileNotFoundError(f"Diagnostics summary not found: {diagnostics_csv}")
    if not risk_cfg_path.exists():
        raise FileNotFoundError(f"Risk profile config not found: {risk_cfg_path}")

    df = pd.read_csv(diagnostics_csv)
    payload = json.loads(risk_cfg_path.read_text(encoding="utf-8"))
    diseases = payload.get("diseases", {})

    backup_name = f"risk_thresholds_and_factors.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    backup_path = risk_cfg_path.parent / backup_name
    backup_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    change_rows = []

    for _, row in df.iterrows():
        disease = str(row["dataset"]).strip().lower()
        if disease not in diseases:
            continue

        tuned_moderate = float(row["best_f1_threshold"])
        tuned_moderate = max(0.10, min(0.90, tuned_moderate))

        thresholds = diseases[disease].get("thresholds", {})
        old_moderate = float(thresholds.get("moderate", 0.35))
        old_high = float(thresholds.get("high", 0.90))

        # Keep high band meaningful and ordered above moderate.
        new_high = max(old_high, round(tuned_moderate + 0.15, 6))
        new_high = min(0.99, new_high)

        thresholds["moderate"] = round(tuned_moderate, 6)
        thresholds["high"] = round(new_high, 6)
        diseases[disease]["thresholds"] = thresholds

        # Store diagnostics trace for reproducibility.
        diseases[disease]["threshold_tuning"] = {
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "reports/current/model_diagnostics/threshold_diagnostics_summary.csv",
            "objective": "maximize_f1",
            "old_moderate": old_moderate,
            "new_moderate": thresholds["moderate"],
            "old_high": old_high,
            "new_high": thresholds["high"],
            "default_f1": float(row["default_f1"]),
            "best_f1": float(row["best_f1"]),
            "default_accuracy": float(row["default_accuracy"]),
            "best_accuracy": float(row["best_accuracy"]),
        }

        change_rows.append(
            {
                "dataset": disease,
                "old_moderate": old_moderate,
                "new_moderate": thresholds["moderate"],
                "old_high": old_high,
                "new_high": thresholds["high"],
                "default_f1": float(row["default_f1"]),
                "best_f1": float(row["best_f1"]),
                "default_accuracy": float(row["default_accuracy"]),
                "best_accuracy": float(row["best_accuracy"]),
            }
        )

    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    risk_cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_dir = project_dir / "reports" / "current" / "model_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    change_df = pd.DataFrame(change_rows).sort_values("dataset")
    change_csv = out_dir / "threshold_tuning_applied_changes.csv"
    change_df.to_csv(change_csv, index=False)

    md_lines = [
        "# Threshold Tuning Applied",
        "",
        f"- Source diagnostics: `{diagnostics_csv.relative_to(project_dir)}`",
        f"- Updated config: `{risk_cfg_path.relative_to(project_dir)}`",
        f"- Backup created: `{backup_path.relative_to(project_dir)}`",
        "",
        "## Changes",
        "",
        change_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "README_THRESHOLD_TUNING_APPLIED.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Updated thresholds: {risk_cfg_path}")
    print(f"Backup: {backup_path}")
    print(f"Change report: {change_csv}")
    print(change_df.to_string(index=False))


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent)
