from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATASETS = {
    "ckd": "ckd_test.csv",
    "hypertension": "hypertension_test.csv",
    "diabetes": "diabetes_test.csv",
}


def _get_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def _evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def _plot_threshold_curve(curve_df: pd.DataFrame, disease: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(curve_df["threshold"], curve_df["accuracy"], label="Accuracy", linewidth=2)
    ax.plot(curve_df["threshold"], curve_df["precision"], label="Precision", linewidth=2)
    ax.plot(curve_df["threshold"], curve_df["recall"], label="Recall", linewidth=2)
    ax.plot(curve_df["threshold"], curve_df["f1"], label="F1", linewidth=2.2)

    best_idx = curve_df["f1"].idxmax()
    best_t = float(curve_df.loc[best_idx, "threshold"])
    best_f1 = float(curve_df.loc[best_idx, "f1"])
    ax.axvline(best_t, linestyle="--", linewidth=1.5, color="black", alpha=0.7)
    ax.text(best_t + 0.005, min(0.98, best_f1 + 0.03), f"best_t={best_t:.3f}", fontsize=10)

    ax.set_title(f"{disease.upper()}: Threshold Diagnostics")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{disease}_threshold_curve.png", dpi=240)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, preds: np.ndarray, disease: str, mode: str, acc: float, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{disease.upper()} {mode} Confusion Matrix\nAccuracy={acc:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / f"{disease}_{mode.lower()}_confusion_matrix.png", dpi=240)
    plt.close(fig)


def run(project_dir: Path) -> None:
    pre_dir = project_dir / "preprocessed_outputs"
    art_dir = project_dir / "artifacts"
    out_dir = project_dir / "reports" / "current" / "model_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for disease, test_file in DATASETS.items():
        model = joblib.load(art_dir / f"{disease}_model.joblib")
        feature_order = json.loads((art_dir / f"{disease}_features.json").read_text(encoding="utf-8"))

        test_df = pd.read_csv(pre_dir / test_file)
        target_col = test_df.columns[-1]
        X_test = test_df[feature_order].copy()
        y_test = test_df[target_col].astype(int).to_numpy()

        scores = _get_scores(model, X_test)
        roc_auc = float(roc_auc_score(y_test, scores))

        default_metrics = _evaluate_at_threshold(y_test, scores, 0.5)
        default_preds = (scores >= 0.5).astype(int)

        thresholds = np.round(np.linspace(0.10, 0.90, 161), 3)
        curve_rows = [_evaluate_at_threshold(y_test, scores, float(t)) for t in thresholds]
        curve_df = pd.DataFrame(curve_rows)

        best_idx = curve_df["f1"].idxmax()
        best_row = curve_df.loc[best_idx].to_dict()
        best_threshold = float(best_row["threshold"])
        best_preds = (scores >= best_threshold).astype(int)

        _plot_threshold_curve(curve_df, disease, out_dir)
        _plot_confusion(y_test, default_preds, disease, "Default_t0.50", float(default_metrics["accuracy"]), out_dir)
        _plot_confusion(y_test, best_preds, disease, "BestF1", float(best_row["accuracy"]), out_dir)

        curve_df.to_csv(out_dir / f"{disease}_threshold_metrics.csv", index=False)

        summary_rows.append(
            {
                "dataset": disease,
                "roc_auc": round(roc_auc, 6),
                "default_threshold": 0.5,
                "default_accuracy": round(float(default_metrics["accuracy"]), 6),
                "default_precision": round(float(default_metrics["precision"]), 6),
                "default_recall": round(float(default_metrics["recall"]), 6),
                "default_f1": round(float(default_metrics["f1"]), 6),
                "best_f1_threshold": round(best_threshold, 6),
                "best_accuracy": round(float(best_row["accuracy"]), 6),
                "best_precision": round(float(best_row["precision"]), 6),
                "best_recall": round(float(best_row["recall"]), 6),
                "best_f1": round(float(best_row["f1"]), 6),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset")
    summary_csv = out_dir / "threshold_diagnostics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    md_lines = [
        "# Model Diagnostics and Threshold Tuning",
        "",
        "This report evaluates default threshold (0.50) vs tuned threshold (best F1) on test sets.",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Generated Artifacts",
        "",
        "- `{disease}_threshold_curve.png`",
        "- `{disease}_default_t0.50_confusion_matrix.png`",
        "- `{disease}_bestf1_confusion_matrix.png`",
        "- `{disease}_threshold_metrics.csv`",
        "- `threshold_diagnostics_summary.csv`",
    ]
    (out_dir / "README_THRESHOLD_DIAGNOSTICS.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved diagnostics to: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent)
