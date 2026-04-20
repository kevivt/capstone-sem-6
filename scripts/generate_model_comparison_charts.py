from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["accuracy", "macro_f1", "roc_auc"]
METRIC_TITLES = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC AUC",
}
DISEASE_DISPLAY = {
    "ckd": "Chronic Kidney Disease (CKD)",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
}
MODEL_DISPLAY = {
    "logistic_regression": "LogReg",
    "decision_tree": "DecisionTree",
    "random_forest": "RandomForest",
    "knn_k5": "KNN(k=5)",
    "gaussian_nb": "GaussianNB",
    "project_final_model": "ProjectFinal",
}


def _model_label(row: pd.Series) -> str:
    key = str(row["model_key"])
    if key == "project_final_model":
        return f"ProjectFinal ({row['model']})"
    short = MODEL_DISPLAY.get(key, key)
    return f"{short} ({row['model']})"


def _plot_per_disease(df: pd.DataFrame, disease: str, out_dir: Path) -> None:
    subset = df[df["dataset"] == disease].copy()
    subset = subset.sort_values(["macro_f1", "roc_auc"], ascending=[False, False]).reset_index(drop=True)
    labels = [_model_label(row) for _, row in subset.iterrows()]

    x = np.arange(len(subset))
    width = 0.24

    fig, ax = plt.subplots(figsize=(16, 7))

    for idx, metric in enumerate(METRICS):
        offset = (idx - 1) * width
        values = subset[metric].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=METRIC_TITLES[metric])

        for bar_idx, bar in enumerate(bars):
            is_project = bool(subset.iloc[bar_idx]["is_project_model"])
            if is_project:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.6)
                bar.set_hatch("//")

    ax.set_title(f"{DISEASE_DISPLAY.get(disease, disease)}: Project Model vs Baselines")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")

    note = "Hatched bars indicate project-final model"
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{disease}_all_models_vs_project.png", dpi=220)
    plt.close(fig)


def _plot_overview(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for disease in sorted(df["dataset"].unique()):
        d = df[df["dataset"] == disease]
        project = d[d["is_project_model"]].iloc[0]
        baseline = d[~d["is_project_model"]].sort_values("macro_f1", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": disease,
                "project_macro_f1": float(project["macro_f1"]),
                "baseline_macro_f1": float(baseline["macro_f1"]),
                "project_roc_auc": float(project["roc_auc"]),
                "baseline_roc_auc": float(baseline["roc_auc"]),
            }
        )

    comp = pd.DataFrame(rows)
    x = np.arange(len(comp))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, comp["project_macro_f1"], width, label="Project")
    axes[0].bar(x + width / 2, comp["baseline_macro_f1"], width, label="Best Baseline")
    axes[0].set_title("Macro F1: Project vs Best Baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DISEASE_DISPLAY.get(d, d) for d in comp["dataset"]], rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, comp["project_roc_auc"], width, label="Project")
    axes[1].bar(x + width / 2, comp["baseline_roc_auc"], width, label="Best Baseline")
    axes[1].set_title("ROC AUC: Project vs Best Baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DISEASE_DISPLAY.get(d, d) for d in comp["dataset"]], rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "overview_project_vs_best_baseline.png", dpi=220)
    plt.close(fig)


def _plot_project_vs_best_per_disease(df: pd.DataFrame, out_dir: Path) -> None:
    for disease in sorted(df["dataset"].unique()):
        d = df[df["dataset"] == disease]
        project = d[d["is_project_model"]].iloc[0]
        baseline = d[~d["is_project_model"]].sort_values("macro_f1", ascending=False).iloc[0]

        rows = pd.DataFrame(
            [
                {
                    "label": "Project",
                    "accuracy": float(project["accuracy"]),
                    "macro_f1": float(project["macro_f1"]),
                    "roc_auc": float(project["roc_auc"]),
                },
                {
                    "label": "Best Baseline",
                    "accuracy": float(baseline["accuracy"]),
                    "macro_f1": float(baseline["macro_f1"]),
                    "roc_auc": float(baseline["roc_auc"]),
                },
            ]
        )

        x = np.arange(len(METRICS))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x - width / 2, rows.iloc[0][METRICS], width=width, label="Project", color="#4e79a7")
        ax.bar(x + width / 2, rows.iloc[1][METRICS], width=width, label="Best Baseline", color="#f28e2b")

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_TITLES[m] for m in METRICS])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_title(f"{DISEASE_DISPLAY.get(disease, disease)}: Project vs Best Baseline")

        fig.tight_layout()
        fig.savefig(out_dir / f"{disease}_project_vs_best_baseline.png", dpi=240)
        plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    report_dir = project_dir / "reports" / "current"
    baseline_csv = report_dir / "baseline_model_comparison.csv"

    if not baseline_csv.exists():
        raise FileNotFoundError(
            f"Missing baseline comparison file: {baseline_csv}. Run benchmark_baseline_models.py first."
        )

    df = pd.read_csv(baseline_csv)
    out_dir = report_dir / "figures" / "model_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    for disease in sorted(df["dataset"].unique()):
        _plot_per_disease(df, disease, out_dir)

    _plot_overview(df, out_dir)
    _plot_project_vs_best_per_disease(df, out_dir)
    print(f"Saved model comparison charts to: {out_dir}")


if __name__ == "__main__":
    main()
