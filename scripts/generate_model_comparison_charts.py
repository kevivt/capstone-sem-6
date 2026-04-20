from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


METRICS = ["accuracy", "macro_f1", "roc_auc"]
METRIC_TITLES = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC AUC",
}

PLOT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
}

DATASETS = {
    "ckd": {
        "train": "ckd_train_smote.csv",
        "test": "ckd_test.csv",
    },
    "hypertension": {
        "train": "hypertension_train_smote.csv",
        "test": "hypertension_test.csv",
    },
    "diabetes": {
        "train": "diabetes_train_smote.csv",
        "test": "diabetes_test.csv",
    },
}

PROJECT_MODEL_FACTORIES = {
    "ckd": lambda: DecisionTreeClassifier(random_state=42),
    "hypertension": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "diabetes": lambda: GaussianNB(),
}

CMAP = "Blues"
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


def _load_split(base_dir: Path, dataset_name: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    spec = DATASETS[dataset_name]
    train_df = pd.read_csv(base_dir / spec["train"])
    test_df = pd.read_csv(base_dir / spec["test"])

    # Apply the same training-row cap/sampling as benchmarking so accuracies match.
    max_train_rows = _resolve_train_row_cap(base_dir.parent / "reports" / "current")
    train_target_col = train_df.columns[-1]
    if max_train_rows is not None and max_train_rows > 0:
        train_df = _stratified_sample(train_df, train_target_col, max_train_rows)

    test_target_col = test_df.columns[-1]

    X_train = train_df.drop(columns=[train_target_col])
    y_train = train_df[train_target_col].astype(int)
    X_test = test_df.drop(columns=[test_target_col])
    y_test = test_df[test_target_col].astype(int)
    return X_train, y_train, X_test, y_test


def _apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def _stratified_sample(df: pd.DataFrame, target_col: str, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df

    parts = []
    for _, group in df.groupby(target_col):
        frac = len(group) / len(df)
        n = max(1, int(round(max_rows * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=random_state))

    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return sampled


def _resolve_train_row_cap(report_dir: Path) -> int | None:
    cfg_path = report_dir / "baseline_benchmark_run_config.json"
    if not cfg_path.exists():
        # Fall back to benchmark default behavior when config is missing.
        return 120000

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cap = int(cfg.get("max_train_rows", 120000))
    if cap <= 0:
        return None
    return cap


def _exclude_duplicate_project_class(subset: pd.DataFrame) -> pd.DataFrame:
    subset = subset.copy()
    project_model_name = str(subset[subset["is_project_model"]].iloc[0]["model"])
    keep_mask = (~subset["is_project_model"]) & (subset["model"].astype(str) == project_model_name)
    subset = subset[~keep_mask].reset_index(drop=True)
    return subset


def _annotate_bar_values(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _plot_per_disease(df: pd.DataFrame, disease: str, out_dir: Path) -> None:
    subset = df[df["dataset"] == disease].copy()
    subset = _exclude_duplicate_project_class(subset)
    subset = subset.sort_values(["macro_f1", "roc_auc"], ascending=[False, False]).reset_index(drop=True)
    labels = [_model_label(row) for _, row in subset.iterrows()]

    x = np.arange(len(subset))
    width = 0.24

    fig, ax = plt.subplots(figsize=(17, 8))

    for idx, metric in enumerate(METRICS):
        offset = (idx - 1) * width
        values = subset[metric].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=METRIC_TITLES[metric])
        _annotate_bar_values(ax, bars, values)

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
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")

    note = "Hatched bars indicate project-final model (duplicate class baseline hidden)"
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)

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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    bars_p_f1 = axes[0].bar(x - width / 2, comp["project_macro_f1"], width, label="Project")
    bars_b_f1 = axes[0].bar(x + width / 2, comp["baseline_macro_f1"], width, label="Best Baseline")
    axes[0].set_title("Macro F1: Project vs Best Baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DISEASE_DISPLAY.get(d, d) for d in comp["dataset"]], rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()
    _annotate_bar_values(axes[0], bars_p_f1, comp["project_macro_f1"].to_numpy(dtype=float))
    _annotate_bar_values(axes[0], bars_b_f1, comp["baseline_macro_f1"].to_numpy(dtype=float))

    bars_p_auc = axes[1].bar(x - width / 2, comp["project_roc_auc"], width, label="Project")
    bars_b_auc = axes[1].bar(x + width / 2, comp["baseline_roc_auc"], width, label="Best Baseline")
    axes[1].set_title("ROC AUC: Project vs Best Baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DISEASE_DISPLAY.get(d, d) for d in comp["dataset"]], rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend()
    _annotate_bar_values(axes[1], bars_p_auc, comp["project_roc_auc"].to_numpy(dtype=float))
    _annotate_bar_values(axes[1], bars_b_auc, comp["baseline_roc_auc"].to_numpy(dtype=float))

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

        fig, ax = plt.subplots(figsize=(11, 6))
        project_vals = rows.iloc[0][METRICS].to_numpy(dtype=float)
        baseline_vals = rows.iloc[1][METRICS].to_numpy(dtype=float)
        bars_project = ax.bar(x - width / 2, project_vals, width=width, label="Project", color="#4e79a7")
        bars_baseline = ax.bar(x + width / 2, baseline_vals, width=width, label="Best Baseline", color="#f28e2b")

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_TITLES[m] for m in METRICS])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_title(f"{DISEASE_DISPLAY.get(disease, disease)}: Project vs Best Baseline")
        _annotate_bar_values(ax, bars_project, project_vals)
        _annotate_bar_values(ax, bars_baseline, baseline_vals)

        fig.tight_layout()
        fig.savefig(out_dir / f"{disease}_project_vs_best_baseline.png", dpi=240)
        plt.close(fig)


def _plot_project_confusion_matrices(project_dir: Path, df: pd.DataFrame, out_dir: Path) -> None:
    preprocessed_dir = project_dir / "preprocessed_outputs"

    for disease in sorted(df["dataset"].unique()):
        X_train, y_train, X_test, y_test = _load_split(preprocessed_dir, disease)
        model = PROJECT_MODEL_FACTORIES[disease]()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        expected_acc = float(df[(df["dataset"] == disease) & (df["is_project_model"])].iloc[0]["accuracy"])
        delta = abs(acc - expected_acc)
        matrix = confusion_matrix(y_test, preds, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
        disp.plot(ax=ax, cmap=CMAP, colorbar=False, values_format="d")
        ax.set_title(
            f"{DISEASE_DISPLAY.get(disease, disease)}: Project Model Confusion Matrix\nAccuracy = {acc:.4f}",
            pad=12,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        fig.tight_layout()
        fig.savefig(out_dir / f"{disease}_project_confusion_matrix.png", dpi=240)
        plt.close(fig)

        print(
            f"{disease}: confusion_accuracy={acc:.6f} csv_project_accuracy={expected_acc:.6f} delta={delta:.8f}"
        )


def main() -> None:
    _apply_plot_style()
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
    _plot_project_confusion_matrices(project_dir, df, out_dir)
    print(f"Saved model comparison charts to: {out_dir}")


if __name__ == "__main__":
    main()
