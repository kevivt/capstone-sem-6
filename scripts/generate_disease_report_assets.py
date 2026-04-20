from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISEASE_ORDER = ["ckd", "hypertension", "diabetes"]
DISEASE_DISPLAY = {
    "ckd": "Chronic Kidney Disease (CKD)",
    "hypertension": "Hypertension",
    "diabetes": "Type 2 Diabetes Mellitus",
}
DATASET_SOURCE = {
    "ckd": "Built from BRFSS (CHCKDNY*) and NHANES (KIQ022/KIQ021) through unified large-data builder.",
    "hypertension": "Built from BRFSS (BPHIGH*) and NHANES blood pressure/questionnaire fields (BPQ/BPX).",
    "diabetes": "Built from BRFSS (DIABETE*) and NHANES (DIQ010) labels.",
}
FINAL_MODEL_NAME = {
    "ckd": "DecisionTree (default, random_state=42)",
    "hypertension": "SVC (RBF, C=4, probability=True)",
    "diabetes": "GaussianNB (default)",
}
FINAL_MODEL_KEY = {
    "ckd": "CKD_DT_default",
    "hypertension": "HTN_SVM_rbf_C4",
    "diabetes": "DIAB_GNB_default",
}
TARGET_COL = {
    "ckd": "ckd_label",
    "hypertension": "prevalentHyp",
    "diabetes": "diabetes",
}


def md_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_plots(project_dir: Path, baseline_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    pre_dir = project_dir / "preprocessed_outputs"
    large_dir = project_dir / "medical datasets" / "large"
    fig_dir = project_dir / "reports" / "current" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    counts = {}

    for disease in DISEASE_ORDER:
        before = len(pd.read_csv(large_dir / f"{disease}_large.csv"))
        after = len(pd.read_csv(pre_dir / f"{disease}_preprocessed.csv"))
        train = len(pd.read_csv(pre_dir / f"{disease}_train_smote.csv"))
        test = len(pd.read_csv(pre_dir / f"{disease}_test.csv"))

        counts[disease] = {
            "before": before,
            "after": after,
            "train": train,
            "test": test,
        }

        bdf = baseline_df[baseline_df["dataset"] == disease].copy().sort_values("macro_f1", ascending=False)
        best = bdf.iloc[0]
        frow = final_df[final_df["model"] == FINAL_MODEL_KEY[disease]].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        row_labels = ["Before\nPreprocess", "After\nPreprocess", "Train\nSMOTE", "Test"]
        row_values = [before, after, train, test]
        bars = axes[0].bar(row_labels, row_values, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
        axes[0].set_title(f"{DISEASE_DISPLAY[disease]} Row Flow")
        axes[0].set_ylabel("Rows")
        axes[0].tick_params(axis="x", labelsize=9)
        axes[0].grid(axis="y", linestyle="--", alpha=0.25)
        for bar, val in zip(bars, row_values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom", fontsize=8)

        metric_labels = ["Macro F1", "ROC AUC"]
        baseline_vals = [float(best["macro_f1"]), float(best["roc_auc"])]
        final_vals = [float(frow["macro_f1"]), float(frow["roc_auc"])]
        x = [0, 1]
        w = 0.35
        axes[1].bar([i - w / 2 for i in x], baseline_vals, width=w, label=f"Best Baseline ({best['model']})", color="#76b7b2")
        axes[1].bar([i + w / 2 for i in x], final_vals, width=w, label="Final Project Model", color="#edc948")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metric_labels)
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"{DISEASE_DISPLAY[disease]} Metrics")
        axes[1].grid(axis="y", linestyle="--", alpha=0.25)
        axes[1].legend(fontsize=8)

        out_path = fig_dir / f"{disease}_summary.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    return counts


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    reports_dir = project_dir / "reports" / "current"

    baseline_df = pd.read_csv(reports_dir / "baseline_model_comparison.csv")
    final_df = pd.read_csv(project_dir / "training_results_summary.csv")

    counts = generate_plots(project_dir, baseline_df, final_df)

    lines = [
        "# Disease Modeling Readme",
        "",
        "This readme summarizes dataset origin, preprocessing, baseline training, and final model results.",
        "",
        "## Architecture-Aligned Pipeline",
        "",
        "1. Data layer: BRFSS and NHANES are merged into disease-specific large datasets.",
        "2. Preprocessing layer: cleaning, encoding, scaling, train/test split, and SMOTE on train split.",
        "3. Modeling layer: baseline model benchmark and project final model training.",
        "4. Evaluation layer: Macro F1 and ROC AUC comparisons.",
        "",
    ]

    common_steps = [
        "Missing token normalization (? and blank variants to NaN).",
        "Drop id column when present.",
        "Numeric/categorical inference using numeric-convertibility threshold.",
        "Numeric imputation: KNNImputer for <= 50000 rows, median imputer above that size.",
        "Numeric scaling with MinMaxScaler.",
        "Categorical handling: most-frequent imputation, then LabelEncoder per column.",
        "Target encoding to binary integer labels.",
        "Stratified 80/20 train/test split.",
        "SMOTE applied only to training split.",
    ]

    for disease in DISEASE_ORDER:
        bdf = baseline_df[baseline_df["dataset"] == disease].copy().sort_values("macro_f1", ascending=False)
        frow = final_df[final_df["model"] == FINAL_MODEL_KEY[disease]].iloc[0]
        best = bdf.iloc[0]

        count_df = pd.DataFrame(
            [
                {"stage": "before_preprocessing", "rows": counts[disease]["before"]},
                {"stage": "after_preprocessing", "rows": counts[disease]["after"]},
                {"stage": "train_smote", "rows": counts[disease]["train"]},
                {"stage": "test", "rows": counts[disease]["test"]},
            ]
        )

        final_comp = pd.DataFrame(
            [
                {
                    "model_type": "best_baseline",
                    "model_name": str(best["model"]),
                    "macro_f1": float(best["macro_f1"]),
                    "roc_auc": float(best["roc_auc"]),
                },
                {
                    "model_type": "final_project",
                    "model_name": FINAL_MODEL_NAME[disease],
                    "macro_f1": float(frow["macro_f1"]),
                    "roc_auc": float(frow["roc_auc"]),
                },
            ]
        )

        lines.extend(
            [
                f"## {DISEASE_DISPLAY[disease]}",
                "",
                f"Dataset source: {DATASET_SOURCE[disease]}",
                f"Input large dataset: medical datasets/large/{disease}_large.csv",
                f"Target column: {TARGET_COL[disease]}",
                "",
                "### Row Counts",
                "",
                md_table(count_df),
                "",
                "### Preprocessing Applied",
                "",
            ]
        )
        for step in common_steps:
            lines.append(f"- {step}")

        lines.extend(
            [
                "",
                "### Baseline Training Results",
                "",
                md_table(bdf[["model", "accuracy", "macro_f1", "roc_auc"]]),
                "",
                "### Final Model vs Best Baseline",
                "",
                md_table(final_comp),
                "",
                f"![{disease} summary graph](figures/{disease}_summary.png)",
                "",
            ]
        )

    out_md = reports_dir / "README_DISEASE_MODELING_SUMMARY.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_md}")
    print(f"Saved figures folder: {reports_dir / 'figures'}")


if __name__ == "__main__":
    main()
