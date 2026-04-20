from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


PROJECT_MODEL_FACTORIES: Dict[str, Callable[[], ClassifierMixin]] = {
    "ckd": lambda: DecisionTreeClassifier(random_state=42),
    "hypertension": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "diabetes": lambda: GaussianNB(),
}


def _to_markdown_table(df: pd.DataFrame) -> str:
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


def _load_split(base_dir: Path, dataset_name: str, max_train_rows: int | None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
    spec = DATASETS[dataset_name]
    train_df = pd.read_csv(base_dir / spec["train"])
    test_df = pd.read_csv(base_dir / spec["test"])

    target_col = train_df.columns[-1]
    if max_train_rows is not None and max_train_rows > 0:
        train_df = _stratified_sample(train_df, target_col, max_train_rows)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    test_target_col = test_df.columns[-1]
    X_test = test_df.drop(columns=[test_target_col])
    y_test = test_df[test_target_col].astype(int)

    return X_train, y_train, X_test, y_test, target_col


def _get_score(model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def _build_baseline_models() -> Dict[str, Callable[[], ClassifierMixin]]:
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "knn_k5": lambda: KNeighborsClassifier(n_neighbors=5),
        "gaussian_nb": lambda: GaussianNB(),
    }


def _build_model_pool(dataset_name: str) -> Dict[str, Tuple[Callable[[], ClassifierMixin], bool]]:
    pool: Dict[str, Tuple[Callable[[], ClassifierMixin], bool]] = {}
    for name, builder in _build_baseline_models().items():
        pool[name] = (builder, False)
    pool["project_final_model"] = (PROJECT_MODEL_FACTORIES[dataset_name], True)
    return pool


def run_benchmark(project_dir: Path, max_train_rows: int | None) -> pd.DataFrame:
    preprocessed_dir = project_dir / "preprocessed_outputs"
    rows = []

    for dataset_name in DATASETS:
        X_train, y_train, X_test, y_test, target_col = _load_split(preprocessed_dir, dataset_name, max_train_rows)

        print(f"\nBenchmarking dataset: {dataset_name}")
        print(f"Target: {target_col} | Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

        for model_name, (model_builder, is_project_model) in _build_model_pool(dataset_name).items():
            model = model_builder()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            scores = _get_score(model, X_test)

            accuracy = float(accuracy_score(y_test, preds))
            macro_f1 = float(f1_score(y_test, preds, average="macro"))
            roc_auc = float(roc_auc_score(y_test, scores))

            rows.append(
                {
                    "dataset": dataset_name,
                    "model_key": model_name,
                    "model": model.__class__.__name__,
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "accuracy": round(accuracy, 6),
                    "macro_f1": round(macro_f1, 6),
                    "roc_auc": round(roc_auc, 6),
                    "is_project_model": bool(is_project_model),
                }
            )
            print(
                f"  {model_name:20s} accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} roc_auc={roc_auc:.4f}"
            )

    return pd.DataFrame(rows).sort_values(["dataset", "macro_f1"], ascending=[True, False]).reset_index(drop=True)


def _write_individual_reports(results: pd.DataFrame, reports_dir: Path) -> None:
    for dataset_name in DATASETS:
        subset = (
            results[results["dataset"] == dataset_name]
            .sort_values("macro_f1", ascending=False)
            .reset_index(drop=True)
        )

        project_row = subset[subset["is_project_model"]].iloc[0]
        baseline_rows = subset[~subset["is_project_model"]].reset_index(drop=True)
        best_baseline = baseline_rows.iloc[0]

        delta_macro_f1 = float(project_row["macro_f1"] - best_baseline["macro_f1"])
        delta_roc_auc = float(project_row["roc_auc"] - best_baseline["roc_auc"])
        delta_accuracy = float(project_row["accuracy"] - best_baseline["accuracy"])

        compare_df = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "candidate": "project_final_model",
                    "model": project_row["model"],
                    "accuracy": float(project_row["accuracy"]),
                    "macro_f1": float(project_row["macro_f1"]),
                    "roc_auc": float(project_row["roc_auc"]),
                },
                {
                    "dataset": dataset_name,
                    "candidate": "best_baseline",
                    "model": best_baseline["model"],
                    "accuracy": float(best_baseline["accuracy"]),
                    "macro_f1": float(best_baseline["macro_f1"]),
                    "roc_auc": float(best_baseline["roc_auc"]),
                },
            ]
        )

        csv_path = reports_dir / f"baseline_vs_project_{dataset_name}.csv"
        md_path = reports_dir / f"BASELINE_VS_PROJECT_{dataset_name.upper()}.md"
        compare_df.to_csv(csv_path, index=False)

        md_lines = [
            f"# {dataset_name.upper()} Baseline vs Project Model",
            "",
            "## Headline",
            "",
            f"- Project model: {project_row['model']}",
            f"- Best baseline: {best_baseline['model']}",
            f"- Delta Macro F1 (project - baseline): {delta_macro_f1:.6f}",
            f"- Delta ROC AUC (project - baseline): {delta_roc_auc:.6f}",
            f"- Delta Accuracy (project - baseline): {delta_accuracy:.6f}",
            "",
            "## Direct Comparison",
            "",
            _to_markdown_table(compare_df),
            "",
            "## Baseline Leaderboard",
            "",
            _to_markdown_table(baseline_rows[["model", "accuracy", "macro_f1", "roc_auc"]]),
            "",
        ]
        md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models for each disease and compare metrics.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=120000,
        help="Optional cap per dataset for training rows (for runtime control). Use 0 to disable cap.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    cap = None if args.max_train_rows == 0 else args.max_train_rows

    results = run_benchmark(project_dir, cap)

    reports_dir = project_dir / "reports" / "current"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "baseline_model_comparison.csv"
    md_path = reports_dir / "BASELINE_MODEL_COMPARISON.md"

    results.to_csv(csv_path, index=False)

    run_cfg_path = reports_dir / "baseline_benchmark_run_config.json"
    run_cfg = {
        "max_train_rows": int(cap) if cap is not None else 0,
        "sampling": "stratified",
        "sampling_random_state": 42,
    }
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    best = results.sort_values(["dataset", "macro_f1"], ascending=[True, False]).groupby("dataset", as_index=False).head(1)
    project_models = results[results["is_project_model"]].copy().sort_values("dataset").reset_index(drop=True)

    compare_rows = []
    for dataset_name in DATASETS:
        best_baseline = (
            results[(results["dataset"] == dataset_name) & (~results["is_project_model"])]
            .sort_values("macro_f1", ascending=False)
            .iloc[0]
        )
        project_row = (
            results[(results["dataset"] == dataset_name) & (results["is_project_model"])]
            .sort_values("macro_f1", ascending=False)
            .iloc[0]
        )
        compare_rows.append(
            {
                "dataset": dataset_name,
                "project_model": project_row["model"],
                "best_baseline": best_baseline["model"],
                "project_macro_f1": float(project_row["macro_f1"]),
                "baseline_macro_f1": float(best_baseline["macro_f1"]),
                "delta_macro_f1": round(float(project_row["macro_f1"] - best_baseline["macro_f1"]), 6),
                "project_roc_auc": float(project_row["roc_auc"]),
                "baseline_roc_auc": float(best_baseline["roc_auc"]),
                "delta_roc_auc": round(float(project_row["roc_auc"] - best_baseline["roc_auc"]), 6),
            }
        )
    compare_df = pd.DataFrame(compare_rows)

    _write_individual_reports(results, reports_dir)

    lines = [
        "# Baseline Model Comparison",
        "",
        "This report compares baseline classifiers on the prepared disease datasets.",
        "",
        f"Training row cap per dataset: {cap if cap is not None else 'No cap'}",
        "",
        "## Best Model Per Dataset (by Macro F1)",
        "",
        _to_markdown_table(best),
        "",
        "## Project Model vs Best Baseline (Per Disease)",
        "",
        _to_markdown_table(compare_df),
        "",
        "## Project Model Rows",
        "",
        _to_markdown_table(project_models[["dataset", "model", "accuracy", "macro_f1", "roc_auc"]]),
        "",
        "## Full Results",
        "",
        _to_markdown_table(results),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Baseline benchmarking complete")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {md_path}")
    print(f"Saved run config: {run_cfg_path}")


if __name__ == "__main__":
    main()
