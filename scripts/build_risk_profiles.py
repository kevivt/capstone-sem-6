from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelSpec:
    disease: str
    target_col: str
    prefix: str
    model_name: str
    hyperparameters: Dict[str, Any]

    def create_model(self):
        if self.model_name == "DecisionTreeClassifier":
            return DecisionTreeClassifier(**self.hyperparameters)
        if self.model_name == "SVC":
            return SVC(**self.hyperparameters)
        if self.model_name == "GaussianNB":
            return GaussianNB(**self.hyperparameters)
        if self.model_name == "LogisticRegression":
            return LogisticRegression(**self.hyperparameters)
        if self.model_name == "RandomForestClassifier":
            return RandomForestClassifier(**self.hyperparameters)
        if self.model_name == "KNeighborsClassifier":
            return KNeighborsClassifier(**self.hyperparameters)
        raise ValueError(f"Unsupported model type: {self.model_name}")


MODEL_SPECS = [
    ModelSpec(
        disease="ckd",
        target_col="class",
        prefix="ckd",
        model_name="DecisionTreeClassifier",
        hyperparameters={"random_state": 42},
    ),
    ModelSpec(
        disease="hypertension",
        target_col="prevalentHyp",
        prefix="hypertension",
        model_name="LogisticRegression",
        hyperparameters={"max_iter": 1000, "random_state": 42},
    ),
    ModelSpec(
        disease="diabetes",
        target_col="Outcome",
        prefix="diabetes",
        model_name="GaussianNB",
        hyperparameters={},
    ),
]


DISEASE_FULL_NAMES = {
    "ckd": "Chronic Kidney Disease",
    "hypertension": "Hypertension",
    "diabetes": "Type Two Diabetes Mellitus",
}


MODEL_FULL_NAMES = {
    "DecisionTreeClassifier": "Decision Tree Classifier",
    "KNeighborsClassifier": "K-Nearest Neighbors Classifier",
    "SVC": "Support Vector Classifier",
    "RandomForestClassifier": "Random Forest Classifier",
    "LogisticRegression": "Logistic Regression",
    "GaussianNB": "Gaussian Naive Bayes",
}


FEATURE_FULL_NAMES = {
    "dm": "Diabetes Mellitus History",
    "htn": "Hypertension History",
    "sg": "Urine Specific Gravity",
    "appet": "Appetite Status",
    "pe": "Pedal Edema",
    "al": "Urine Albumin",
    "pc": "Pus Cell Status",
    "rbc": "Red Blood Cell Status",
    "ane": "Anemia",
    "bgr": "Blood Glucose Random",
    "sysBP": "Systolic Blood Pressure",
    "diaBP": "Diastolic Blood Pressure",
    "BPMeds": "Blood Pressure Medication Use",
    "heartRate": "Heart Rate",
    "cigsPerDay": "Cigarettes Smoked Per Day",
    "BMI": "Body Mass Index",
    "male": "Biological Sex (Male)",
    "totChol": "Total Cholesterol",
    "Glucose": "Plasma Glucose Concentration",
    "Insulin": "Two Hour Serum Insulin",
    "SkinThickness": "Triceps Skin Fold Thickness",
    "Age": "Age",
    "BloodPressure": "Diastolic Blood Pressure",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Pregnancies": "Number of Pregnancies",
    "age": "Age",
    "education": "Education Level",
    "glucose": "Glucose",
}


def expand_feature_name(feature_key: str) -> str:
    full_name = FEATURE_FULL_NAMES.get(feature_key)
    if full_name:
        return f"{full_name} ({feature_key})"
    return feature_key


def load_split(base_dir: Path, prefix: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(base_dir / f"{prefix}_train_smote.csv")
    test_df = pd.read_csv(base_dir / f"{prefix}_test.csv")

    resolved_target = target_col if target_col in train_df.columns else train_df.columns[-1]
    test_target = resolved_target if resolved_target in test_df.columns else test_df.columns[-1]

    X_train = train_df.drop(columns=[resolved_target])
    y_train = train_df[resolved_target].astype(int)
    X_test = test_df.drop(columns=[test_target])
    y_test = test_df[test_target].astype(int)
    return X_train, y_train, X_test, y_test


def stratified_sample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int | None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y

    sampled_idx = (
        y.groupby(y, group_keys=False)
        .apply(lambda s: s.sample(n=max(1, int(round(max_rows * (len(s) / len(y))))), random_state=random_state))
        .index
    )
    if len(sampled_idx) > max_rows:
        sampled_idx = pd.Index(sampled_idx).to_series().sample(n=max_rows, random_state=random_state).index

    return X.loc[sampled_idx].reset_index(drop=True), y.loc[sampled_idx].reset_index(drop=True)


def get_probability_scores(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        return np.asarray(scores, dtype=float)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X_test), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))

    preds = np.asarray(model.predict(X_test), dtype=float)
    return preds


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    thresholds = np.unique(np.round(np.clip(scores, 0.0, 1.0), 6))
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=float)

    best_t = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_precision = float(precision)
            best_recall = float(recall)

    moderate_threshold = float(np.clip(best_t, 0.35, 0.75))
    high_threshold = float(max(np.quantile(scores, 0.85), moderate_threshold + 0.15))
    high_threshold = float(min(0.90, high_threshold))
    if high_threshold <= moderate_threshold:
        high_threshold = float(min(0.90, moderate_threshold + 0.10))

    return {
        "moderate": round(float(moderate_threshold), 6),
        "high": round(float(high_threshold), 6),
        "best_f1": round(float(best_f1), 6),
        "precision": round(float(best_precision), 6),
        "recall": round(float(best_recall), 6),
    }


def get_top_factors(model: Any, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 10) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coefs[0]) if coefs.ndim > 1 else np.abs(coefs)
    else:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importances = np.maximum(perm.importances_mean, 0.0)

    ranking = (
        pd.DataFrame({"feature": X_test.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    total = float(ranking["importance"].sum())
    if total <= 0:
        ranking["relative_importance"] = 0.0
    else:
        ranking["relative_importance"] = ranking["importance"] / total

    return [
        {
            "feature": str(row.feature),
            "importance": round(float(row.importance), 8),
            "relative_importance": round(float(row.relative_importance), 8),
        }
        for row in ranking.itertuples(index=False)
    ]


def build_profiles(max_train_rows: int | None = None, max_eval_rows: int | None = 40000) -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "preprocessed_outputs"

    result: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diseases": {},
    }

    for spec in MODEL_SPECS:
        X_train, y_train, X_test, y_test = load_split(data_dir, spec.prefix, spec.target_col)
        X_train, y_train = stratified_sample_xy(X_train, y_train, max_train_rows, random_state=42)
        X_eval, y_eval = stratified_sample_xy(X_test, y_test, max_eval_rows, random_state=42)

        model = spec.create_model()
        model.fit(X_train, y_train)

        scores = get_probability_scores(model, X_eval)
        thresholds = find_best_threshold(y_eval.to_numpy(dtype=int), scores)
        preds = (scores >= thresholds["moderate"]).astype(int)

        disease_payload = {
            "model": spec.model_name,
            "hyperparameters": spec.hyperparameters,
            "target_col": spec.target_col,
            "thresholds": {
                "moderate": thresholds["moderate"],
                "high": thresholds["high"],
            },
            "metrics": {
                "roc_auc": round(float(roc_auc_score(y_eval, scores)), 6),
                "precision": thresholds["precision"],
                "recall": thresholds["recall"],
                "f1": thresholds["best_f1"],
            },
            "top_factors": get_top_factors(model, X_eval, y_eval),
        }
        result["diseases"][spec.disease] = disease_payload

    return result


def write_markdown_report(profile: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Disease Threshold and Risk Factor Report",
        "",
        f"Generated: {profile['generated_at']}",
        "",
    ]

    for disease, payload in profile["diseases"].items():
        disease_name = DISEASE_FULL_NAMES.get(disease, disease)
        model_name = MODEL_FULL_NAMES.get(payload["model"], payload["model"])

        lines.append(f"## {disease_name}")
        lines.append("")
        lines.append(f"- Model: {model_name}")
        lines.append(f"- Moderate risk threshold: {payload['thresholds']['moderate']}")
        lines.append(f"- High risk threshold: {payload['thresholds']['high']}")
        lines.append(
            "- Area Under the Receiver Operating Characteristic Curve: "
            f"{payload['metrics']['roc_auc']}"
        )
        lines.append(f"- Precision: {payload['metrics']['precision']}")
        lines.append(f"- Recall: {payload['metrics']['recall']}")
        lines.append(f"- F1 Score: {payload['metrics']['f1']}")
        lines.append("- Top risk factors:")
        for item in payload["top_factors"]:
            expanded_name = expand_feature_name(item["feature"])
            lines.append(
                f"  - {expanded_name} (importance={item['importance']}, relative importance={item['relative_importance']})"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disease risk thresholds and top factors.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional stratified cap per disease for model fitting. Default 0 uses the full available training data.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=40000,
        help="Optional stratified cap per disease for thresholding and factor analysis. Use 0 to disable cap.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    artifact_dir = project_dir / "artifacts"
    reports_dir = project_dir / "reports" / "current"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_cap = None if args.max_train_rows == 0 else args.max_train_rows
    eval_cap = None if args.max_eval_rows == 0 else args.max_eval_rows
    profile = build_profiles(max_train_rows=train_cap, max_eval_rows=eval_cap)

    json_path = artifact_dir / "risk_thresholds_and_factors.json"
    md_path = reports_dir / "RISK_THRESHOLDS_AND_FACTORS_REPORT.md"

    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    write_markdown_report(profile, md_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
