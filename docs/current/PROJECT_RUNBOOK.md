# Project Runbook

Date: 2026-04-13

This runbook is the operational source of truth for routine project execution.

## 1. Environment Setup

From workspace root:

```powershell
& ".venv/Scripts/Activate.ps1"
Set-Location "Sem_6_Capstone"
```

Install dependencies if needed:

```powershell
pip install -r requirements-ckd.txt
pip install -r requirements-backend.txt
```

## 2. Data Pipeline Order

Active disease-risk modeling now uses the disease-specific full-feature datasets:
- CKD: UCI CKD fallback through `scripts/preprocess_all_datasets.py --source legacy`
- Hypertension: `medical datasets/raw/hypertension/framingham_heart_study.csv`
- Diabetes: `medical datasets/raw/diabetes/diabetes.csv`
- Nutrition KB: `medical datasets/raw/unified_food_kb_20260222_093221.csv`

### Active preprocessing command

```powershell
python scripts/preprocess_all_datasets.py --source legacy
```

This generates:
- `preprocessed_outputs/ckd_preprocessed.csv`
- `preprocessed_outputs/hypertension_preprocessed.csv`
- `preprocessed_outputs/diabetes_preprocessed.csv`
- `preprocessed_outputs/*_train_smote.csv`
- `preprocessed_outputs/*_test.csv`

## 3. Model and Risk Profile Pipeline

```powershell
python scripts/train_models.py --data-source preprocessed
python scripts/build_risk_profiles.py
python scripts/benchmark_baseline_models.py
```

This updates:
- `artifacts/*_model.joblib`
- `artifacts/*_features.json`
- `artifacts/risk_thresholds_and_factors.json`
- `training_results_summary.csv`

## 4. Backend Validation and Run

```powershell
python scripts/backend_smoke_test.py
uvicorn backend.app:app --reload
```

Endpoints:
- `GET /health`
- `POST /predict-risk`
- `POST /recommend-plan`
- `POST /log-meal`
- `POST /risk-report`

## 5. Maintenance Rules

- Keep legacy review materials under `docs/archive/legacy_review_materials/`.
- Remove transient Python caches (`__pycache__/`) when sharing/submitting.
- Do not edit generated artifacts manually; regenerate using scripts.
