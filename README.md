# Sem 6 Capstone - Disease Risk + Diet Recommendation

This repository contains the Capstone Phase-2 implementation for a disease-aware diet recommendation workflow that combines:

- disease risk prediction (CKD, Hypertension, Diabetes),
- risk thresholding and factor profiling,
- rule-based diet calibration,
- food recommendation from a unified nutrition knowledge base,
- backend API and terminal demo utilities.

The project is designed for reproducible evaluation while avoiding very large raw datasets in git.

## Current Project Status

- Trained model artifacts are available under `artifacts/`.
- Terminal prediction flow is working via `scripts/predict_from_terminal.py`.
- Backend smoke flow is working via `scripts/backend_smoke_test.py`.
- Review assets (reports/PPT/PDF/comparison outputs) are present in the repository.
- Current deployed model family in artifacts:
  - `ckd`: `DecisionTreeClassifier`
  - `hypertension`: `LogisticRegression`
  - `diabetes`: `GaussianNB`

## Repository Structure

- `backend/` - FastAPI backend, schemas, rule logic, reporting, persistence.
- `scripts/` - preprocessing, training, benchmarking, smoke tests, terminal flows.
- `artifacts/` - deployed model artifacts (`*_model.joblib`, `*_features.json`, thresholds/factors).
- `docs/` - runbooks, progress logs, implementation guides.
- `demo_inputs/` - sample prediction inputs.
- `reports/` - generated reports and analysis outputs.
- `esa templates/` - institute templates (PPT/report format references).
- `medical datasets/` - local dataset area (raw + generated; large files intentionally ignored).

## Why Large Files Are Ignored

The project uses very large raw medical survey files (especially BRFSS `.XPT`) that can exceed 1 GB per file. GitHub has strict upload limits (100 MB per file), and pushing those files would fail and make the repository unmanageable.

To keep the repository shareable and reviewable:

- raw BRFSS/XPT/ZIP files are ignored,
- virtual environment files are ignored,
- large generated tables are ignored by default.

This means the repository contains code, configs, documentation, and model-serving artifacts, while heavy local data stays on each contributor's machine.

## Environment Setup

From repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-ckd.txt
pip install -r requirements-backend.txt
```

## Typical Execution Flow

### 1) Preprocess datasets

```powershell
python scripts/preprocess_all_datasets.py --source legacy
```

### 2) Train artifacts

```powershell
python scripts/train_models.py --data-source preprocessed
python scripts/build_risk_profiles.py
python scripts/benchmark_baseline_models.py
```

### 3) Validate backend

```powershell
python scripts/backend_smoke_test.py
```

### 4) Run backend server

```powershell
uvicorn backend.app:app --reload
```

### 5) Run terminal prediction

```powershell
python scripts/predict_from_terminal.py --guided
```

## API Endpoints (Current)

- `GET /health`
- `POST /predict-risk`
- `POST /recommend-plan`
- `POST /log-meal`
- `POST /risk-report`

## Model and Feature Contract

Serving relies on feature order declared in:

- `artifacts/ckd_features.json`
- `artifacts/hypertension_features.json`
- `artifacts/diabetes_features.json`

Risk thresholds and top factors are declared in:

- `artifacts/risk_thresholds_and_factors.json`

Any raw-input mapping layer must transform user-friendly payloads into these exact model feature schemas in the same column order.

## Known Limitations and Assumptions

- Current serving path assumes model-ready feature payloads; raw clinical input mapping is the next implementation phase.
- Some historical large/survey-style datasets may not align with ideal clinical feature availability. This is documented in `docs/current/PANEL_PROGRESS_LOG.md`.
- Nutrition outputs are recommendation support, not medical diagnosis or treatment.

## Documentation Pointers

- `docs/current/PROJECT_RUNBOOK.md` - operational run steps.
- `docs/current/PANEL_PROGRESS_LOG.md` - current decisions, caveats, and next priorities.
- `docs/current/RAW_INPUT_PREDICTION_GUIDE.md` - raw-input workflow notes (as implemented/updated).
- `docs/current/TERMINAL_DEMO_SAMPLES.md` - terminal usage/demo examples.

## ESA Template Outputs

Generated Phase-2 assets are available under:

- `outputs/esa_phase2/`

These are generated from institute templates in `esa templates/` and can be regenerated or edited directly before submission.
