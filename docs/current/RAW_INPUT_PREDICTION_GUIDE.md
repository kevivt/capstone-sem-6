# Raw Input Prediction Guide

Date: 2026-04-20

## What This Adds

This phase adds a patient-friendly raw-input prediction flow on top of the already working model-artifact flow.

The repository now supports:
- model-ready feature prediction
- raw patient-input prediction
- raw patient-input risk reporting with diet calibration
- guided raw terminal entry

## New Backend Endpoints

### `GET /raw-input-template/{disease}`
Returns a compact example raw payload template for:
- `ckd`
- `hypertension`
- `diabetes`

### `POST /predict-risk-raw`
Accepts:
- `disease`
- `raw_inputs`
- optional `user_id`

Returns:
- predicted class
- risk score
- risk level
- model name
- transformed internal feature vector
- validation warnings

### `POST /risk-report-raw`
Accepts:
- `disease`
- `raw_inputs`
- optional `profile`
- optional `user_id`
- `top_n_foods`

Returns:
- full disease risk report
- transformed internal feature vector
- validation warnings
- disease-aware diet plan
- recommended foods

## Terminal Usage

### Show raw template

```powershell
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --disease ckd --show-raw-template
```

### Predict from raw sample JSON

```powershell
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --raw-input demo_inputs\raw_samples\ckd_raw_sample_1.json
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --raw-input demo_inputs\raw_samples\hypertension_raw_sample_1.json
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --raw-input demo_inputs\raw_samples\diabetes_raw_sample_1.json
```

### Guided raw terminal mode

```powershell
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --guided-raw
```

## Important Assumption / Limitation

The currently deployed models were trained on reduced survey-style large tables.

Because of that, some columns in the saved artifact feature sets were constant zeros during training:

- CKD:
  - `sysBP`
  - `diaBP`
  - `glucose`
  - `prevalentHyp`
  - `source`

- Hypertension:
  - `sysBP`
  - `diaBP`
  - `glucose`
  - `source`

- Diabetes:
  - `sysBP`
  - `diaBP`
  - `glucose`
  - `source`

For model compatibility, the raw-input mapping layer forces those fields to `0.0` when producing the transformed feature vector.

This means:
- raw patient BP/glucose values are accepted for intake completeness
- but some of them are not yet meaningfully used by the deployed models
- the API and terminal flow now return warnings to make this explicit

## Why This Design Was Chosen

- It preserves the current trained artifacts and working backend.
- It adds a user-facing input layer without breaking the current model-ready path.
- It documents limitations honestly instead of pretending the deployed models consume richer clinical inputs than they actually do.

## Files Added / Updated

- `backend/raw_input.py`
- `backend/schemas.py`
- `backend/app.py`
- `backend/rules.py`
- `backend/reporting.py`
- `scripts/predict_from_terminal.py`
- `scripts/backend_smoke_test.py`
- `demo_inputs/raw_samples/*.json`

## Demo Flow

1. Run `scripts/backend_smoke_test.py`
2. Run `scripts/predict_from_terminal.py --disease ckd --show-raw-template`
3. Run one or more `--raw-input` demo files
4. Call `/risk-report-raw` to show transformed features, warnings, and diet-plan output together
