# Panel Progress Log

This file is the current technical handoff for the disease-risk modeling stack.

## Last Updated
- April 19, 2026

## Current Decision
- We are no longer using the BRFSS-heavy `large_500k` feature tables for active model serving.
- Audit showed those tables were not viable for lab-based disease risk prediction because several key columns became all-missing and then constant after preprocessing.
- Active serving now uses disease-specific full-feature datasets:
  - CKD: UCI Chronic Kidney Disease dataset via the existing fallback in `preprocess_all_datasets.py`
  - Hypertension: `medical datasets/raw/hypertension/framingham_heart_study.csv`
  - Diabetes: `medical datasets/raw/diabetes/diabetes.csv`

## Why We Switched
- `large_ckd.csv`: `sysBP`, `diaBP`, `glucose`, and `prevalentHyp` were entirely missing.
- `hypertension_large_500k.csv`: `sysBP`, `diaBP`, and `glucose` were entirely missing.
- `diabetes_large_500k.csv`: `sysBP`, `diaBP`, and `glucose` were entirely missing.
- Those columns were being imputed/scaled into constant zeros, so the earlier models were effectively learning from reduced-survey features rather than the intended clinical inputs.

## Active Preprocessing Command

```powershell
python scripts/preprocess_all_datasets.py --source legacy
```

## Active Training Command

```powershell
python scripts/train_models.py --data-source preprocessed
python scripts/build_risk_profiles.py
python scripts/benchmark_baseline_models.py
```

## Current Serving Models
- CKD: `RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)`
- Hypertension: `RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)`
- Diabetes: `RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)`

## Current Metrics Snapshot
Source: `training_results_summary.csv`

| Model | Macro F1 | ROC AUC |
| --- | --- | --- |
| CKD_RF_150 | 0.986753 | 1.000000 |
| HTN_RF_150 | 0.876365 | 0.945124 |
| DIAB_RF_150 | 0.717456 | 0.809630 |

## Active Feature Sets

### CKD
- `age`
- `bp`
- `sg`
- `al`
- `su`
- `rbc`
- `pc`
- `pcc`
- `ba`
- `bgr`
- `bu`
- `sc`
- `sod`
- `pot`
- `hemo`
- `pcv`
- `wbcc`
- `rbcc`
- `htn`
- `dm`
- `cad`
- `appet`
- `pe`
- `ane`

### Hypertension
- `male`
- `age`
- `education`
- `currentSmoker`
- `cigsPerDay`
- `BPMeds`
- `prevalentStroke`
- `diabetes`
- `totChol`
- `sysBP`
- `diaBP`
- `BMI`
- `heartRate`
- `glucose`
- `TenYearCHD`

### Diabetes
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

## Verification Completed
- `python -m py_compile scripts/train_models.py scripts/build_risk_profiles.py scripts/benchmark_baseline_models.py scripts/predict_from_terminal.py`
- `python scripts/backend_smoke_test.py`

## Current Caveats
- Terminal `--interactive` mode is now the correct path for entering all model columns.
- The earlier `--raw-interactive` bridge was removed because it only matched the old reduced feature artifacts.
- Diabetes still has the smallest dataset of the three, so it is the most likely candidate for future expansion if we later decide to download more data.

## Next Priority
- Build a proper disease-specific raw user input layer on top of these full clinical feature sets.
- After that, connect the new risk scores directly into nutrition and diet calibration.
