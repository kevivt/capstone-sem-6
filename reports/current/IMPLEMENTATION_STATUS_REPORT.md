# Implementation Status Report

Date: 2026-04-19
Project: Personalized Diet Recommendation and Risk Prediction

## What Was Completed

- Extracted the ISA-1 slide heading sequence from the legacy review PDF and rebuilt a new presentation using the same headings.
- Updated the review material to include dataset row counts, column counts, exact column names, source-site names, terminal screenshot commands, and a project-vs-baseline comparison graph.
- Fixed terminal prediction by changing model loading to lazy-load only the requested disease artifact.
- Replaced the oversized saved artifacts with lighter final models so terminal prediction is practical again.
- Updated demo terminal JSON payloads to match the actual deployed feature schemas.
- Retrained project-final models on the full prepared train splits instead of the previous `120000` cap.

## Final Models In Use

| Disease | Final model | Full train rows used | Test rows | Macro F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| CKD | DecisionTreeClassifier | 690956 | 91118 | 0.518282 | 0.591821 |
| Hypertension | LogisticRegression | 785242 | 100000 | 0.540125 | 0.926586 |
| Diabetes | GaussianNB | 754970 | 100000 | 0.525395 | 0.898102 |

Source: `training_results_summary.csv`

## Dataset Summary

| Disease | Source file | Rows | Columns | Target |
| --- | --- | ---: | ---: | --- |
| CKD | `medical datasets/large/ckd_large.csv` | 455590 | 10 | `ckd_label` |
| Hypertension | `medical datasets/large/hypertension_large_500k.csv` | 500000 | 13 | `prevalentHyp` |
| Diabetes | `medical datasets/large/diabetes_large_500k.csv` | 500000 | 10 | `diabetes` |

### CKD columns
`age, male, BMI, sysBP, diaBP, glucose, diabetes, prevalentHyp, source, ckd_label`

### Hypertension columns
`male, age, education, currentSmoker, cigsPerDay, prevalentStroke, BMI, sysBP, diaBP, glucose, diabetes, source, prevalentHyp`

### Diabetes columns
`age, male, BMI, sysBP, diaBP, glucose, prevalentHyp, currentSmoker, source, diabetes`

## Source Sites Used Across The Project Amalgamation

- UCI Machine Learning Repository
- Framingham Heart Study
- Pima Indians Diabetes Dataset
- CDC BRFSS Annual Data
- CDC NHANES 2017-2018
- ICMR-NIN IFCT 2017
- Mendeley Data
- Open Food Facts India
- Kaggle

## Baseline Comparison Note

- `reports/current/BASELINE_MODEL_COMPARISON.md` was regenerated.
- Baseline benchmarking still uses the built-in `120000` training-row cap per disease for runtime control.
- The saved project-final artifacts do **not** use that cap; they use the full train splits shown above.

## Files Added Or Updated

- `backend/model_registry.py`
- `scripts/train_models.py`
- `scripts/build_risk_profiles.py`
- `scripts/benchmark_baseline_models.py`
- `demo_inputs/terminal_samples/ckd_sample_1.json`
- `demo_inputs/terminal_samples/hypertension_sample_1.json`
- `demo_inputs/terminal_samples/diabetes_sample_1.json`
- `outputs/isa-review2/DATASET_SUMMARY_FOR_SLIDES.md`
- `outputs/isa-review2/TERMINAL_SCREENSHOT_COMMANDS.md`
- `reports/current/BASELINE_MODEL_COMPARISON.md`
- `reports/current/baseline_model_comparison.csv`
- `reports/current/RISK_THRESHOLDS_AND_FACTORS_REPORT.md`
- `reports/current/IMPLEMENTATION_STATUS_REPORT.md`
- `outputs/isa-review2/isa_review2_updated_apr19.pptx`

## Safe Cleanup Completed

- Removed the obsolete `scripts/__pycache__` directory if present.
- Old oversized model artifacts were replaced by the current final artifacts during retraining.

## Next Step

- Continue implementation from the current working artifact set by building the raw-input mapping layer, then reconnect those risk outputs into the diet recommendation logic.
