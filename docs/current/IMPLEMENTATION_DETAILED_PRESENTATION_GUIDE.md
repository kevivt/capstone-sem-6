# Implementation Data and Model Summary for Panel

Date: 2026-04-13
Project: Personalized Diet Recommendation and Risk Prediction

## 1. What This File Contains

This is the corrected implementation summary with only the core points requested:

- Dataset names
- Rows before and after preprocessing/splitting
- Baseline model accuracy
- Finalized serving models

## 2. Dataset Details (Names + Rows Before and After)

Primary disease datasets used:

- CKD: Chronic Kidney Disease dataset (UCI)
- Hypertension: Framingham Heart Study dataset
- Diabetes: Pima Indians Diabetes dataset

Large-data combination used for expansion pipeline:

- BRFSS (Behavioral Risk Factor Surveillance System)
- NHANES (National Health and Nutrition Examination Survey)

Combined outputs produced from BRFSS + NHANES:

- CKD large table
- Hypertension large table
- Diabetes large table

Nutrition recommendation dataset:

- Unified Food Knowledge Base (combined nutrition dataset)

Generated files (after preprocessing and split):

- preprocessed_outputs/ckd_preprocessed.csv
- preprocessed_outputs/ckd_train_smote.csv
- preprocessed_outputs/ckd_test.csv
- preprocessed_outputs/hypertension_preprocessed.csv
- preprocessed_outputs/hypertension_train_smote.csv
- preprocessed_outputs/hypertension_test.csv
- preprocessed_outputs/diabetes_preprocessed.csv
- preprocessed_outputs/diabetes_train_smote.csv
- preprocessed_outputs/diabetes_test.csv

### 2.1 Row Count Table

| Dataset | Before (raw large CSV) | After preprocessing | Train (SMOTE) | Test |
| --- | ---: | ---: | ---: | ---: |
| CKD | 455590 | 455590 | 690956 | 91118 |
| Hypertension | 5824 | 5824 | 5930 | 1165 |
| Diabetes | 74822 | 74822 | 106636 | 14965 |

Notes:

- Raw and preprocessed rows can match because preprocessing here focuses on cleaning, encoding, and feature transformation.
- Train rows can increase after SMOTE balancing.

## 3. Baseline Model Accuracy

Source: reports/current/BASELINE_MODEL_COMPARISON.md

### 3.1 Best Baseline Per Dataset

| Dataset | Baseline model | Train rows | Test rows | Accuracy | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CKD | decision_tree | 120000 | 91118 | 0.753035 | 0.495961 | 0.585231 |
| Hypertension | logistic_regression | 5930 | 1165 | 0.727897 | 0.716861 | 0.807064 |
| Diabetes | gaussian_nb | 106636 | 14965 | 0.943201 | 0.831251 | 0.942183 |

## 4. Model Finalization (Serving Models)

Finalized serving models are the ones stored as runtime artifacts and used by the backend.

Source files:

- training_results_summary.csv
- reports/current/PROJECT_VS_BASELINE.md
- artifacts/*.joblib

### 4.1 Final Models Used in Backend

| Dataset | Finalized project model | Project Macro F1 | Project ROC-AUC |
| --- | --- | ---: | ---: |
| CKD | CKD_KNN_n1_distance_kdtree | 0.514303 | 0.528565 |
| Hypertension | HTN_SVM_rbf_C4 | 0.721738 | 0.802784 |
| Diabetes | DIAB_RF_n120_d4_leaf125 | 0.820006 | 0.948276 |

### 4.2 Finalization Logic (Short)

- Candidate models were compared disease-wise.
- Selection prioritized macro-level class balance quality and ranking quality (Macro F1 + ROC-AUC), then deployed with fixed artifacts.
- Final artifact files:
  - artifacts/ckd_model.joblib
  - artifacts/hypertension_model.joblib
  - artifacts/diabetes_model.joblib

## 5. One-Line Viva Description

This implementation takes CKD, hypertension, and diabetes inputs, predicts disease risk with finalized disease-specific models, and converts risk outputs into actionable nutrition guidance through backend rules and food recommendation logic.
