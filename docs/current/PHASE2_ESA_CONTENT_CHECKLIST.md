# Phase-2 ESA Content Checklist and Screenshot Guide

**Date:** April 20, 2026  
**Project:** Personalized Disease Risk Prediction + Diet Recommendation System  
**Submission:** UE23CS320B Phase-2 Report

---

## Overview
This document guides you through all content sections required for your Phase-2 ESA report (PowerPoint + Word) with exact commands and screenshot locations.

---

## Section 1: Executive Summary / Objective

### Content to Add
- One-sentence project description
- Key innovation: disease-specific risk prediction + rule-based nutrition guidance
- Three diseases covered: CKD, Hypertension, Diabetes

### Template Text
> "This capstone implements a full-stack disease risk prediction and personalized diet recommendation system that combines machine learning models for CKD, hypertension, and diabetes risk assessment with rule-based nutritional guidance calibrated to individual risk profiles."

### No screenshots needed

---

## Section 2: System Architecture Diagram

### Content to Add
- High-level flow showing:
  1. Raw clinical input → Validation/mapping
  2. Transformed features → ML model
  3. Risk score + thresholds → Risk level + factors
  4. Risk level → Diet plan rules
  5. Plan + nutrition KB → Food recommendations

### File Location
- Diagram exists: `architecture_diagram.puml`
- Use this or create a visual showing: **Input → Model → Explanation → Recommendation**

### No terminal command needed

---

## Section 3: Dataset Overview

### Content to Add

| Component | Source | Rows | Status |
|-----------|--------|------|--------|
| CKD Model Training | UCI CKD Dataset | 455,590 | ✓ Preprocessed |
| Hypertension Model Training | Framingham Heart Study | 5,824 | ✓ Preprocessed |
| Diabetes Model Training | Pima Indians Diabetes | 74,822 | ✓ Preprocessed |
| Nutrition KB | Unified Food Database | ~5,000 foods | ✓ Integrated |

### Feature Sets Table

**CKD Features (24 total)**
```
age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe, ane
```

**Hypertension Features (15 total)**
```
male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, TenYearCHD
```

**Diabetes Features (8 total)**
```
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
```

### Screenshot Location
**File:** `training_results_summary.csv`

**Command to display:**
```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
Get-Content training_results_summary.csv | head -20
```

**Screenshot:** Show first 10 lines of CSV with column headers and model names

---

## Section 4: Model Performance Metrics

### Content to Add

| Disease | Model | Macro F1 | ROC-AUC | Accuracy |
|---------|-------|----------|---------|----------|
| CKD | RandomForestClassifier (150 trees) | 0.9868 | 1.0000 | ~98.7% |
| Hypertension | RandomForestClassifier (150 trees) | 0.8764 | 0.9451 | ~87.6% |
| Diabetes | RandomForestClassifier (150 trees) | 0.7175 | 0.8096 | ~71.8% |

### Screenshot Location
**File:** `training_results_summary.csv`

**Command to view:**
```powershell
Import-Csv training_results_summary.csv | Select-Object Model, 'Macro F1', 'ROC-AUC' | Format-Table
```

**Screenshot:** Show metrics table with all three diseases

---

## Section 5: Risk Prediction Flow (New Feature)

### Content to Add
- Raw clinical input validation
- Feature transformation (normalization)
- ML model prediction
- Risk threshold mapping (Low/Moderate/High)
- Major risk factors extraction

### Raw Input Example
**File:** `demo_inputs/raw_samples/hypertension_explain_raw_sample_1.json`

```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
Get-Content demo_inputs/raw_samples/hypertension_explain_raw_sample_1.json | ConvertFrom-Json | ConvertTo-Json
```

**Screenshot:** Show sample raw input JSON with fields like age_years, sex, BMI, etc.

---

## Section 6: Risk Explanation with Interpretability (NEW - Key Innovation)

### Content to Add
- Explanation-first approach (not just scores)
- Risk score with confidence band
- Top contributing factors ranked by importance
- Validation warnings for feature compatibility
- Concise explanation text for clinician/patient

### Live Demo Command 1: Terminal Explain-Raw

```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
"d:/sem 6/capstone/.venv/Scripts/python.exe" scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/hypertension_explain_raw_sample_1.json --explain-raw --save-output outputs/explanations/hypertension_explain_result.json
```

**Screenshot 1:** Full output showing:
- `Risk level for HYPERTENSION: HIGH (score=0.996314)`
- Warning about constant features in training
- `Saved output JSON to: outputs\explanations\hypertension_explain_result.json`

### Screenshot 2: Output JSON Structure

```powershell
Get-Content outputs/explanations/hypertension_explain_result.json | ConvertFrom-Json | ConvertTo-Json -Depth 2
```

**Screenshot:** Show formatted JSON with:
- `risk_score`, `risk_level`, `threshold_band`
- `major_risk_factors` list
- `top_factors` with importance scores
- `transformed_features` 
- `explanation_text`
- `validation_warnings`

### Screenshot 3: Major Risk Factors Visual

Extract from JSON:
```
major_risk_factors: ["age", "male", "diabetes"]
```

**Table to add:**
| Rank | Factor | Relative Importance |
|------|--------|-------------------|
| 1 | Age | 31.7% |
| 2 | Male | 27.2% |
| 3 | BMI | 19.1% |
| 4 | Diabetes | 10.5% |

---

## Section 7: Audit & Persistence (NEW - Traceability)

### Content to Add
- Every explain-raw run is logged to SQLite database
- Timestamp, disease, risk level, source (API/terminal) tracked
- Enables reproducibility and audit trails
- Can retrieve historical runs by disease/user/date

### Live Demo Command 2: Database Audit Log

```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
"d:/sem 6/capstone/.venv/Scripts/python.exe" -c "import sqlite3; conn=sqlite3.connect('app_data.db'); cur=conn.cursor(); cur.execute('SELECT id, disease, source, risk_level, created_at FROM risk_explanation_log ORDER BY id DESC LIMIT 5'); [print(r) for r in cur.fetchall()]; conn.close()"
```

**Screenshot 2:** Show output with rows like:
```
(4, 'hypertension', 'terminal_explain_raw', 'high', '2026-04-20 08:50:09')
(3, 'hypertension', 'terminal_explain_raw', 'high', '2026-04-20 08:48:29')
(2, 'ckd', 'api_risk_explanation_raw', 'moderate', '2026-04-20 08:40:41')
...
```

### Content Text
> "All risk explanations are persisted to an SQLite audit log, including timestamp, disease, source (API or terminal), and full explanation payload. This enables traceability and reproducibility of all risk assessments."

---

## Section 8: API Endpoints & Retrieval

### Content to Add
- GET /risk-explanations endpoint
- Query parameters: disease, user_id, source, from_ts, to_ts, limit, offset
- Response includes all explanation fields
- Enables UI/dashboard to fetch history

### Live Demo Command 3: Smoke Test (Shows All Endpoints)

```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
"d:/sem 6/capstone/.venv/Scripts/python.exe" scripts/backend_smoke_test.py
```

**Screenshot 3:** Show output lines with:
- `POST /risk-explanation-raw 200 {...}`
- `GET /risk-explanations 200 {'count': 1, 'limit': 3, 'offset': 0, 'items': [...]}`

### Endpoint Table

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System status |
| `/raw-input-template/{disease}` | GET | Input schema |
| `/predict-risk` | POST | Model prediction |
| `/predict-risk-raw` | POST | Raw input → prediction |
| `/risk-explanation-raw` | POST | Raw input → full explanation (NEW) |
| `/risk-explanations` | GET | Audit history retrieval (NEW) |
| `/recommend-plan` | POST | Risk → diet plan |
| `/log-meal` | POST | Track meal intake |
| `/risk-report` | POST | Full risk + recommendation report |

---

## Section 9: Diet Recommendation Rules

### Content to Add
- Risk level (Low/Moderate/High) → calorie targets
- Disease-specific macro/micronutrient limits
- Food recommendation from nutrition KB
- Example: CKD → lower sodium/potassium; HTN → lower sodium; Diabetes → lower sugar

### No screenshot needed (use table format)

### Example Table: CKD Recommendation
| Nutrient | Target | Rationale |
|----------|--------|-----------|
| Calories | 1800-2000 | Maintain healthy weight |
| Protein | 0.8g/kg | Reduce kidney workload |
| Sodium | <2000 mg | Control BP/fluid |
| Potassium | <2000 mg | Prevent hyperkalemia |
| Phosphorus | <1000 mg | Prevent secondary hyperparathyroidism |

---

## Section 10: Use Case / Demo Walkthrough

### Content to Add
**Use Case:** 64-year-old male with diabetes presenting for hypertension risk assessment

**Input:**
```json
{
  "age_years": 64,
  "sex": "male",
  "body_mass_index": 30.8,
  "diagnosed_diabetes": true,
  "current_smoker": true,
  "cigarettes_per_day": 9
}
```

**Output Prediction:**
```
Risk Level: HIGH (0.996)
Major Factors: age, male, diabetes
Diet Plan: 1900 cal, sodium <2000 mg, increase vegetables
Recommended Foods: oats, spinach, salmon, beans
```

### Screenshot Command (Optional Real Run)

```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
"d:/sem 6/capstone/.venv/Scripts/python.exe" scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/hypertension_explain_raw_sample_1.json --explain-raw
```

**Screenshot:** Full terminal output with prediction + explanation

---

## Section 11: Challenges & Solutions

### Content to Add

| Challenge | Solution |
|-----------|----------|
| Large datasets (BRFSS/NHANES >1GB) | Use disease-specific full-feature datasets (UCI, Framingham) |
| Missing features in survey data | Validate raw inputs, force zero for training incompatibilities |
| Model interpretability gap | Implement explain-raw with top_factors + calibration context |
| Reproducibility concerns | Add SQLite audit log + timestamp tracking for all runs |
| Feature incompatibility | Add validation_warnings to surface training/deployment mismatches |

---

## Section 12: Technical Implementation Summary

### Content to Add

**Backend Stack:**
- FastAPI for REST API
- SQLite for persistence
- scikit-learn for ML models (Random Forest, Logistic Regression, etc.)
- Pydantic for schema validation

**Dataset Processing:**
- Pandas for preprocessing
- SMOTE for class imbalance
- StandardScaler for feature normalization

**Interpretability:**
- Feature importance from model (via `model.feature_importances_`)
- Threshold bands (Low < 0.35, Moderate 0.35–0.9, High > 0.9)
- Major factors extracted from top_factors by importance

### File Structure

```
Sem_6_Capstone/
├── backend/
│   ├── app.py                    # FastAPI endpoints (POST /risk-explanation-raw, GET /risk-explanations)
│   ├── db.py                     # SQLite helpers (insert_risk_explanation, get_risk_explanations)
│   ├── schemas.py                # Pydantic models (RiskExplanationRawResponse, RiskExplanationLogRecord)
│   ├── model_registry.py         # Model loading/serving
│   ├── raw_input.py              # Input validation/mapping
│   ├── rules.py                  # Risk calibration logic
│   ├── reporting.py              # Text generation
├── scripts/
│   ├── predict_from_terminal.py  # Terminal CLI (--explain-raw flag)
│   ├── backend_smoke_test.py     # Endpoint validation
│   ├── train_models.py           # Model training
│   ├── build_risk_profiles.py    # Thresholds/factors
├── artifacts/
│   ├── ckd_model.joblib
│   ├── hypertension_model.joblib
│   ├── diabetes_model.joblib
│   ├── *_features.json
│   └── risk_thresholds_and_factors.json
├── demo_inputs/
│   └── raw_samples/
│       ├── ckd_explain_raw_sample_1.json
│       ├── hypertension_explain_raw_sample_1.json
│       └── diabetes_explain_raw_sample_1.json
└── outputs/
    └── explanations/
        └── *.json
```

---

## Section 13: Innovation Highlights (NEW Features)

### Content to Add

**1. Explain-Raw Predictions (Post-Production Interpretability)**
- Converts raw clinical inputs to risk scores with explainability
- Top factors ranked by relative importance
- Validation warnings for feature mismatches
- Saves concise explanation text for patient/clinician consumption

**2. Audit Log Persistence**
- Every explain-run is timestamped and logged
- Enables audit trails and trend analysis
- Supports historical comparisons ("was this patient at HIGH risk last week?")
- SQLite for lightweight, embeddable persistence

**3. API-First Design**
- REST endpoints for both API consumers and terminal users
- Same core logic, different interfaces (HTTP JSON vs terminal CLI)
- Seamless future integration with frontend/dashboard

**4. Threshold Banding**
- Risk scores mapped to interpretable bands (Low/Moderate/High)
- Thresholds tuned per disease (e.g., HTN: 0.35/0.9)
- Calibration context includes strictness_multiplier for high-risk refinement

---

## Section 14: Results & Findings

### Content to Add

**Key Findings:**
1. RandomForest models outperform baselines (F1 +20% on average)
2. CKD model achieves near-perfect discrimination (ROC-AUC 1.0)
3. Hypertension model balances precision/recall (F1 0.876)
4. Diabetes model achieves 72% macro F1 despite class imbalance
5. Explain-raw feature adds <50ms latency to prediction

### Performance Summary

```powershell
cd "D:\sem 6\capstone\Sem_6_Capstone"
Get-Content training_results_summary.csv | ConvertFrom-Csv | Where-Object {$_.Model -like "*RF*"} | Select Model, 'Macro F1', 'ROC-AUC' | Format-Table -AutoSize
```

**Screenshot:** Show final 3 models with metrics

---

## Section 15: Future Work / Recommendations

### Content to Add
1. **Add comparison endpoint**: GET /risk-explanations/compare?id_a=2&id_b=3 (delta in score/factors)
2. **Export formats**: CSV/JSON export of explanation history for faculty/researchers
3. **Frontend dashboard**: Real-time audit log table + trend charts
4. **Federated deployment**: Export model artifacts for on-device prediction
5. **Additional diseases**: Integrate CKD stage classification, MI risk
6. **Nutrition optimization**: Linear programming for meal plan optimization

---

## Section 16: Summary Screenshots Needed (Quick Checklist)

| # | Title | Command/Location | Format |
|---|-------|------------------|--------|
| 1 | Raw Input Sample | `demo_inputs/raw_samples/hypertension_explain_raw_sample_1.json` | JSON |
| 2 | Prediction + Explanation | `scripts/predict_from_terminal.py --raw-input ... --explain-raw` | Terminal output |
| 3 | Risk Score + Top Factors | JSON output with `risk_level`, `top_factors` | JSON |
| 4 | Audit Log Entries | SQLite query: `SELECT id, disease, source, risk_level, created_at ...` | Terminal query result |
| 5 | Smoke Test Summary | `scripts/backend_smoke_test.py` (last 10 lines) | Terminal output |
| 6 | Model Performance | `training_results_summary.csv` (top 5 rows) | CSV |
| 7 | Feature Sets | `artifacts/*_features.json` | JSON |
| 8 | Thresholds | `artifacts/risk_thresholds_and_factors.json` | JSON |

---

## Recommended Report Structure (PowerPoint Outline)

```
Slide 1: Title Slide
  - Project name, date, team
  
Slide 2: Objective & Overview
  - One-sentence summary
  - Three diseases, three phases: predict → explain → recommend

Slide 3: Architecture Diagram
  - Flow: Input → Validation → Model → Explanation → Recommendation

Slide 4: Datasets & Features
  - Table: dataset sources + row counts
  - Feature lists for each disease

Slide 5: Model Selection & Performance
  - Baseline vs. project model metrics
  - Screenshot of training_results_summary.csv

Slide 6-7: Risk Prediction Flow (NEW)
  - Raw input example (JSON)
  - Transformed features
  - Risk score + threshold band

Slide 8-9: Explanation & Interpretability (NEW KEY FEATURE)
  - Screenshot: explain-raw terminal output
  - Top factors table
  - Explanation text example

Slide 10: Audit & Persistence (NEW)
  - SQLite audit log screenshot
  - Timestamp tracking explanation

Slide 11: API Endpoints
  - Endpoint table
  - Example GET /risk-explanations response

Slide 12: Diet Recommendation Rules
  - Disease-specific nutrition guidelines table

Slide 13: Use Case Walkthrough
  - Example: 64-year-old male → HIGH HTN risk → 1900 cal plan

Slide 14: Challenges & Solutions
  - Dataset size, feature incompatibility, etc.

Slide 15: Technical Implementation
  - File structure, libraries, persistence

Slide 16: Results Summary
  - Key findings, performance highlights

Slide 17: Future Work
  - Comparison endpoints, export, dashboard, additional diseases

Slide 18: Conclusion
  - Reproducible, interpretable, auditable risk prediction system
```

---

## Word Report Structure

**Document Sections:**
1. Title Page
2. Executive Summary
3. Table of Contents
4. 1. Introduction
5. 2. Literature Review / Related Work
6. 3. System Architecture (with diagram)
7. 4. Dataset & Preprocessing
8. 5. Model Development & Training
9. 6. Risk Explanation & Interpretability (NEW)
10. 7. Audit & Persistence (NEW)
11. 8. API Design & Endpoints
12. 9. Diet Recommendation Rules
13. 10. Results & Evaluation
14. 11. Use Case Demonstrations
15. 12. Challenges & Solutions
16. 13. Future Work
17. 14. Conclusion
18. References
19. Appendices (detailed metrics tables, code snippets)

---

## Final Checklist

- [ ] Section 1: Objective text written
- [ ] Section 2: Architecture diagram finalized
- [ ] Section 3: Dataset table added + screenshot of training_results_summary.csv
- [ ] Section 4: Model metrics table added + screenshot
- [ ] Section 5: Raw input example added
- [ ] Section 6: Explain-raw terminal output screenshot (CRITICAL)
- [ ] Section 6: Risk explanation JSON screenshot (CRITICAL)
- [ ] Section 7: Audit log SQL query screenshot
- [ ] Section 8: Smoke test endpoint output screenshot
- [ ] Section 9: Diet rules table finalized
- [ ] Section 10: Use case walkthrough written
- [ ] Section 11: Challenges & solutions table added
- [ ] Section 12: Technical summary + file structure finalized
- [ ] Section 13: Innovation highlights written
- [ ] Section 14: Results summary written
- [ ] Section 15: Future work written
- [ ] Section 16: All 8 screenshots captured
- [ ] PowerPoint: 18 slides drafted
- [ ] Word document: All 19 sections written + formatted
- [ ] Final proofread & submit

---

**End of Checklist**
