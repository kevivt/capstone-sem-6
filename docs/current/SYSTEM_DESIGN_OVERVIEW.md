# High-Level System Design Overview

## Personalized Disease Risk Prediction + Diet Recommendation System

---

## 1. Logical User Groups

### 1.1 **Patients / End Users**
- **Role:** Primary users of the system
- **Interactions:** 
  - Input medical/health data (age, BMI, blood pressure, glucose levels, etc.)
  - Receive risk assessments (LOW, MODERATE, HIGH)
  - View personalized diet recommendations
  - Log meals and daily activities
  - Receive alerts for high-risk conditions
- **Access Method:** Mobile/Web interface or terminal CLI
- **Data Access:** Own health data, personalized reports

### 1.2 **Clinicians / Healthcare Providers**
- **Role:** Monitor patients and provide clinical oversight
- **Interactions:**
  - View patient risk profiles and explanations
  - Monitor high-risk alerts in real-time
  - Review audit logs of patient assessments
  - Adjust risk thresholds and rules
  - Export patient reports for EHR integration
- **Access Method:** Secure REST API, dashboard
- **Data Access:** Assigned patients' full health records and audit trails

### 1.3 **System Administrators**
- **Role:** Maintain system integrity and performance
- **Interactions:**
  - Manage database (backups, migrations)
  - Monitor API performance and latency
  - Update ML models and thresholds
  - Configure system parameters
  - Manage user access and permissions
- **Access Method:** Direct database access, admin CLI
- **Data Access:** Full system audit logs

### 1.4 **Public Health Officials**
- **Role:** Population-level surveillance and epidemiology
- **Interactions:**
  - View aggregated disease prevalence statistics
  - Generate cohort reports (anonymized)
  - Track disease trends over time
  - Identify high-risk geographic regions
- **Access Method:** Aggregation API, reporting dashboards
- **Data Access:** Anonymized, aggregated data only

---

## 2. Application Components

### 2.1 **FastAPI Backend Server**
**Purpose:** Core application logic and REST API
- **Endpoints:**
  - `POST /predict-risk` – Standard risk prediction
  - `POST /predict-risk-raw` – Raw feature input with preprocessing
  - `POST /risk-explanation-raw` – Interpretable risk prediction with explanation
  - `GET /risk-explanations` – Retrieve historical predictions
  - `POST /recommend-plan` – Generate diet recommendations
  - `POST /log-meal` – Record meal consumption
  - `POST /risk-report` – Generate comprehensive risk report
  - `POST /risk-report-raw` – Report with raw input handling
  - `GET /health` – System health check
- **Technologies:** Python 3.13, FastAPI, Pydantic
- **Performance:** <50ms per prediction, lightweight
- **Deployment:** Runs on local server or cloud (on-premise for HIPAA compliance)

### 2.2 **Prediction Engine**
**Purpose:** Machine learning inference for disease risk assessment
- **Models:** RandomForestClassifier (150 trees each)
  - CKD (Chronic Kidney Disease): 10 input features
  - Hypertension: 13 input features
  - Diabetes: 10 input features
- **Performance Metrics:**
  - CKD: Macro F1=0.9868, ROC-AUC=1.0000
  - Hypertension: Macro F1=0.8764, ROC-AUC=0.9451
  - Diabetes: Macro F1=0.7175, ROC-AUC=0.8096
- **Output:** Risk score (0.0-1.0) + predicted class
- **Location:** `backend/model_registry.py`

### 2.3 **Explanation Engine**
**Purpose:** Provide interpretability and clinical insight
- **Functions:**
  - Extract top risk factors (ranked by importance)
  - Generate risk level bands (LOW <0.35, MODERATE 0.35-0.9, HIGH >0.9)
  - Create human-readable explanation text
  - Identify validation warnings (feature distribution mismatches)
  - Provide calibration context for threshold adjustment
- **Output:** `RiskExplanationRawResponse` with:
  - `risk_score` – Numeric prediction (0.0-1.0)
  - `risk_level` – Category (LOW/MODERATE/HIGH)
  - `threshold_band` – Specific threshold range
  - `top_factors` – Top 5 features ranked by importance
  - `validation_warnings` – Feature distribution alerts
  - `explanation_text` – Narrative explanation
  - `calibration_context` – Adjustment multipliers
- **Location:** `backend/reporting.py`

### 2.4 **Recommendation Engine**
**Purpose:** Generate personalized diet plans
- **Functions:**
  - Map disease type to dietary constraints
  - Generate nutrient targets (sodium, potassium, carbs, etc.)
  - Create meal recommendations
  - Adapt recommendations based on preferences
- **Rules:** Disease-specific guardrails (KDOQI, ADA, DASH guidelines)
- **Output:** `DietPlanResponse` with meal suggestions and macro targets
- **Location:** `backend/rules.py`

### 2.5 **Audit & Logging Service**
**Purpose:** Ensure reproducibility and compliance
- **Functions:**
  - Log all predictions with full payloads
  - Track data transformations (raw → preprocessed)
  - Record timestamps and data sources
  - Enable historical retrieval and comparison
- **Queries:** Filter by disease, user_id, source (API/terminal), date range
- **Location:** `backend/db.py` (`risk_explanation_log` table)

### 2.6 **Input Validation & Preprocessing**
**Purpose:** Ensure data quality and consistency
- **Functions:**
  - Normalize raw inputs to training distribution
  - Apply feature scaling (StandardScaler)
  - Handle missing values
  - Validate feature ranges
  - Provide detailed error messages
- **Output:** Preprocessed feature arrays ready for model input
- **Location:** `backend/raw_input.py`, `backend/schemas.py`

---

## 3. Data Components

### 3.1 **SQLite Database** (`app_data.db`)

#### Table 1: `risk_profile`
Stores patient risk assessments and outcomes
- Columns: id, user_id, disease, model_name, predicted_class, risk_score, created_at
- Purpose: Historical risk tracking

#### Table 2: `risk_explanation_log`
Audit log for all explain-raw predictions (NEW)
- Columns: id, user_id, disease, model_name, predicted_class, risk_score, risk_level, thresholds, threshold_band, major_risk_factors_json, validation_warnings_json, raw_inputs_json, transformed_features_json, top_factors_json, calibration_context_json, explanation_text, source ('api' or 'terminal_explain_raw'), created_at
- Purpose: Full reproducibility and compliance

#### Table 3: `diet_plan`
Stores diet recommendations
- Columns: id, user_id, disease, plan_type, created_at
- Purpose: Track recommended diets

#### Table 4: `meal_log`
Records user meal consumption
- Columns: id, user_id, meal_name, date_logged, created_at
- Purpose: Monitor adherence

#### Table 5: `alert`
Critical health alerts
- Columns: id, user_id, alert_type, severity, created_at
- Purpose: High-risk notifications

### 3.2 **ML Model Artifacts** (`artifacts/`)
- **CKD Model:** `ckd_model.joblib` (RandomForest, 150 trees)
- **Hypertension Model:** `hypertension_model.joblib` (RandomForest, 150 trees)
- **Diabetes Model:** `diabetes_model.joblib` (RandomForest, 150 trees)
- **Feature Lists:** `ckd_features.json`, `hypertension_features.json`, `diabetes_features.json`
- **Thresholds:** `risk_thresholds_and_factors.json` (risk bands, importance scores)

### 3.3 **Training Datasets** (`preprocessed_outputs/`)
**Total Records:** 1,455,590 across three diseases

| Dataset | File | Rows | Columns | Features |
|---------|------|------|---------|----------|
| **CKD** | `ckd_preprocessed.csv` | 455,590 | 10 | age, male, BMI, sysBP, diaBP, glucose, diabetes, prevalentHyp, source, ckd_label |
| **Hypertension** | `hypertension_preprocessed.csv` | 500,000 | 13 | male, age, education, currentSmoker, cigsPerDay, prevalentStroke, BMI, sysBP, diaBP, glucose, diabetes, source, prevalentHyp |
| **Diabetes** | `diabetes_preprocessed.csv` | 500,000 | 10 | age, male, BMI, sysBP, diaBP, glucose, prevalentHyp, currentSmoker, source, diabetes |

### 3.4 **Demo & Test Data** (`demo_inputs/`)
- Raw input examples for each disease
- Explain-raw sample inputs
- Terminal sample commands
- Pre-populated for validation and testing

### 3.5 **Outputs & Reports** (`outputs/`)
- Risk explanation JSON files (from explain-raw API)
- Report PDFs and exports
- Audit logs and audit trails
- Performance metrics and confusion matrices

---

## 4. Interfacing Systems

### 4.1 **EHR Integration Points** (Future)
- **Interface Standard:** HL7 FHIR (Fast Healthcare Interoperability Resources)
- **Data Exchange:**
  - Inbound: Patient demographics, vital signs, lab results
  - Outbound: Risk assessments, recommendations, alerts
- **Implementation:** REST API designed for FHIR compatibility
- **Timeline:** Phase 3 of commercialization roadmap

### 4.2 **File System Interfaces**
- **Input:** CSV, JSON raw input files
- **Output:** JSON risk explanations, CSV reports
- **Location:** `demo_inputs/`, `outputs/`, `preprocessed_outputs/`

### 4.3 **Terminal/CLI Interface**
- **Script:** `scripts/predict_from_terminal.py`
- **Purpose:** Standalone risk predictions for batch processing or scripted workflows
- **Usage:** `python predict_from_terminal.py --raw-input input.json --explain-raw`
- **Audit:** Logs to same SQLite database with source='terminal_explain_raw'

### 4.4 **Analytics & Monitoring Systems** (Future)
- **Data Source:** `risk_explanation_log` table
- **Purpose:** Track system performance, model drift, user engagement
- **Integration:** Direct database queries or REST API aggregation endpoint

---

## 5. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER GROUPS                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Patients │  │Clinicians│  │  Admins  │  │Public Health │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
└───────┼─────────────┼─────────────┼────────────────┼────────────┘
        │             │             │                │
        │ (Mobile/    │ (Secure     │ (Direct DB/    │ (Aggregation
        │  Web/CLI)   │  API)       │  Admin CLI)    │  API)
        │             │             │                │
┌───────▼─────────────▼─────────────▼────────────────▼────────────┐
│                    FASTAPI BACKEND SERVER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ REST API Endpoints                                       │  │
│  │ • POST /predict-risk-raw                                 │  │
│  │ • POST /risk-explanation-raw  ◄─── Main Endpoint        │  │
│  │ • GET  /risk-explanations      ◄─── Retrieval            │  │
│  │ • POST /recommend-plan         ◄─── Diet Recommendations │  │
│  │ • POST /log-meal                                         │  │
│  │ • POST /risk-report                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ▲                 │                  │                   │
│         │                 │                  │                   │
│  ┌──────┴──────┬──────────▼─────┬───────────▼──────┐            │
│  │ Input       │ Prediction     │ Explanation      │            │
│  │ Validation  │ Engine         │ Engine           │            │
│  │ & Preproc   │ (Models)       │ (Interpretab.)   │            │
│  │             │                │                  │            │
│  │ • Normalize │ • CKD          │ • Top Factors    │            │
│  │ • Scale     │ • Hypertension │ • Risk Bands     │            │
│  │ • Validate  │ • Diabetes     │ • Warnings       │            │
│  │             │ • F1/ROC-AUC   │ • Explanations   │            │
│  └──────┬──────┴──────┬─────────┴───────┬──────────┘            │
│         │             │                 │                        │
│         │    ┌────────▼────────┐        │                        │
│         │    │ Recommendation  │        │                        │
│         │    │ Engine          │        │                        │
│         │    │ (Rules-based)   │        │                        │
│         │    │                 │        │                        │
│         │    │ • KDOQI (CKD)   │        │                        │
│         │    │ • ADA (Diabetes)│        │                        │
│         │    │ • DASH (HTN)    │        │                        │
│         │    └────────┬────────┘        │                        │
│         │             │                 │                        │
│         └─────────────┼─────────────────┘                        │
│                       │ All predictions logged                    │
│                       ▼                                           │
│         ┌──────────────────────────────┐                        │
│         │  Audit & Logging Service     │                        │
│         │                              │                        │
│         │  • risk_explanation_log      │                        │
│         │  • Timestamp & source        │                        │
│         │  • Full payload persistence  │                        │
│         │  • Filtering & retrieval     │                        │
│         └──────────┬───────────────────┘                        │
└──────────────────────┼────────────────────────────────────────────┘
                       │
                       │ SQL queries
                       ▼
    ┌──────────────────────────────────────────┐
    │     SQLite Database (app_data.db)        │
    │  ┌──────────────────────────────────┐   │
    │  │ risk_explanation_log             │   │
    │  │ (AUDIT TRAIL - NEW)              │   │
    │  │ • id, user_id, disease           │   │
    │  │ • risk_score, risk_level         │   │
    │  │ • top_factors_json               │   │
    │  │ • validation_warnings_json       │   │
    │  │ • raw_inputs_json                │   │
    │  │ • explanation_text               │   │
    │  │ • source, created_at             │   │
    │  └──────────────────────────────────┘   │
    │  ┌──────────────────────────────────┐   │
    │  │ risk_profile                     │   │
    │  │ diet_plan                        │   │
    │  │ meal_log                         │   │
    │  │ alert                            │   │
    │  └──────────────────────────────────┘   │
    └──────────────────────────────────────────┘
                       ▲
                       │
                       │ Training data
                       │
    ┌──────────────────────────────────────────┐
    │   ML Model Artifacts (artifacts/)        │
    │  ┌──────────────────────────────────┐   │
    │  │ ckd_model.joblib                 │   │
    │  │ hypertension_model.joblib        │   │
    │  │ diabetes_model.joblib            │   │
    │  │ risk_thresholds_and_factors.json │   │
    │  │ {feature}_features.json (×3)     │   │
    │  └──────────────────────────────────┘   │
    └──────────────────────────────────────────┘
                       ▲
                       │ Training/validation
                       │
    ┌──────────────────────────────────────────┐
    │  Training Datasets (preprocessed_outputs)│
    │  ┌──────────────────────────────────┐   │
    │  │ ckd_preprocessed.csv             │   │
    │  │ (455,590 rows, 10 columns)       │   │
    │  │                                  │   │
    │  │ hypertension_preprocessed.csv    │   │
    │  │ (500,000 rows, 13 columns)       │   │
    │  │                                  │   │
    │  │ diabetes_preprocessed.csv        │   │
    │  │ (500,000 rows, 10 columns)       │   │
    │  │                                  │   │
    │  │ Total: 1,455,590 records         │   │
    │  └──────────────────────────────────┘   │
    └──────────────────────────────────────────┘
```

---

## 6. Data Flow: Explain-Raw Prediction

### Step 1: User Input (Terminal or API)
```
Input: JSON with raw patient features
{
  "age": 65,
  "male": 1,
  "BMI": 28.5,
  "sysBP": 145,
  "diaBP": 92,
  "glucose": 180,
  "diabetes": 1,
  "prevalentHyp": 1
}
```

### Step 2: Request Processing
- **Route:** FastAPI endpoint receives POST request
- **Validation:** Pydantic schema validates input structure
- **Preprocessing:** `raw_input.py` normalizes and scales features

### Step 3: Prediction
- **Model Selection:** Load appropriate RandomForest model (CKD/HTN/DM)
- **Inference:** Predict risk score (0.0-1.0) and class
- **Performance:** <50ms latency

### Step 4: Explanation Generation
- **Risk Banding:** Map score to LOW/MODERATE/HIGH
- **Top Factors:** Extract top 5 features by importance
- **Warnings:** Check for feature distribution mismatches
- **Text Generation:** Create human-readable explanation
- **Calibration:** Provide threshold adjustment context

### Step 5: Audit Logging
- **Persistence:** Save full explanation to `risk_explanation_log`
- **Payload:** Include raw inputs, preprocessed features, explanation text
- **Timestamp:** Record exact time and source (API or terminal)
- **Retrieval:** Enable historical query via GET /risk-explanations

### Step 6: Response to User
```json
{
  "risk_score": 0.8234,
  "predicted_class": 1,
  "risk_level": "HIGH",
  "threshold_band": "0.35-0.9",
  "top_factors": [
    {"feature": "glucose", "importance": 0.32},
    {"feature": "age", "importance": 0.28},
    ...
  ],
  "validation_warnings": [...],
  "explanation_text": "Your risk score of 0.82 indicates HIGH risk...",
  "calibration_context": {...}
}
```

---

## 7. Integration Points

### 7.1 **Terminal Integration**
- **Script:** `scripts/predict_from_terminal.py`
- **Advantage:** Batch processing, non-interactive workflows
- **Audit:** Same database logging as API
- **Example:** `python predict_from_terminal.py --raw-input input.json --explain-raw`

### 7.2 **Future EHR Integration** (FHIR)
- **API Design:** Already REST-first (ready for FHIR adapters)
- **Data Mapping:** Need FHIR Observation → raw feature mapping
- **Timeline:** Phase 3 commercialization roadmap
- **Benefit:** Seamless clinic workflow integration

### 7.3 **Monitoring & Analytics** (Future)
- **Data Source:** `risk_explanation_log` table
- **Queries:** Aggregated statistics on disease prevalence, model performance
- **Dashboard:** Track system health, model drift, user engagement

---

## 8. System Quality Attributes

| Attribute | Requirement | Implementation |
|-----------|-------------|-----------------|
| **Performance** | <50ms per prediction | Lightweight RandomForest, optimized SQLite queries |
| **Scalability** | Handle 1000+ concurrent users | Stateless FastAPI, database connection pooling |
| **Reliability** | 99.9% uptime | Health checks, error handling, graceful degradation |
| **Security** | HIPAA-compliant | On-premise deployment, encrypted database, audit logs |
| **Maintainability** | Easy model updates | Artifact versioning, feature lists in JSON |
| **Usability** | Clear explanations | Top factors, risk text, validation warnings |
| **Compliance** | Full audit trail | SQLite risk_explanation_log with source & timestamp |

---

## 9. Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.13.1 |
| **API Framework** | FastAPI | Latest |
| **Database** | SQLite | v3 |
| **ML Framework** | scikit-learn | Latest |
| **Data Validation** | Pydantic | v2 |
| **Serialization** | JSON | Native |
| **Deployment** | On-premise | Local/Cloud-ready |

---

## 10. Security & Compliance

### 10.1 **Data Privacy**
- **On-premise Deployment:** No cloud data transit
- **Database Encryption:** SQLite with optional encryption
- **Access Control:** API-level authentication (future: OAuth2)
- **Audit Logging:** Every prediction timestamped and logged

### 10.2 **HIPAA Compliance** (Target)
- **Minimum:** Audit logs with user identification and timestamps ✓
- **In Progress:** Access control, data encryption
- **Future:** Business associate agreements for cloud deployment

### 10.3 **Model Governance**
- **Version Control:** Models stored as joblib artifacts with hash IDs
- **Feature Validation:** Feature lists validated at runtime
- **Threshold Governance:** Risk thresholds configurable, tracked

---

## 11. Scalability & Future Roadmap

### Phase 1: Foundation (Complete)
- ✓ Risk prediction for 3 diseases
- ✓ Explain-raw feature with audit logging
- ✓ REST API with retrieval endpoints
- ✓ SQLite persistence

### Phase 2: Validation (Current)
- Clinical validation studies
- Feasibility analysis with stakeholders
- Regulatory requirements assessment

### Phase 3: Regulatory & Pilot (12-18 months)
- FDA/CE Mark regulatory pathways
- Clinical trial with EHR integration
- Pilot deployment at 2-3 healthcare centers

### Phase 4: Scale (24+ months)
- National/regional deployment
- Integration with major EHR vendors (Epic, Cerner)
- Multi-center observational studies
- Commercialization through partnerships

---
