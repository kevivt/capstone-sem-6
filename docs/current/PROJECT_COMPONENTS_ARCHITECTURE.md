# Project Components Architecture

## Personalized Disease Risk Prediction + Diet Recommendation System

---

## 1. Component Hierarchy Map

```
PERSONALIZED DISEASE RISK PREDICTION + DIET RECOMMENDATION SYSTEM
│
├── 1. API & WEB LAYER (FastAPI)
│   ├── 1.1 REST Endpoints (9 endpoints)
│   │   ├── GET /health
│   │   ├── POST /predict-risk
│   │   ├── POST /predict-risk-raw
│   │   ├── POST /risk-explanation-raw ⭐ MAIN
│   │   ├── GET /risk-explanations (Retrieval)
│   │   ├── POST /recommend-plan
│   │   ├── POST /log-meal
│   │   ├── POST /risk-report
│   │   └── POST /risk-report-raw
│   │
│   ├── 1.2 Request/Response Validation (Pydantic)
│   │   ├── RawInputRequest
│   │   ├── RiskExplanationRawResponse
│   │   ├── RiskExplanationLogRecord
│   │   └── RiskExplanationLogListResponse
│   │
│   └── 1.3 Documentation (Auto-generated)
│       ├── Swagger UI (/docs)
│       ├── ReDoc (/redoc)
│       └── OpenAPI Schema (/openapi.json)
│
├── 2. PROCESSING PIPELINE (Core Logic)
│   ├── 2.1 Input Validation & Preprocessing
│   │   ├── Feature validation
│   │   ├── Data normalization (StandardScaler)
│   │   ├── Outlier detection
│   │   └── Missing value handling
│   │
│   ├── 2.2 Prediction Engine (ML Models)
│   │   ├── CKD Model (RandomForest, 150 trees)
│   │   ├── Hypertension Model (RandomForest, 150 trees)
│   │   └── Diabetes Model (RandomForest, 150 trees)
│   │
│   ├── 2.3 Explanation Engine (Interpretability)
│   │   ├── Feature importance ranking
│   │   ├── Risk level banding (LOW/MODERATE/HIGH)
│   │   ├── Validation warning generation
│   │   └── Explanation text synthesis
│   │
│   ├── 2.4 Recommendation Engine (Rules-based)
│   │   ├── Disease-specific dietary rules (KDOQI, ADA, DASH)
│   │   ├── Nutrient target calculation
│   │   └── Meal planning
│   │
│   └── 2.5 Audit & Logging Service
│       ├── Prediction persistence
│       ├── Audit trail creation
│       └── History retrieval (filtering, pagination)
│
├── 3. DATA & KNOWLEDGE LAYER
│   ├── 3.1 SQLite Database (app_data.db)
│   │   ├── risk_profile (Standard predictions)
│   │   ├── risk_explanation_log (Audit trail - NEW)
│   │   ├── diet_plan (Recommendations)
│   │   ├── meal_log (User consumption tracking)
│   │   └── alert (Critical notifications)
│   │
│   ├── 3.2 ML Model Artifacts (artifacts/)
│   │   ├── ckd_model.joblib
│   │   ├── hypertension_model.joblib
│   │   ├── diabetes_model.joblib
│   │   ├── ckd_features.json
│   │   ├── hypertension_features.json
│   │   ├── diabetes_features.json
│   │   └── risk_thresholds_and_factors.json
│   │
│   ├── 3.3 Training Datasets (preprocessed_outputs/)
│   │   ├── ckd_preprocessed.csv (455,590 rows)
│   │   ├── hypertension_preprocessed.csv (500,000 rows)
│   │   └── diabetes_preprocessed.csv (500,000 rows)
│   │
│   ├── 3.4 Configuration & Demo Data (demo_inputs/)
│   │   ├── ckd_risk_report_input.json
│   │   ├── hypertension_risk_report_input.json
│   │   ├── diabetes_risk_report_input.json
│   │   └── raw_samples/ (Explain-raw demo inputs)
│   │
│   └── 3.5 Generated Outputs (outputs/)
│       ├── explanations/ (Risk explanation JSON)
│       ├── esa_phase2/ (ESA submission outputs)
│       └── reports/ (Generated reports)
│
├── 4. INTEGRATION LAYER
│   ├── 4.1 Terminal/CLI Interface
│   │   ├── predict_from_terminal.py (Standalone CLI)
│   │   └── Database logging integration
│   │
│   ├── 4.2 File System Interface
│   │   ├── CSV input/output
│   │   ├── JSON demo data
│   │   └── Report generation
│   │
│   ├── 4.3 Future EHR Integration (FHIR)
│   │   ├── HL7 FHIR adapters (Phase 3)
│   │   └── EHR system bridges
│   │
│   └── 4.4 Analytics & Monitoring
│       ├── Health check endpoint
│       ├── Logging (Python logging)
│       └── Metrics collection (Future: Prometheus)
│
└── 5. DEVELOPMENT & DEPLOYMENT
    ├── 5.1 Development Tools
    │   ├── Python 3.13.1
    │   ├── Virtual environment (.venv)
    │   ├── pytest (Testing)
    │   ├── black (Formatting)
    │   └── flake8 (Linting)
    │
    ├── 5.2 Deployment Options
    │   ├── On-premise (Current: laptop/server)
    │   ├── Docker (Containerization - Future)
    │   ├── Kubernetes (Orchestration - Phase 3+)
    │   └── Cloud (AWS/Azure - Phase 3+)
    │
    ├── 5.3 Documentation
    │   ├── Markdown guides (docs/current/)
    │   ├── API documentation (Swagger/ReDoc)
    │   ├── Code docstrings
    │   └── Architecture diagrams
    │
    └── 5.4 Version Control
        ├── Git repository
        ├── GitHub/GitLab
        └── Branch strategy (main/develop/feature)
```

---

## 2. Component Dependency Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                             │
│                   (Mobile/Web/CLI/API)                           │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│           1. FastAPI APPLICATION SERVER                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Route Matching & Request Handling                        │   │
│  │ (router.post("/risk-explanation-raw"))                  │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │           2. PYDANTIC VALIDATION LAYER                   │   │
│  │  ┌───────────────────────────────────────────────────┐   │   │
│  │  │ Parse JSON → RawInputRequest (type checking)      │   │   │
│  │  │ Validate: age, male, BMI, glucose, etc.           │   │   │
│  │  │ Error handling with detailed messages             │   │   │
│  │  └───────────────┬─────────────────────────────────┘   │   │
│  └────────────────┼─────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │       3. INPUT PREPROCESSING COMPONENT               │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │ • Feature extraction & ordering              │    │   │
│  │  │ • StandardScaler normalization               │    │   │
│  │  │ • Range validation (vs training dist)        │    │   │
│  │  │ • Missing value imputation                   │    │   │
│  │  │ Returns: transformed features + metadata     │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  └────────────────┼─────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │     4. PREDICTION ENGINE (ML Models)                 │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │ Load Model (from artifacts/):                │    │   │
│  │  │ • model.joblib (RandomForest, 150 trees)     │    │   │
│  │  │                                              │    │   │
│  │  │ Predict:                                     │    │   │
│  │  │ • risk_score (0.0-1.0)                       │    │   │
│  │  │ • predicted_class (0 or 1)                   │    │   │
│  │  │ • Feature importance (from model)            │    │   │
│  │  │                                              │    │   │
│  │  │ Performance (<50ms per prediction)           │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  └────────────────┼─────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │       5. EXPLANATION ENGINE (Interpretability)        │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │ • Extract top 5 features by importance       │    │   │
│  │  │ • Map score to risk band (LOW/MODERATE/HIGH) │    │   │
│  │  │ • Generate validation warnings               │    │   │
│  │  │ • Create explanation text                    │    │   │
│  │  │ • Provide calibration context                │    │   │
│  │  │                                              │    │   │
│  │  │ Returns: RiskExplanationRawResponse          │    │   │
│  │  │ {                                            │    │   │
│  │  │   risk_score, risk_level, top_factors,       │    │   │
│  │  │   validation_warnings, explanation_text,     │    │   │
│  │  │   calibration_context                        │    │   │
│  │  │ }                                            │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  └────────────────┼─────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │      6. AUDIT & LOGGING SERVICE (Persistence)        │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │ • Format response payload                    │    │   │
│  │  │ • Prepare audit log record                   │    │   │
│  │  │ • Call insert_risk_explanation(...)          │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  └────────────────┼─────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │    7. DATABASE LAYER (SQLite persistence)            │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │ INSERT INTO risk_explanation_log             │    │   │
│  │  │   (id, user_id, disease, risk_score,         │    │   │
│  │  │    risk_level, top_factors_json,             │    │   │
│  │  │    validation_warnings_json,                 │    │   │
│  │  │    raw_inputs_json,                          │    │   │
│  │  │    transformed_features_json,                │    │   │
│  │  │    explanation_text, source, created_at)     │    │   │
│  │  │                                              │    │   │
│  │  │ File: app_data.db                            │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  └────────────────┼─────────────────────────────────────┘   │
│                   │                                          │
│                   ▼ (Commit complete)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │   8. RESPONSE GENERATION (JSON)                      │    │
│  │  Return RiskExplanationRawResponse to client        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    JSON RESPONSE (200 OK)                    │
│  {                                                           │
│    "risk_score": 0.8234,                                    │
│    "predicted_class": 1,                                    │
│    "risk_level": "HIGH",                                    │
│    "threshold_band": "0.35-0.9",                            │
│    "top_factors": [...],                                    │
│    "validation_warnings": [...],                            │
│    "explanation_text": "Your risk is HIGH...",              │
│    "calibration_context": {...}                             │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Module-Level Components

### Module 1: `backend/app.py`
**Purpose:** FastAPI application and endpoint definitions
```
Imports:
  • FastAPI
  • Pydantic models (from schemas.py)
  • Database functions (from db.py)
  • Prediction logic (from model_registry.py, reporting.py)
  • Health checks

Exports:
  • app (FastAPI instance)
  • 9 REST endpoints

Lines: ~400
Dependencies: fastapi, pydantic, others
```

### Module 2: `backend/schemas.py`
**Purpose:** Pydantic request/response validation models
```
Classes:
  • RawInputRequest (all disease input fields)
  • RiskExplanationRawResponse (explanation + factors)
  • RiskExplanationLogRecord (database record schema)
  • RiskExplanationLogListResponse (pagination wrapper)
  • RecommendationResponse (diet plan)
  • DietPlanResponse

Lines: ~200
Dependencies: pydantic
```

### Module 3: `backend/model_registry.py`
**Purpose:** ML model loading and inference
```
Functions:
  • load_models() → Dict[disease, RandomForestClassifier]
  • predict(disease, features) → (score, class, importance)
  • get_feature_names(disease) → List[str]

Artifacts Loaded:
  • {disease}_model.joblib (3 files)
  • {disease}_features.json (3 files)

Lines: ~150
Dependencies: joblib, sklearn, json
```

### Module 4: `backend/reporting.py`
**Purpose:** Explanation generation and risk band mapping
```
Functions:
  • generate_explanation(...) → RiskExplanationRawResponse
  • map_to_risk_band(score) → str (LOW/MODERATE/HIGH)
  • extract_top_factors(importance, k=5) → List[Dict]
  • generate_explanation_text(...) → str
  • validate_features(...) → List[str] (warnings)
  • get_calibration_context(...) → Dict

Lines: ~250
Dependencies: json, numpy, logic
```

### Module 5: `backend/raw_input.py`
**Purpose:** Input validation and preprocessing
```
Functions:
  • validate_raw_input(data) → bool
  • preprocess(data) → numpy.ndarray (scaled)
  • normalize_features(data) → Dict
  • check_feature_ranges(data) → List[warning]

Classes:
  • FeatureValidator
  • FeatureScaler

Lines: ~200
Dependencies: sklearn, numpy
```

### Module 6: `backend/rules.py`
**Purpose:** Diet recommendation rules (KDOQI, ADA, DASH)
```
Constants:
  • DISEASE_NUTRIENT_LIMITS (sodium, potassium, carbs, etc.)
  • DISEASE_MEAL_RECOMMENDATIONS

Functions:
  • generate_diet_plan(disease, preferences) → DietPlanResponse
  • calculate_nutrient_targets(disease, profile) → Dict

Lines: ~300
Dependencies: logic, json
```

### Module 7: `backend/db.py`
**Purpose:** SQLite database management
```
Functions:
  • init_db() → None
  • insert_risk_explanation(...) → int (record_id)
  • get_risk_explanations(...) → List[RiskExplanationLogRecord]
  • insert_diet_plan(...) → int
  • insert_meal_log(...) → int
  • insert_alert(...) → int
  • get_connection() → sqlite3.Connection

Constants:
  • DB_PATH = 'app_data.db'
  • SQL schema definitions (CREATE TABLE statements)

Lines: ~400
Dependencies: sqlite3, json, datetime
```

---

## 4. Data Flow Components

### Data Component 1: Input Data Path
```
JSON Input (from client)
    ↓
Pydantic Validation (schemas.py)
    ↓
Raw Input Processing (raw_input.py)
    ↓
Feature Extraction & Normalization
    ↓
StandardScaler (fitted on training data)
    ↓
Ready for ML Model
```

### Data Component 2: Model Artifact Path
```
Training Data (1.4M records)
    ↓
Training Script (scripts/train_models.py)
    ↓
Trained RandomForest Model
    ↓
Serialization with joblib
    ↓
artifacts/{disease}_model.joblib
    ↓
Load at startup (backend/app.py)
    ↓
In-memory for prediction
```

### Data Component 3: Explanation Payload Path
```
Feature Importance (from model.feature_importances_)
    ↓
Ranking (top 5 by importance)
    ↓
Feature Names (from {disease}_features.json)
    ↓
Feature Values (from user input)
    ↓
Risk Band Mapping (score → LOW/MODERATE/HIGH)
    ↓
Text Synthesis (rule-based explanation)
    ↓
Calibration Context (threshold multipliers)
    ↓
RiskExplanationRawResponse object
```

### Data Component 4: Audit Trail Path
```
API Request Received
    ↓
Prediction Executed
    ↓
Explanation Generated
    ↓
Audit Record Created
    ↓
INSERT INTO risk_explanation_log
    (user_id, disease, risk_score, risk_level,
     top_factors_json, explanation_text, source='api', created_at)
    ↓
SQLite Database (app_data.db)
    ↓
Retrievable via GET /risk-explanations
    (with filtering by disease, user_id, source, date range)
```

---

## 5. Technology Component Mapping

| Component | Technology | File | Purpose |
|-----------|-----------|------|---------|
| API Server | FastAPI | backend/app.py | HTTP endpoint handling |
| Request Validation | Pydantic | backend/schemas.py | Input/output type safety |
| ML Inference | scikit-learn | backend/model_registry.py | Risk prediction |
| Explanation | Python logic | backend/reporting.py | Interpretability |
| Preprocessing | scikit-learn | backend/raw_input.py | Feature normalization |
| Diet Rules | JSON config | backend/rules.py | Recommendation logic |
| Database | SQLite | backend/db.py | Persistence & audit log |
| Models | joblib | artifacts/ | Trained ML models |
| Features | JSON config | artifacts/ | Feature metadata |
| Training Data | CSV | preprocessed_outputs/ | Historical training data |
| CLI Tool | Python | scripts/predict_from_terminal.py | Terminal interface |

---

## 6. Component Interaction Matrix

### API Endpoints (1) → Processing Pipeline (2)
```
Endpoint (1)          Processing Component (2)              
─────────────────────────────────────────────────────────
/health               → Health check (trivial)
/predict-risk         → Preprocessing → Prediction → Response
/predict-risk-raw     → Preprocessing → Prediction → Response
/risk-explanation-raw → Preprocessing → Prediction → Explanation → Audit
/risk-explanations    → Database query (retrieval)
/recommend-plan       → Recommendation engine
/log-meal             → Database insert
/risk-report          → Full report generation
/risk-report-raw      → Full report with raw input
```

### Processing Pipeline (2) → Data Layer (3)
```
Component              Data Required                    Data Persisted
──────────────────────────────────────────────────────────────────
Input Validation      → Feature names (artifacts/)    → (none)
Prediction Engine     → Model (artifacts/)            → (none)
Explanation Engine    → Thresholds (artifacts/)       → (none)
Recommendation        → Nutrient rules (backend/)     → (none)
Audit Service         → (none)                        → risk_explanation_log
```

### Data Layer Components (3) ↔ External Systems (4)
```
Component                  Input From               Output To
─────────────────────────────────────────────────────────────
risk_explanation_log       ← API predictions       → GET /risk-explanations
risk_profile              ← /predict-risk requests  → Reports
diet_plan                 ← /recommend-plan         → Recommendations
meal_log                  ← /log-meal               → Adherence tracking
Training Datasets         → Model training scripts  → artifacts/*.joblib
Demo Inputs               → Test/validation         → Smoke tests
Generated Outputs         ← All predictions         → /outputs/ directory
```

---

## 7. Component Scalability View

### Current Capacity (Phase 1)
```
Component                  Capacity              Bottleneck
─────────────────────────────────────────────────────────────
API Server (FastAPI)       ~1000 req/sec         CPU (preprocessing)
ML Models (in-memory)      Unlimited             Model load time (~500ms)
SQLite Database            ~10M records          Concurrent writes
Disk Space                 ~100MB                Database growth
Memory Usage               ~500MB                Model artifacts (3×8MB)
Prediction Latency         <50ms                 Preprocessing (feature scaling)
```

### Phase 2 Scaling (With Docker)
```
• Container-based deployment
• Multiple API instances (load balancer)
• Database connection pooling
• Model caching improvements
• Expected: 10,000+ req/sec
```

### Phase 3+ Scaling (Enterprise)
```
• Kubernetes orchestration
• PostgreSQL (replace SQLite)
• Redis caching (predictions)
• EHR integration servers
• Monitoring (Prometheus)
• Expected: 100,000+ req/sec
```

---

## 8. Component Risk & Mitigation

| Component | Risk | Mitigation |
|-----------|------|-----------|
| Single SQLite database | Concurrency bottleneck | Phase 2: PostgreSQL migration |
| On-premise deployment | HIPAA burden | Document security controls |
| RandomForest models | Model drift over time | Phase 2: Automated retraining |
| Feature preprocessing | Mismatch with training | Validation warnings system |
| API performance | Slow response times | Caching (Phase 2), optimization |
| Terminal interface | No persistence by default | Integrated with database (done) |

---

## 9. Summary: Complete Component Checklist

### ✓ Implemented Components
- [x] FastAPI application with 9 endpoints
- [x] Pydantic request/response validation
- [x] 3 RandomForest models (CKD, HTN, DM)
- [x] Preprocessing pipeline (StandardScaler)
- [x] Explanation engine (top factors, risk bands, text)
- [x] SQLite database with 5 tables
- [x] Audit logging (risk_explanation_log)
- [x] History retrieval API (GET /risk-explanations)
- [x] Terminal CLI interface
- [x] Demo data and sample inputs
- [x] Smoke test validation script

### 🔜 Planned Components (Phases 2+)
- [ ] Docker containerization
- [ ] Kubernetes orchestration
- [ ] PostgreSQL migration
- [ ] FHIR integration adapters
- [ ] EHR system bridges
- [ ] Advanced monitoring (Prometheus, Grafana)
- [ ] Model retraining pipeline
- [ ] Advanced interpretability (SHAP, LIME)
- [ ] LLM integration for recommendations
- [ ] Federated learning

---
