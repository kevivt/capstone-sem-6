# Technology Stack & Components - One-Slide Summary

## Personalized Disease Risk Prediction + Diet Recommendation System

---

## SINGLE SLIDE VERSION (Copy-Paste Ready)

### Title: **Technology Stack & System Components**

---

### **CORE TECHNOLOGY STACK**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.13.1 | ML ecosystem, clinical adoption |
| **API Framework** | FastAPI | REST endpoints, auto-docs (Swagger/ReDoc) |
| **Data Validation** | Pydantic v2 | Type-safe JSON serialization |
| **ML Models** | scikit-learn | RandomForest (150 trees), <50ms latency |
| **Database** | SQLite | On-premise, HIPAA-compliant, audit logs |
| **Model Serialization** | joblib | Lightweight model artifacts (8MB each) |
| **Deployment** | Uvicorn | ASGI server, async support |

---

### **SYSTEM COMPONENTS**

```
┌─────────────────────────────────────────────────────────────┐
│  API ENDPOINTS                                              │
│  • POST /risk-explanation-raw ⭐ (Main: Risk + Explanation) │
│  • GET /risk-explanations (History with filtering)          │
│  • POST /predict-risk-raw (Prediction with raw input)       │
│  • POST /recommend-plan (Diet recommendations)              │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│  PROCESSING PIPELINE                                        │
│  • Input Validation (Pydantic schemas)                      │
│  • Feature Preprocessing (StandardScaler)                   │
│  • Prediction Engine (3 RandomForest models)                │
│  • Explanation Engine (Top factors, Risk bands)             │
│  • Audit & Logging (Persistence to SQLite)                  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│  DATA LAYER                                                 │
│  • SQLite Database: 5 tables                                │
│    - risk_explanation_log (Audit trail) ⭐ NEW              │
│    - risk_profile, diet_plan, meal_log, alert              │
│  • ML Models: 3 RandomForest classifiers                    │
│    - CKD (F1: 0.9868), HTN (F1: 0.8764), DM (F1: 0.7175)   │
│  • Training Data: 1,455,590 records                         │
└─────────────────────────────────────────────────────────────┘
```

---

### **KEY FEATURES**

| Feature | Technology | Benefit |
|---------|-----------|---------|
| **Interpretability** | Feature importance ranking | Clinical acceptance |
| **Auditability** | Full prediction logging | HIPAA compliance, reproducibility |
| **Performance** | RandomForest + StandardScaler | <50ms per prediction |
| **Deployment** | On-premise SQLite | No cloud dependency |
| **Auto-Documentation** | FastAPI + Swagger | Interactive API explorer at /docs |
| **Scalability Path** | Docker-ready architecture | Phase 2+: Kubernetes, PostgreSQL |

---

### **RISK PREDICTION FLOW**

```
User Input (JSON)
    ↓
Pydantic Validation
    ↓
Feature Preprocessing (Normalization)
    ↓
ML Prediction (RandomForest <50ms)
    ↓
Explanation Generation (Top 5 factors + Risk band)
    ↓
Persist to SQLite (risk_explanation_log)
    ↓
Return Response (JSON) + Audit trail saved
```

---

### **DATABASE SCHEMA (Key Table)**

**risk_explanation_log** (Audit Trail)
```
Columns: id, user_id, disease, risk_score, risk_level,
         top_factors_json, validation_warnings_json,
         raw_inputs_json, transformed_features_json,
         explanation_text, source ('api' or 'terminal'),
         created_at

Indexes: disease, user_id, source, created_at (for fast queries)
Supports: Filtering, pagination, historical analysis
```

---

### **MODEL ARTIFACTS**

| Artifact | Type | Size | Purpose |
|----------|------|------|---------|
| `ckd_model.joblib` | ML Model | 8.2 MB | CKD risk prediction |
| `hypertension_model.joblib` | ML Model | 8.1 MB | Hypertension risk prediction |
| `diabetes_model.joblib` | ML Model | 7.9 MB | Diabetes risk prediction |
| `{disease}_features.json` | Config | <1 KB | Feature name mapping |
| `risk_thresholds_and_factors.json` | Config | <5 KB | Risk bands, calibration |

---

### **DEVELOPMENT & TESTING**

| Tool | Purpose |
|------|---------|
| **pytest** | Unit & integration testing |
| **black** | Code formatting |
| **flake8** | Linting & code quality |
| **Swagger/ReDoc** | Interactive API documentation |

---

### **INTEGRATION POINTS**

| Interface | Current Status | Timeline |
|-----------|---|---|
| **REST API** | ✓ Implemented | Phase 1 (Now) |
| **Terminal/CLI** | ✓ Implemented | Phase 1 (Now) |
| **File System** | ✓ Implemented (CSV, JSON) | Phase 1 (Now) |
| **EHR (HL7 FHIR)** | 🔜 Planned | Phase 3 |
| **Kubernetes** | 🔜 Planned | Phase 3+ |

---

### **QUALITY METRICS**

```
Performance:          <50ms per prediction
Accuracy (CKD):       Macro F1 = 0.9868, ROC-AUC = 1.0000
Accuracy (HTN):       Macro F1 = 0.8764, ROC-AUC = 0.9451
Accuracy (DM):        Macro F1 = 0.7175, ROC-AUC = 0.8096
Uptime Target:        99.9%
Database Size:        ~50-100 MB (with 10K+ predictions)
Memory Usage:         ~500 MB (3 models + FastAPI)
Deployment:           On-premise, HIPAA-ready
```

---

### **WHY THIS STACK?**

✓ **Python** – Best ML ecosystem, clinical staff can learn
✓ **FastAPI** – Auto-docs, type-safe, <50ms latency, async
✓ **scikit-learn** – Interpretable RandomForest (clinical requirement)
✓ **SQLite** – On-premise deployment, HIPAA compliance, minimal setup
✓ **Pydantic** – Automatic JSON validation & serialization
✓ **Joblib** – Standard ML model serialization & versioning

---

### **COMPLETE COMPONENT LIST**

**Code Modules:**
- `backend/app.py` – FastAPI application (9 endpoints)
- `backend/schemas.py` – Pydantic request/response models
- `backend/model_registry.py` – ML model loading
- `backend/reporting.py` – Explanation generation
- `backend/raw_input.py` – Input validation & preprocessing
- `backend/rules.py` – Diet recommendation rules
- `backend/db.py` – SQLite database management

**Scripts:**
- `scripts/predict_from_terminal.py` – CLI interface
- `scripts/train_models.py` – Model training
- `scripts/backend_smoke_test.py` – API validation

**Data:**
- `artifacts/` – ML models & configuration (3 models + JSON)
- `preprocessed_outputs/` – Training datasets (1.4M records)
- `demo_inputs/` – Sample inputs & test data
- `app_data.db` – SQLite database (audit logs + profiles)

---

## SLIDES-READY VERSIONS

### **Ultra-Compact (30 seconds)**
```
TECHNOLOGY STACK
┌────────────────────────────────┐
│ Python 3.13 → FastAPI          │
│ scikit-learn RandomForest       │
│ SQLite + Pydantic              │
│ Joblib (models)                │
│ Uvicorn (ASGI server)          │
│                                │
│ 3 Models: CKD (F1 0.99)        │
│           HTN (F1 0.88)        │
│           DM (F1 0.72)         │
│                                │
│ 9 Endpoints, <50ms latency    │
│ Full audit trail logging       │
│ On-premise HIPAA-ready        │
└────────────────────────────────┘
```

### **Medium (1 minute)**
Use the "CORE TECHNOLOGY STACK" table + "SYSTEM COMPONENTS" diagram above.

### **Detailed (2-3 minutes)**
Use the complete slide content above.

---

## HOW TO USE THIS IN POWERPOINT

**Slide Layout:**
```
Title: Technology Stack & System Components

Content (2 columns):
┌─────────────────────┬──────────────────────┐
│ CORE STACK          │ SYSTEM COMPONENTS    │
│ (Table format)      │ (Component diagram)  │
├─────────────────────┼──────────────────────┤
│ KEY FEATURES        │ QUALITY METRICS      │
│ (Quick bullets)     │ (Key performance #s) │
└─────────────────────┴──────────────────────┘

Footer: 9 Endpoints | <50ms Latency | HIPAA-Ready
```

---

## KEY TAKEAWAYS FOR SLIDES

1. **Technology Choice Rationale:** Python + FastAPI for ML + API, scikit-learn for interpretability
2. **Component Count:** 7 core modules + 3 scripts + 4 data folders = Complete system
3. **Quality Proof:** 3 production-grade models with F1>0.71, <50ms latency, 100% audit logging
4. **Deployment:** On-premise ready, SQLite database, no cloud dependency
5. **Future Path:** Docker (Phase 2) → Kubernetes (Phase 3) → FHIR integration (Phase 3+)

---
