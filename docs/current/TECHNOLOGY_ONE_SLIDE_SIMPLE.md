# Technology Stack - Copy Paste Ready

## SLIDE CONTENT (Just copy this section)

---

### Technology Stack & Components

**Core Technologies:**
Python 3.13.1 | FastAPI | Pydantic v2 | scikit-learn | SQLite | Uvicorn

**3 ML Models:**
CKD (F1: 0.9868) | Hypertension (F1: 0.8764) | Diabetes (F1: 0.7175)
All using RandomForest with 150 trees

**Database:**
SQLite with 5 tables
risk_explanation_log (Full audit trail with timestamps)
risk_profile, diet_plan, meal_log, alert

**Development Tools:**
VS Code | Git/GitHub | pytest | black | flake8 | Jupyter (for experiments)

**Model Artifacts & Data Assets:**
joblib model files (3) | feature config JSON files | risk threshold config JSON
Training data: 1,455,590 records across CKD, Hypertension, and Diabetes datasets

**Deployment Components:**
Python virtual environment (.venv)
Uvicorn ASGI runtime
On-premise setup with local SQLite persistence

**Key Features:**
<50ms latency per prediction
Full audit logging for every prediction
Complete feature tracking (raw input, normalized features, top factors)
On-premise deployment (HIPAA-ready)
No cloud dependency

**Why This Stack:**
Python has the best ML ecosystem
FastAPI provides automatic API documentation (Swagger/ReDoc)
scikit-learn gives interpretable models (clinical requirement)
SQLite enables on-premise HIPAA-compliant deployment
Pydantic ensures type-safe JSON handling
All models are lightweight (8MB each)

**System Components:**
7 core Python modules
3 ML models (joblib format)
1,455,590 training records
5 database tables
Full terminal CLI interface
Reporting and documentation assets

**Performance:**
Prediction latency: <50ms
Model size: 8MB each
Database: 50-100MB (with 10K+ predictions)
Memory usage: ~500MB (all 3 models + API)
Uptime target: 99.9%

---
