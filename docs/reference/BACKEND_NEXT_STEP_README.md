# Backend Next Step Implementation

This file documents the first implementation slice beyond modeling:

- FastAPI backend: backend/app.py
- SQLite persistence: backend/db.py
- Inference model loading: backend/model_registry.py
- Recommendation and alert rules: backend/rules.py
- Serving model artifact trainer: train_models.py
- API smoke test: backend_smoke_test.py
- Threshold and factor profiler: build_risk_profiles.py
- Unified nutrition report engine: backend/reporting.py

## Endpoints

- GET /health
- POST /predict-risk
- POST /recommend-plan
- POST /log-meal
- POST /risk-report

## Run

```powershell
pip install -r requirements-backend.txt
python scripts/preprocess_all_datasets.py --source legacy
python scripts/train_models.py --data-source preprocessed
python scripts/build_risk_profiles.py
python scripts/backend_smoke_test.py
uvicorn backend.app:app --reload
```

## Notes

- Active risk modeling now uses full disease-specific clinical feature sets from CKD/UCI, Framingham, and Pima.
- Current recommendation logic now integrates unified food-nutrient KB filtering and scoring.
- /predict-risk expects features aligned to the active artifact feature columns for each disease.
- /risk-report returns threshold-aware risk band, top disease factors, personalized macro/sodium/potassium plan, and top food candidates.
