import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient

from backend.app import app
from backend.db import init_db
from backend.model_registry import registry


def run_smoke_test() -> None:
    init_db()
    registry.load()
    client = TestClient(app)

    health = client.get("/health")
    print("GET /health", health.status_code, health.json())

    raw_template = client.get("/raw-input-template/ckd")
    print("GET /raw-input-template/ckd", raw_template.status_code, raw_template.json())

    predict_payload = {
        "disease": "ckd",
        "user_id": "u_demo_01",
        "features": {},
    }
    predict = client.post("/predict-risk", json=predict_payload)
    print("POST /predict-risk", predict.status_code, predict.json())

    raw_predict_payload = {
        "disease": "ckd",
        "user_id": "u_demo_raw_01",
        "raw_inputs": {
            "age_years": 57,
            "sex": "female",
            "body_mass_index": 29.1,
            "diagnosed_diabetes": True,
            "diagnosed_hypertension": True,
            "systolic_bp_mmhg": 146,
            "diastolic_bp_mmhg": 90,
            "fasting_glucose_mg_dl": 136,
        },
    }
    raw_predict = client.post("/predict-risk-raw", json=raw_predict_payload)
    print("POST /predict-risk-raw", raw_predict.status_code, raw_predict.json())

    raw_explain = client.post(
        "/risk-explanation-raw",
        json={
            "disease": "ckd",
            "user_id": "u_demo_raw_explain_01",
            "raw_inputs": raw_predict_payload["raw_inputs"],
        },
    )
    print("POST /risk-explanation-raw", raw_explain.status_code, raw_explain.json())

    explain_history = client.get(
        "/risk-explanations",
        params={"disease": "ckd", "source": "api_risk_explanation_raw", "limit": 3, "offset": 0},
    )
    print("GET /risk-explanations", explain_history.status_code, explain_history.json())

    raw_predict_hyp_payload = {
        "disease": "hypertension",
        "user_id": "u_demo_raw_01_hyp",
        "raw_inputs": {
            "age_years": 64,
            "sex": "male",
            "education_level": 5,
            "current_smoker": True,
            "cigarettes_per_day": 9,
            "prior_stroke": False,
            "body_mass_index": 30.8,
            "diagnosed_diabetes": True,
            "systolic_bp_mmhg": 150,
            "diastolic_bp_mmhg": 92,
            "fasting_glucose_mg_dl": 144,
        },
    }
    raw_predict_hyp = client.post("/predict-risk-raw", json=raw_predict_hyp_payload)
    print("POST /predict-risk-raw (hypertension)", raw_predict_hyp.status_code, raw_predict_hyp.json())

    recommend_payload = {
        "user_id": "u_demo_01",
        "disease": "ckd",
        "risk_score": 0.72,
        "profile": {"calories_target": 2100},
    }
    recommend = client.post("/recommend-plan", json=recommend_payload)
    print("POST /recommend-plan", recommend.status_code, recommend.json())

    log_meal_payload = {
        "user_id": "u_demo_01",
        "planned_calories": 700,
        "consumed_calories": 980,
        "notes": "Lunch with high sodium food.",
    }
    log_meal = client.post("/log-meal", json=log_meal_payload)
    print("POST /log-meal", log_meal.status_code, log_meal.json())

    report_payload = {
        "disease": "ckd",
        "user_id": "u_demo_01",
        "features": {},
        "profile": {"calories_target": 2000},
        "top_n_foods": 5,
    }
    report = client.post("/risk-report", json=report_payload)
    print("POST /risk-report", report.status_code, report.json())

    raw_report_payload = {
        "disease": "diabetes",
        "user_id": "u_demo_raw_02",
        "raw_inputs": {
            "age_years": 61,
            "sex": "male",
            "body_mass_index": 31.7,
            "current_smoker": False,
            "diagnosed_hypertension": True,
            "systolic_bp_mmhg": 152,
            "diastolic_bp_mmhg": 94,
            "fasting_glucose_mg_dl": 174,
        },
        "profile": {"calories_target": 1900},
        "top_n_foods": 5,
    }
    raw_report = client.post("/risk-report-raw", json=raw_report_payload)
    print("POST /risk-report-raw", raw_report.status_code, raw_report.json())


if __name__ == "__main__":
    run_smoke_test()
