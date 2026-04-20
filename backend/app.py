from fastapi import FastAPI, HTTPException

from backend.db import (
    get_open_alert_count,
    init_db,
    insert_alert,
    insert_diet_plan,
    insert_meal_log,
    insert_risk_profile,
)
from backend.model_registry import registry
from backend.raw_input import get_raw_template, map_raw_payload
from backend.reporting import build_report_text, recommend_foods
from backend.rules import assess_meal_deviation, build_plan, classify_risk_level
from backend.schemas import (
    PredictRiskRawRequest,
    PredictRiskRawResponse,
    LogMealRequest,
    LogMealResponse,
    PredictRiskRequest,
    PredictRiskResponse,
    RiskReportRawRequest,
    RiskReportRawResponse,
    RiskReportRequest,
    RiskReportResponse,
    RecommendPlanRequest,
    RecommendPlanResponse,
)


app = FastAPI(title="Diet-Risk Backend", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    init_db()
    registry.load()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": sorted(list(registry.models.keys())),
        "open_alerts": get_open_alert_count(),
    }


@app.get("/raw-input-template/{disease}")
def raw_input_template(disease: str) -> dict:
    try:
        return get_raw_template(disease)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex


@app.post("/predict-risk", response_model=PredictRiskResponse)
def predict_risk(req: PredictRiskRequest) -> PredictRiskResponse:
    try:
        pred, score, model_name = registry.predict(req.disease, req.features)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    if req.user_id:
        insert_risk_profile(req.user_id, req.disease, score, pred)

    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {})
    threshold_used = float(thresholds.get("moderate", 0.5))
    predicted_class = int(1 if score >= threshold_used else 0)
    risk_level = classify_risk_level(score, thresholds)

    return PredictRiskResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        threshold_used=round(threshold_used, 6),
        model_name=model_name,
    )


@app.post("/predict-risk-raw", response_model=PredictRiskRawResponse)
def predict_risk_raw(req: PredictRiskRawRequest) -> PredictRiskRawResponse:
    try:
        mapped = map_raw_payload(req.disease, req.raw_inputs)
        pred, score, model_name = registry.predict(req.disease, mapped.transformed_features)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    if req.user_id:
        insert_risk_profile(req.user_id, req.disease, score, pred)

    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {})
    threshold_used = float(thresholds.get("moderate", 0.5))
    predicted_class = int(1 if score >= threshold_used else 0)
    risk_level = classify_risk_level(score, thresholds)

    return PredictRiskRawResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        threshold_used=round(threshold_used, 6),
        model_name=model_name,
        transformed_features=mapped.transformed_features,
        raw_inputs=mapped.raw_inputs,
        validation_warnings=mapped.validation_warnings,
    )


@app.post("/recommend-plan", response_model=RecommendPlanResponse)
def recommend_plan(req: RecommendPlanRequest) -> RecommendPlanResponse:
    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {})
    plan = build_plan(req.disease, req.risk_score, req.profile, thresholds)
    insert_diet_plan(req.user_id, req.disease, plan)

    return RecommendPlanResponse(
        user_id=req.user_id,
        disease=req.disease,
        **plan,
    )


@app.post("/log-meal", response_model=LogMealResponse)
def log_meal(req: LogMealRequest) -> LogMealResponse:
    result = assess_meal_deviation(req.planned_calories, req.consumed_calories)
    meal_log_id = insert_meal_log(
        user_id=req.user_id,
        planned_calories=req.planned_calories,
        consumed_calories=req.consumed_calories,
        deviation_flag=result["deviation_flag"],
        deviation_percent=result["deviation_percent"],
        notes=req.notes,
    )

    alert_created = False
    alert_message = None

    if result["deviation_flag"]:
        alert_message = (
            f"Meal deviation detected: {result['deviation_percent']}% "
            f"({result['severity']} severity)."
        )
        insert_alert(req.user_id, result["severity"], alert_message)
        alert_created = True

    return LogMealResponse(
        meal_log_id=meal_log_id,
        user_id=req.user_id,
        deviation_flag=result["deviation_flag"],
        deviation_percent=result["deviation_percent"],
        alert_created=alert_created,
        alert_message=alert_message,
    )


@app.post("/risk-report", response_model=RiskReportResponse)
def risk_report(req: RiskReportRequest) -> RiskReportResponse:
    try:
        pred, score, model_name = registry.predict(req.disease, req.features)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {"moderate": 0.4, "high": 0.75})
    top_factors = profile_cfg.get("top_factors", [])

    threshold_used = float(thresholds.get("moderate", 0.5))
    predicted_class = int(1 if score >= threshold_used else 0)
    risk_level = classify_risk_level(score, thresholds)

    plan = build_plan(req.disease, score, req.profile, thresholds)
    foods = recommend_foods(req.disease, risk_level, plan, top_n=req.top_n_foods)
    narrative = build_report_text(req.disease, score, risk_level, top_factors)

    if req.user_id:
        insert_risk_profile(req.user_id, req.disease, score, pred)
        insert_diet_plan(req.user_id, req.disease, plan)

    return RiskReportResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        model_name=model_name,
        thresholds={
            "moderate": round(float(thresholds.get("moderate", 0.4)), 6),
            "high": round(float(thresholds.get("high", 0.75)), 6),
        },
        top_factors=top_factors,
        plan=plan,
        recommended_foods=foods,
        report_text=narrative,
    )


@app.post("/risk-report-raw", response_model=RiskReportRawResponse)
def risk_report_raw(req: RiskReportRawRequest) -> RiskReportRawResponse:
    try:
        mapped = map_raw_payload(req.disease, req.raw_inputs)
        pred, score, model_name = registry.predict(req.disease, mapped.transformed_features)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {"moderate": 0.4, "high": 0.75})
    top_factors = profile_cfg.get("top_factors", [])

    threshold_used = float(thresholds.get("moderate", 0.5))
    predicted_class = int(1 if score >= threshold_used else 0)
    risk_level = classify_risk_level(score, thresholds)

    plan = build_plan(req.disease, score, req.profile, thresholds)
    foods = recommend_foods(req.disease, risk_level, plan, top_n=req.top_n_foods)
    narrative = build_report_text(req.disease, score, risk_level, top_factors)

    if req.user_id:
        insert_risk_profile(req.user_id, req.disease, score, pred)
        insert_diet_plan(req.user_id, req.disease, plan)

    return RiskReportRawResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        model_name=model_name,
        thresholds={
            "moderate": round(float(thresholds.get("moderate", 0.4)), 6),
            "high": round(float(thresholds.get("high", 0.75)), 6),
        },
        top_factors=top_factors,
        plan=plan,
        recommended_foods=foods,
        report_text=narrative,
        transformed_features=mapped.transformed_features,
        raw_inputs=mapped.raw_inputs,
        validation_warnings=mapped.validation_warnings,
    )
