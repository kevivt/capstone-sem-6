from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timezone
from pathlib import Path
import csv

from backend.db import (
    get_risk_explanations,
    get_risk_explanation_summary,
    get_open_alert_count,
    init_db,
    insert_alert,
    insert_diet_plan,
    insert_meal_log,
    insert_risk_explanation,
    insert_risk_profile,
)
from backend.model_registry import registry
from backend.raw_input import get_raw_template, map_raw_payload
from backend.reporting import build_explanation_text, build_report_text, recommend_foods
from backend.rules import (
    assess_meal_deviation,
    build_calibration_context,
    build_plan,
    classify_risk_level,
)
from backend.schemas import (
    PredictRiskRawRequest,
    PredictRiskRawResponse,
    LogMealRequest,
    LogMealResponse,
    PredictRiskRequest,
    PredictRiskResponse,
    RiskExplanationRawRequest,
    RiskExplanationRawResponse,
    RiskExplanationLogListResponse,
    RiskReportRawRequest,
    RiskReportRawResponse,
    RiskReportRequest,
    RiskReportResponse,
    RecommendPlanRequest,
    RecommendPlanResponse,
)


app = FastAPI(title="Diet-Risk Backend", version="0.1.0")


def _read_project_model_metrics() -> list[dict]:
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "reports" / "current" / "baseline_model_comparison.csv"
    if not csv_path.exists():
        return []

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("model_key") != "project_final_model":
                continue
            rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "accuracy": float(row.get("accuracy", 0.0)),
                    "macro_f1": float(row.get("macro_f1", 0.0)),
                    "roc_auc": float(row.get("roc_auc", 0.0)),
                }
            )
    rows.sort(key=lambda r: r.get("dataset", ""))
    return rows


def _count_generated_charts() -> int:
    base_dir = Path(__file__).resolve().parent.parent
    fig_dir = base_dir / "reports" / "current" / "figures" / "model_comparisons"
    if not fig_dir.exists():
        return 0
    return len(list(fig_dir.glob("*.png")))


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


@app.get("/implementation-progress")
def implementation_progress() -> dict:
    registry.load_feature_metadata()
    models_available = sorted(list(registry.features.keys()))
    models_loaded = sorted(list(registry.models.keys()))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_available": models_available,
        "models_loaded": models_loaded,
        "open_alerts": get_open_alert_count(),
        "audit_summary": get_risk_explanation_summary(),
        "project_model_metrics": _read_project_model_metrics(),
        "generated_model_comparison_charts": _count_generated_charts(),
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
    calibration_context = build_calibration_context(
        disease=req.disease,
        risk_score=score,
        thresholds=thresholds,
        top_factors=profile_cfg.get("top_factors", []),
        transformed_features=req.features,
    )

    return PredictRiskResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        threshold_used=round(threshold_used, 6),
        thresholds={
            "moderate": round(float(thresholds.get("moderate", 0.5)), 6),
            "high": round(float(thresholds.get("high", 0.75)), 6),
        },
        threshold_band=str(calibration_context.get("threshold_band", "below_moderate")),
        major_risk_factors=list(calibration_context.get("major_risk_factors", [])),
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
    calibration_context = build_calibration_context(
        disease=req.disease,
        risk_score=score,
        thresholds=thresholds,
        top_factors=profile_cfg.get("top_factors", []),
        transformed_features=mapped.transformed_features,
    )

    return PredictRiskRawResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        threshold_used=round(threshold_used, 6),
        thresholds={
            "moderate": round(float(thresholds.get("moderate", 0.5)), 6),
            "high": round(float(thresholds.get("high", 0.75)), 6),
        },
        threshold_band=str(calibration_context.get("threshold_band", "below_moderate")),
        major_risk_factors=list(calibration_context.get("major_risk_factors", [])),
        model_name=model_name,
        transformed_features=mapped.transformed_features,
        raw_inputs=mapped.raw_inputs,
        validation_warnings=mapped.validation_warnings,
    )


@app.post("/risk-explanation-raw", response_model=RiskExplanationRawResponse)
def risk_explanation_raw(req: RiskExplanationRawRequest) -> RiskExplanationRawResponse:
    try:
        mapped = map_raw_payload(req.disease, req.raw_inputs)
        pred, score, model_name = registry.predict(req.disease, mapped.transformed_features)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    if req.user_id:
        insert_risk_profile(req.user_id, req.disease, score, pred)

    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {})
    top_factors = profile_cfg.get("top_factors", [])
    threshold_used = float(thresholds.get("moderate", 0.5))
    predicted_class = int(1 if score >= threshold_used else 0)
    risk_level = classify_risk_level(score, thresholds)
    calibration_context = build_calibration_context(
        disease=req.disease,
        risk_score=score,
        thresholds=thresholds,
        top_factors=top_factors,
        transformed_features=mapped.transformed_features,
    )
    explanation_text = build_explanation_text(
        disease=req.disease,
        risk_score=score,
        risk_level=risk_level,
        threshold_band=str(calibration_context.get("threshold_band", "below_moderate")),
        major_risk_factors=list(calibration_context.get("major_risk_factors", [])),
        validation_warnings=mapped.validation_warnings,
    )

    insert_risk_explanation(
        disease=req.disease,
        model_name=model_name,
        predicted_class=predicted_class,
        risk_score=score,
        risk_level=risk_level,
        threshold_moderate=float(thresholds.get("moderate", 0.5)),
        threshold_high=float(thresholds.get("high", 0.75)),
        threshold_band=str(calibration_context.get("threshold_band", "below_moderate")),
        major_risk_factors=list(calibration_context.get("major_risk_factors", [])),
        validation_warnings=mapped.validation_warnings,
        raw_inputs=mapped.raw_inputs,
        transformed_features=mapped.transformed_features,
        top_factors=top_factors,
        calibration_context=calibration_context,
        explanation_text=explanation_text,
        user_id=req.user_id,
        source="api_risk_explanation_raw",
    )

    return RiskExplanationRawResponse(
        disease=req.disease,
        predicted_class=predicted_class,
        risk_score=round(score, 6),
        risk_level=risk_level,
        threshold_used=round(threshold_used, 6),
        thresholds={
            "moderate": round(float(thresholds.get("moderate", 0.5)), 6),
            "high": round(float(thresholds.get("high", 0.75)), 6),
        },
        threshold_band=str(calibration_context.get("threshold_band", "below_moderate")),
        major_risk_factors=list(calibration_context.get("major_risk_factors", [])),
        model_name=model_name,
        transformed_features=mapped.transformed_features,
        raw_inputs=mapped.raw_inputs,
        validation_warnings=mapped.validation_warnings,
        top_factors=top_factors,
        calibration_context=calibration_context,
        explanation_text=explanation_text,
    )


@app.get("/risk-explanations", response_model=RiskExplanationLogListResponse)
def risk_explanations(
    disease: str | None = None,
    user_id: str | None = None,
    source: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> RiskExplanationLogListResponse:
    items = get_risk_explanations(
        disease=disease,
        user_id=user_id,
        source=source,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
        offset=offset,
    )
    return RiskExplanationLogListResponse(
        count=len(items),
        limit=limit,
        offset=offset,
        items=items,
    )


@app.post("/recommend-plan", response_model=RecommendPlanResponse)
def recommend_plan(req: RecommendPlanRequest) -> RecommendPlanResponse:
    profile_cfg = registry.get_risk_profile(req.disease)
    thresholds = profile_cfg.get("thresholds", {})
    calibration_context = build_calibration_context(
        disease=req.disease,
        risk_score=req.risk_score,
        thresholds=thresholds,
        top_factors=profile_cfg.get("top_factors", []),
        transformed_features={},
    )
    plan = build_plan(req.disease, req.risk_score, req.profile, thresholds, calibration_context)
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

    calibration_context = build_calibration_context(
        disease=req.disease,
        risk_score=score,
        thresholds=thresholds,
        top_factors=top_factors,
        transformed_features=req.features,
    )
    plan = build_plan(req.disease, score, req.profile, thresholds, calibration_context)
    foods = recommend_foods(req.disease, risk_level, plan, top_n=req.top_n_foods)
    narrative = build_report_text(req.disease, score, risk_level, top_factors, calibration_context)

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
        calibration_context=calibration_context,
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

    calibration_context = build_calibration_context(
        disease=req.disease,
        risk_score=score,
        thresholds=thresholds,
        top_factors=top_factors,
        transformed_features=mapped.transformed_features,
    )
    plan = build_plan(req.disease, score, req.profile, thresholds, calibration_context)
    foods = recommend_foods(req.disease, risk_level, plan, top_n=req.top_n_foods)
    narrative = build_report_text(req.disease, score, risk_level, top_factors, calibration_context)

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
        calibration_context=calibration_context,
        plan=plan,
        recommended_foods=foods,
        report_text=narrative,
        transformed_features=mapped.transformed_features,
        raw_inputs=mapped.raw_inputs,
        validation_warnings=mapped.validation_warnings,
    )
