from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


DiseaseType = Literal["ckd", "hypertension", "diabetes"]
SexType = Literal["male", "female"]


class PredictRiskRequest(BaseModel):
    disease: DiseaseType
    features: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None


class PredictRiskResponse(BaseModel):
    disease: DiseaseType
    predicted_class: int
    risk_score: float
    risk_level: Literal["low", "moderate", "high"]
    threshold_used: float
    model_name: str


class CKDRawInput(BaseModel):
    age_years: int = Field(ge=18, le=100)
    sex: SexType
    body_mass_index: float = Field(ge=10.0, le=90.0)
    diagnosed_diabetes: bool = False
    diagnosed_hypertension: bool = False
    systolic_bp_mmhg: Optional[float] = Field(default=None, ge=70.0, le=260.0)
    diastolic_bp_mmhg: Optional[float] = Field(default=None, ge=40.0, le=160.0)
    fasting_glucose_mg_dl: Optional[float] = Field(default=None, ge=40.0, le=450.0)


class HypertensionRawInput(BaseModel):
    age_years: int = Field(ge=18, le=100)
    sex: SexType
    education_level: int = Field(default=5, ge=1, le=9)
    current_smoker: bool = False
    cigarettes_per_day: float = Field(default=0.0, ge=0.0, le=100.0)
    prior_stroke: bool = False
    body_mass_index: float = Field(ge=10.0, le=90.0)
    diagnosed_diabetes: bool = False
    systolic_bp_mmhg: Optional[float] = Field(default=None, ge=70.0, le=260.0)
    diastolic_bp_mmhg: Optional[float] = Field(default=None, ge=40.0, le=160.0)
    fasting_glucose_mg_dl: Optional[float] = Field(default=None, ge=40.0, le=450.0)


class DiabetesRawInput(BaseModel):
    age_years: int = Field(ge=18, le=100)
    sex: SexType
    body_mass_index: float = Field(ge=10.0, le=90.0)
    current_smoker: bool = False
    diagnosed_hypertension: bool = False
    systolic_bp_mmhg: Optional[float] = Field(default=None, ge=70.0, le=260.0)
    diastolic_bp_mmhg: Optional[float] = Field(default=None, ge=40.0, le=160.0)
    fasting_glucose_mg_dl: Optional[float] = Field(default=None, ge=40.0, le=450.0)


class PredictRiskRawRequest(BaseModel):
    disease: DiseaseType
    raw_inputs: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None


class PredictRiskRawResponse(PredictRiskResponse):
    transformed_features: Dict[str, Any]
    raw_inputs: Dict[str, Any]
    validation_warnings: list[str] = Field(default_factory=list)


class RecommendPlanRequest(BaseModel):
    user_id: str
    disease: DiseaseType
    profile: Dict[str, Any] = Field(default_factory=dict)
    risk_score: float = Field(ge=0.0, le=1.0)


class RecommendPlanResponse(BaseModel):
    user_id: str
    disease: DiseaseType
    calories_target: int
    protein_target_g: int
    carb_target_g: int
    fat_target_g: int
    sodium_limit_mg: int
    potassium_limit_mg: int
    sugar_limit_g: Optional[int] = None
    phosphorus_limit_mg: Optional[int] = None
    fiber_target_g: Optional[int] = None
    fluid_target_ml: Optional[int] = None
    diet_focus: list[str] = Field(default_factory=list)
    recommendation_note: str


class LogMealRequest(BaseModel):
    user_id: str
    planned_calories: int = Field(gt=0)
    consumed_calories: int = Field(gt=0)
    notes: str = ""


class LogMealResponse(BaseModel):
    meal_log_id: int
    user_id: str
    deviation_flag: bool
    deviation_percent: float
    alert_created: bool
    alert_message: Optional[str] = None


class RiskReportRequest(BaseModel):
    disease: DiseaseType
    features: Dict[str, Any] = Field(default_factory=dict)
    profile: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    top_n_foods: int = Field(default=8, ge=1, le=25)


class RiskReportResponse(BaseModel):
    disease: DiseaseType
    predicted_class: int
    risk_score: float
    risk_level: Literal["low", "moderate", "high"]
    model_name: str
    thresholds: Dict[str, float]
    top_factors: list[Dict[str, Any]]
    plan: Dict[str, Any]
    recommended_foods: list[Dict[str, Any]]
    report_text: str


class RiskReportRawRequest(BaseModel):
    disease: DiseaseType
    raw_inputs: Dict[str, Any] = Field(default_factory=dict)
    profile: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    top_n_foods: int = Field(default=8, ge=1, le=25)


class RiskReportRawResponse(RiskReportResponse):
    transformed_features: Dict[str, Any]
    raw_inputs: Dict[str, Any]
    validation_warnings: list[str] = Field(default_factory=list)
