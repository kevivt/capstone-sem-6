from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from pydantic import ValidationError

from backend.model_registry import registry
from backend.schemas import CKDRawInput, DiabetesRawInput, HypertensionRawInput


@dataclass
class RawMappingResult:
    disease: str
    raw_inputs: Dict[str, Any]
    transformed_features: Dict[str, float]
    validation_warnings: list[str]


FIELD_METADATA: Dict[str, list[dict[str, Any]]] = {
    "ckd": [
        {"name": "age_years", "label": "Age (years)", "required": True},
        {"name": "sex", "label": "Sex (male/female)", "required": True},
        {"name": "body_mass_index", "label": "BMI", "required": True},
        {"name": "diagnosed_diabetes", "label": "Diagnosed diabetes (true/false)", "required": False},
        {"name": "diagnosed_hypertension", "label": "Diagnosed hypertension (true/false)", "required": False},
        {"name": "systolic_bp_mmhg", "label": "Systolic BP mmHg", "required": False},
        {"name": "diastolic_bp_mmhg", "label": "Diastolic BP mmHg", "required": False},
        {"name": "fasting_glucose_mg_dl", "label": "Fasting glucose mg/dL", "required": False},
    ],
    "hypertension": [
        {"name": "age_years", "label": "Age (years)", "required": True},
        {"name": "sex", "label": "Sex (male/female)", "required": True},
        {"name": "education_level", "label": "Education level (1-9)", "required": False},
        {"name": "current_smoker", "label": "Current smoker (true/false)", "required": False},
        {"name": "cigarettes_per_day", "label": "Cigarettes per day", "required": False},
        {"name": "prior_stroke", "label": "Prior stroke (true/false)", "required": False},
        {"name": "body_mass_index", "label": "BMI", "required": True},
        {"name": "diagnosed_diabetes", "label": "Diagnosed diabetes (true/false)", "required": False},
        {"name": "systolic_bp_mmhg", "label": "Systolic BP mmHg", "required": False},
        {"name": "diastolic_bp_mmhg", "label": "Diastolic BP mmHg", "required": False},
        {"name": "fasting_glucose_mg_dl", "label": "Fasting glucose mg/dL", "required": False},
    ],
    "diabetes": [
        {"name": "age_years", "label": "Age (years)", "required": True},
        {"name": "sex", "label": "Sex (male/female)", "required": True},
        {"name": "body_mass_index", "label": "BMI", "required": True},
        {"name": "current_smoker", "label": "Current smoker (true/false)", "required": False},
        {"name": "diagnosed_hypertension", "label": "Diagnosed hypertension (true/false)", "required": False},
        {"name": "systolic_bp_mmhg", "label": "Systolic BP mmHg", "required": False},
        {"name": "diastolic_bp_mmhg", "label": "Diastolic BP mmHg", "required": False},
        {"name": "fasting_glucose_mg_dl", "label": "Fasting glucose mg/dL", "required": False},
    ],
}


RAW_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "ckd": {
        "age_years": 56,
        "sex": "female",
        "body_mass_index": 27.4,
        "diagnosed_diabetes": True,
        "diagnosed_hypertension": True,
        "systolic_bp_mmhg": 142,
        "diastolic_bp_mmhg": 88,
        "fasting_glucose_mg_dl": 132,
    },
    "hypertension": {
        "age_years": 63,
        "sex": "male",
        "education_level": 5,
        "current_smoker": True,
        "cigarettes_per_day": 10,
        "prior_stroke": False,
        "body_mass_index": 31.2,
        "diagnosed_diabetes": True,
        "systolic_bp_mmhg": 154,
        "diastolic_bp_mmhg": 96,
        "fasting_glucose_mg_dl": 140,
    },
    "diabetes": {
        "age_years": 58,
        "sex": "male",
        "body_mass_index": 30.1,
        "current_smoker": False,
        "diagnosed_hypertension": True,
        "systolic_bp_mmhg": 146,
        "diastolic_bp_mmhg": 90,
        "fasting_glucose_mg_dl": 168,
    },
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _minmax(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return _clip01((float(value) - min_value) / (max_value - min_value))


def _binary(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


def _sex_to_male(value: str) -> float:
    return 1.0 if str(value).strip().lower() == "male" else 0.0


def get_raw_template(disease: str) -> Dict[str, Any]:
    if disease not in RAW_EXAMPLES:
        raise ValueError(f"Unsupported disease '{disease}'.")
    return {"disease": disease, "raw_inputs": RAW_EXAMPLES[disease]}


def get_raw_field_metadata(disease: str) -> list[dict[str, Any]]:
    if disease not in FIELD_METADATA:
        raise ValueError(f"Unsupported disease '{disease}'.")
    return FIELD_METADATA[disease]


def map_raw_payload(disease: str, raw_inputs: Dict[str, Any]) -> RawMappingResult:
    unknown_fields = _find_unknown_raw_fields(disease, raw_inputs)
    if disease == "ckd":
        result = _map_ckd(raw_inputs)
    elif disease == "hypertension":
        result = _map_hypertension(raw_inputs)
    elif disease == "diabetes":
        result = _map_diabetes(raw_inputs)
    else:
        raise ValueError(f"Unsupported disease '{disease}'.")

    if unknown_fields:
        result.validation_warnings.append(
            "Unknown raw-input fields were ignored: " + ", ".join(sorted(unknown_fields))
        )

    result.transformed_features = _align_to_artifact_feature_order(
        disease, result.transformed_features, result.validation_warnings
    )
    return result


def _find_unknown_raw_fields(disease: str, raw_inputs: Dict[str, Any]) -> set[str]:
    valid_names = {field["name"] for field in FIELD_METADATA.get(disease, [])}
    return {key for key in raw_inputs.keys() if key not in valid_names}


def _align_to_artifact_feature_order(
    disease: str,
    transformed_features: Dict[str, float],
    warnings: list[str],
) -> Dict[str, float]:
    registry.load_feature_metadata()
    expected_order = registry.features.get(disease, [])
    if not expected_order:
        warnings.append(
            "No artifact feature metadata found; using mapper output order as fallback."
        )
        return transformed_features

    aligned: Dict[str, float] = {}
    missing: list[str] = []
    for feature in expected_order:
        if feature in transformed_features:
            aligned[feature] = float(transformed_features[feature])
        else:
            aligned[feature] = 0.0
            missing.append(feature)

    extra = [k for k in transformed_features.keys() if k not in expected_order]
    if missing:
        warnings.append(
            "Mapper did not produce some expected model features; defaulted to 0.0: "
            + ", ".join(missing)
        )
    if extra:
        warnings.append(
            "Mapper produced extra features not used by deployed artifact: "
            + ", ".join(extra)
        )
    return aligned


def _map_ckd(raw_inputs: Dict[str, Any]) -> RawMappingResult:
    try:
        payload = CKDRawInput(**raw_inputs)
    except ValidationError as ex:
        raise ValueError(ex.json()) from ex

    warnings = [
        "The deployed CKD model was trained on a reduced survey-style feature set. "
        "sysBP, diaBP, glucose, prevalentHyp, and source were constant in training and are forced to 0.0 for model compatibility."
    ]
    features = {
        "age": _minmax(payload.age_years, 18, 80),
        "male": _sex_to_male(payload.sex),
        "BMI": _minmax(payload.body_mass_index, 12.0, 79.75),
        "sysBP": 0.0,
        "diaBP": 0.0,
        "glucose": 0.0,
        "diabetes": _binary(payload.diagnosed_diabetes),
        "prevalentHyp": 0.0,
        "source": 0.0,
    }
    return RawMappingResult("ckd", payload.model_dump(), features, warnings)


def _map_hypertension(raw_inputs: Dict[str, Any]) -> RawMappingResult:
    try:
        payload = HypertensionRawInput(**raw_inputs)
    except ValidationError as ex:
        raise ValueError(ex.json()) from ex

    warnings = [
        "The deployed hypertension model was trained on a reduced survey-style feature set. "
        "sysBP, diaBP, glucose, and source were constant in training and are forced to 0.0 for model compatibility."
    ]
    cigarettes = payload.cigarettes_per_day if payload.current_smoker else 0.0
    features = {
        "male": _sex_to_male(payload.sex),
        "age": _minmax(payload.age_years, 18, 80),
        "education": _minmax(payload.education_level, 1, 9),
        "currentSmoker": _binary(payload.current_smoker),
        "cigsPerDay": _minmax(cigarettes, 0, 99),
        "prevalentStroke": _binary(payload.prior_stroke),
        "BMI": _minmax(payload.body_mass_index, 12.0, 79.82),
        "sysBP": 0.0,
        "diaBP": 0.0,
        "glucose": 0.0,
        "diabetes": _binary(payload.diagnosed_diabetes),
        "source": 0.0,
    }
    return RawMappingResult("hypertension", payload.model_dump(), features, warnings)


def _map_diabetes(raw_inputs: Dict[str, Any]) -> RawMappingResult:
    try:
        payload = DiabetesRawInput(**raw_inputs)
    except ValidationError as ex:
        raise ValueError(ex.json()) from ex

    warnings = [
        "The deployed diabetes model was trained on a reduced survey-style feature set. "
        "sysBP, diaBP, glucose, and source were constant in training and are forced to 0.0 for model compatibility."
    ]
    features = {
        "age": _minmax(payload.age_years, 18, 80),
        "male": _sex_to_male(payload.sex),
        "BMI": _minmax(payload.body_mass_index, 12.0, 79.88),
        "sysBP": 0.0,
        "diaBP": 0.0,
        "glucose": 0.0,
        "prevalentHyp": _binary(payload.diagnosed_hypertension),
        "currentSmoker": _binary(payload.current_smoker),
        "source": 0.0,
    }
    return RawMappingResult("diabetes", payload.model_dump(), features, warnings)
