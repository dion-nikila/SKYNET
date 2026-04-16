from typing import Any, Dict, List, Optional, Literal
import math
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic import model_validator


class Location(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    name: Optional[str] = None
    location_id: Optional[str] = None


class TimeRequest(BaseModel):
    mode: Literal["now"] = "now"


class OodOptions(BaseModel):
    soft_q: float = Field(default=0.05, ge=0.0, le=0.49)
    hard_q: float = Field(default=0.01, ge=0.0, le=0.49)


class OptionsRequest(BaseModel):
    history_hours_target: int = Field(
        default=72,
        ge=24,
        le=240,
        description=(
            "Operational history window in hours (default 72). "
            "When available history is below 168h, weekly lag-derived features may be imputed from trained defaults."
        ),
    )
    top_k_drivers: int = Field(default=6, ge=3, le=12)
    ood: OodOptions = Field(default_factory=OodOptions)


class CustomScenarioItem(BaseModel):
    category: Literal["wind", "humidity", "temperature", "emission_proxy"]
    direction: Literal["increase", "decrease"]
    magnitude: Literal["small", "medium", "large"]


class ScenarioRequest(BaseModel):
    type: Literal["macro", "custom"]
    scenario_id: Optional[str] = None
    intensity: int = Field(default=50, ge=0, le=100)
    items: Optional[List[CustomScenarioItem]] = None

    @model_validator(mode="after")
    def _validate_guided_items(self):
        # Guided interventions use type="custom" + items; enforce one row per
        # category so crafted payloads cannot stack duplicate category effects.
        if self.type != "custom" or not self.items:
            return self
        categories = [str(item.category) for item in self.items]
        if len(set(categories)) != len(categories):
            raise ValueError("Guided interventions must not contain duplicate categories.")
        return self


class CustomWhatIfOverrides(BaseModel):
    PM10: Optional[float] = Field(default=None, ge=0.0, le=1000.0)
    NO2: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    CO: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    temperature: Optional[float] = Field(default=None, ge=-50.0, le=60.0)
    humidity: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    wind_speed: Optional[float] = Field(default=None, ge=0.0, le=80.0)
    pressure: Optional[float] = Field(
        default=None,
        ge=850.0,
        le=1100.0,
        description="User-facing pressure input in hPa; backend converts to model pressure representation.",
    )
    O3: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    SO2: Optional[float] = Field(default=None, ge=0.0, le=500.0)

    @field_validator(
        "PM10",
        "NO2",
        "CO",
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "O3",
        "SO2",
        mode="before",
    )
    @classmethod
    def _finite_number_only(cls, value):
        if value is None:
            return value
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("Override values must be finite numbers.")
        return v


class InteractiveForecastRequest(BaseModel):
    request_id: Optional[str] = None
    forecast_mode: Literal["live", "custom"] = "live"
    custom_impact_mode: Literal["conservative", "stronger_realistic"] = "conservative"
    location: Location
    time: TimeRequest = Field(default_factory=TimeRequest)
    scenario: ScenarioRequest
    custom_overrides: Optional[CustomWhatIfOverrides] = None
    options: OptionsRequest = Field(default_factory=OptionsRequest)


class ExceededFeature(BaseModel):
    feature: str
    feature_label: Optional[str] = None
    value: float
    bound: str
    q01: Optional[float] = None
    q05: Optional[float] = None
    q95: Optional[float] = None
    q99: Optional[float] = None
    severity: Literal["soft", "hard"]


class OodHealth(BaseModel):
    method: str
    soft_range: Dict[str, float]
    hard_range: Dict[str, float]
    requested_soft_range: Optional[Dict[str, float]] = None
    requested_hard_range: Optional[Dict[str, float]] = None
    notes: List[str] = Field(default_factory=list)
    flag: bool
    score: float
    soft_count: int = 0
    hard_count: int = 0
    features_exceeded: List[ExceededFeature]


class HistoryHealth(BaseModel):
    target_hours: int
    available_hours: int
    used_hours: int
    coverage_ratio: float = 0.0


class GapHealth(BaseModel):
    gap_count: int
    largest_gap_hours: int


class ImputationHealth(BaseModel):
    imputed_features: int
    total_features: int
    ratio: float
    features: List[str] = Field(default_factory=list)


class FallbackHealth(BaseModel):
    level: int
    label: str
    notes: str


class ExtremeInputEvent(BaseModel):
    feature: str
    value: float
    q01: float
    q99: float
    side: Literal["below_q01", "above_q99"]


class ExtremeInputHealth(BaseModel):
    method: str
    count: int = 0
    flag: bool = False
    notes: List[str] = Field(default_factory=list)
    events: List[ExtremeInputEvent] = Field(default_factory=list)


class HealthResponse(BaseModel):
    history: HistoryHealth
    gaps: GapHealth
    imputation: ImputationHealth
    fallback: FallbackHealth
    ood: OodHealth
    extreme_inputs: ExtremeInputHealth
    quality_score: float
    quality_label: str
    reliability: Optional["ReliabilityGuidance"] = None
    uncertainty: Optional["UncertaintyGuidance"] = None


class ReliabilityComponent(BaseModel):
    name: str
    score: float
    weight: float
    rationale: str


class ReliabilityGuidance(BaseModel):
    method: str
    score: float
    label: str
    components: List[ReliabilityComponent] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class UncertaintyBand(BaseModel):
    coverage_pct: int
    lower: float
    upper: float
    width: float


class UncertaintyGuidance(BaseModel):
    method: str
    available: bool
    note: str
    caveats: List[str] = Field(default_factory=list)
    calibration_sample_size: Optional[int] = None
    baseline_bands: List[UncertaintyBand] = Field(default_factory=list)
    scenario_bands: List[UncertaintyBand] = Field(default_factory=list)
    scenario_inflation: Optional[float] = None


class DriverItem(BaseModel):
    feature: str
    feature_label: Optional[str] = None
    value: float
    shap: float
    direction: Literal["up", "down"]
    plain_text: Optional[str] = None


class ShapBlock(BaseModel):
    top_drivers: List[DriverItem]
    summary_text: str
    plain_language: List[str] = Field(default_factory=list)
    method: Optional[str] = None
    target_space: Optional[str] = None
    target_space_note: Optional[str] = None
    base_value: Optional[float] = None
    contrib_sum: Optional[float] = None
    reconstructed_signal: Optional[float] = None
    prediction_signal: Optional[float] = None
    additivity_error: Optional[float] = None
    additivity_ok: Optional[bool] = None
    additivity_tolerance: Optional[float] = None
    prediction_alignment_error: Optional[float] = None
    prediction_alignment_ok: Optional[bool] = None


class DeltaShapItem(BaseModel):
    feature: str
    feature_label: Optional[str] = None
    baseline_shap: float
    scenario_shap: float
    delta_shap: float
    sign_flip: bool
    plain_text: Optional[str] = None


class DeltaShapBlock(BaseModel):
    top_changes: List[DeltaShapItem]
    summary_text: str
    plain_language: List[str] = Field(default_factory=list)


class PredictionBlock(BaseModel):
    delta_pm25_t_plus_1: float
    pm25_t_plus_1: float


class BaselineResponse(BaseModel):
    inputs_snapshot: Dict[str, float]
    prediction: PredictionBlock
    shap: ShapBlock


class AppliedOverride(BaseModel):
    category: str
    feature: str
    from_value: float = Field(alias="from")
    to_value: float = Field(alias="to")
    clamped: bool
    reason: str
    requested_direction: Optional[Literal["increase", "decrease", "unchanged"]] = None
    effective_direction: Optional[Literal["increase", "decrease", "unchanged"]] = None
    direction_limited: bool = False


class ImpactPreview(BaseModel):
    level: Literal["low", "medium", "high"]
    score: float
    note: str
    basis: Literal["heuristic_estimate"] = "heuristic_estimate"
    factors: List[str] = Field(default_factory=list)


class ScenarioResponse(BaseModel):
    scenario_id: str
    scenario_mode: Literal["macro", "guided_intervention", "manual_custom", "baseline"] = "baseline"
    intensity: int
    applied_overrides: List[AppliedOverride]
    prediction: PredictionBlock
    shap: ShapBlock
    custom_impact_mode: Optional[Literal["conservative", "stronger_realistic"]] = None
    impact_preview: Optional[ImpactPreview] = None


class DeltaResponse(BaseModel):
    pm25_change: float
    delta_shap: DeltaShapBlock


class DataSourceNotice(BaseModel):
    weather: bool = True
    air_quality: bool = True
    terms_notice: str


class ResponseMeta(BaseModel):
    request_id: str
    generated_at: str
    model_version: str
    forecast_mode: Literal["live", "custom"] = "live"
    custom_impact_mode: Optional[Literal["conservative", "stronger_realistic"]] = None
    baseline_source: str = "live_api"
    baseline_timestamp: Optional[str] = None
    live_data_used: bool = True
    overrides_applied: bool = False
    mode_note: Optional[str] = None
    location_name: Optional[str] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    location_id: Optional[str] = None
    data_sources: Dict[str, DataSourceNotice]


class RunMeta(BaseModel):
    persisted: bool
    run_id: Optional[str] = None


class InteractiveForecastResponse(BaseModel):
    meta: ResponseMeta
    health: HealthResponse
    baseline: BaselineResponse
    scenario: ScenarioResponse
    delta: DeltaResponse
    run: RunMeta


class ScenarioCard(BaseModel):
    scenario_id: str
    title: str
    description: str
    category: str
    default_intensity: int
    knobs: List[str]


class LocationOption(BaseModel):
    location_id: str
    name: str
    country: str
    region: str
    lat: float
    lon: float


class FeatureImportanceItem(BaseModel):
    feature: str
    feature_label: str
    score: float
    pct: float


class ModelTestMetrics(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mbe: Optional[float] = None
    directional_acc_pct: Optional[float] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    schema_version: int
    target_type: str
    features_count: int
    scaling_applied: bool = False
    scaling_note: Optional[str] = None
    outlier_note: Optional[str] = None
    has_feature_quantiles: bool
    has_global_shap: bool
    bounds_preview: Dict[str, Dict[str, float]]
    feature_importance: List[FeatureImportanceItem] = Field(default_factory=list)
    historical_test_metrics: Optional[ModelTestMetrics] = None
    run_logging_enabled: bool = False
    run_logging_available: bool = False
    run_logging_status: Literal["enabled", "disabled", "degraded"] = "disabled"
    run_logging_note: Optional[str] = None
    model_ready: bool = True
    training_lineage: Optional[Dict[str, Any]] = None


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    scenario_id: str
    scenario_mode: Optional[Literal["macro", "guided_intervention", "manual_custom", "baseline"]] = None
    intensity: int
    pm25_change: float
    ood_flag: bool


class RunDetail(BaseModel):
    run_id: str
    created_at: str
    response_json: Dict


class RuntimeStatusResponse(BaseModel):
    ready: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)


# Explicit rebuild keeps forward-referenced health blocks stable across
# different import orders during tests/startup.
HealthResponse.model_rebuild()
