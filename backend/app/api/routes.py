from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from ..core.settings import settings
from ..models.schema import (
    AppliedOverride,
    BaselineResponse,
    DataSourceNotice,
    DeltaResponse,
    DeltaShapBlock,
    DriverItem,
    ExceededFeature,
    HealthResponse,
    InteractiveForecastRequest,
    InteractiveForecastResponse,
    ModelInfoResponse,
    PredictionBlock,
    ResponseMeta,
    RuntimeStatusResponse,
    LocationOption,
    RunDetail,
    RunMeta,
    RunSummary,
    ScenarioCard,
    ScenarioResponse,
    ShapBlock,
)
from ..services.data_client import DataClient
from ..services.explainer import ExplainerService
from ..services.feature_builder import FeatureBuilder
from ..services.health_engine import HealthEngine
from ..services.history_assembler import HistoryAssembler
from ..services.baseline_context import BaselineContextService
from ..services.location_catalog import LocationCatalog
from ..services.model_runner import ModelRunner
from ..services.run_logger import RunLogger
from ..services.scenario_engine import ScenarioEngine

router = APIRouter()

data_client = DataClient()
history_assembler = HistoryAssembler()
feature_builder = FeatureBuilder()
scenario_engine = ScenarioEngine()
model_runner = ModelRunner()
explainer = ExplainerService()
health_engine = HealthEngine()
# Do not make API startup depend on SQLite when run logging is disabled.
run_logger_init_error: str | None = None
if settings.ENABLE_RUN_LOGGING:
    try:
        run_logger = RunLogger(settings.SQLITE_PATH)
    except Exception as exc:
        run_logger = None
        run_logger_init_error = f"{exc.__class__.__name__}: {exc}"
else:
    run_logger = None
baseline_context_service = BaselineContextService(data_client=data_client, history_assembler=history_assembler)
location_catalog = LocationCatalog()


def _model_dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    return obj.dict(by_alias=True)


def _model_validate(model_cls, payload: Dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _normalize_custom_overrides_for_model(overrides: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """
    Normalize user-facing custom override payload into model feature representation.

    Current user contract keeps pressure in hPa for readability; model features use
    the kPa-equivalent pressure representation used by the active final model metadata.
    """
    out = dict(overrides or {})
    pressure_converted = False
    if "pressure" in out and out.get("pressure") is not None:
        try:
            pressure_hpa = float(out["pressure"])
        except Exception:
            pressure_hpa = None
        if pressure_hpa is not None and pressure_hpa >= 200.0:
            out["pressure"] = float(pressure_hpa * 0.1)
            pressure_converted = True
    return out, pressure_converted


def _run_logger_status_payload() -> Dict[str, Any]:
    enabled = bool(settings.ENABLE_RUN_LOGGING)
    available = bool(enabled and run_logger is not None)
    status = "enabled" if available else ("disabled" if not enabled else "degraded")
    note = None
    if not enabled:
        note = "Run logging disabled by SKYNET_ENABLE_RUN_LOGGING=0."
    elif run_logger is None:
        note = "Run logging requested, but logger initialization failed."
        if run_logger_init_error:
            note = f"{note} {run_logger_init_error}"
    return {
        "enabled": enabled,
        "available": available,
        "status": status,
        "note": note,
    }


def runtime_status_payload() -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}
    ready = True

    meta = None
    try:
        meta = model_runner.get_meta()
        checks["model_meta_loaded"] = True
    except Exception as exc:
        checks["model_meta_loaded"] = False
        details["model_meta_error"] = f"{exc.__class__.__name__}: {exc}"
        ready = False

    if meta is not None:
        features = list(meta.get("features", []))
        checks["feature_schema_present"] = bool(features)
        details["features_count"] = int(len(features))
        if not features:
            ready = False

        model_path = meta.get("model_path")
        model_present = False
        model_path_resolved = None
        if isinstance(model_path, str) and model_path.strip():
            p = Path(model_path.strip())
            if not p.is_absolute():
                p = model_runner.model_meta_path.parent / p
            model_path_resolved = str(p.resolve())
            model_present = bool(p.exists())
        else:
            # Backward compatibility for embedded model objects.
            model_present = bool(meta.get("model") is not None)

        checks["model_artifact_present"] = bool(model_present)
        details["model_artifact_path"] = model_path_resolved
        if not model_present:
            ready = False

    logger_state = _run_logger_status_payload()
    details["run_logging_enabled"] = bool(logger_state["enabled"])
    details["run_logging_available"] = bool(logger_state["available"])
    details["run_logging_status"] = str(logger_state["status"])
    if logger_state.get("note"):
        details["run_logging_note"] = str(logger_state["note"])
    checks["run_logging_ready_or_optional"] = bool(
        (not logger_state["enabled"]) or logger_state["available"]
    )

    return {"ready": bool(ready), "checks": checks, "details": details}


@router.get("/scenarios", response_model=list[ScenarioCard])
def get_scenarios():
    cards = scenario_engine.list_cards()
    return [_model_validate(ScenarioCard, c) for c in cards]


@router.get("/locations", response_model=list[LocationOption])
def get_locations():
    rows = location_catalog.list_locations()
    return [_model_validate(LocationOption, r) for r in rows]


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    meta = model_runner.get_meta()
    feature_quantiles = meta.get("feature_quantiles", {}) or {}
    global_shap_mean_abs = meta.get("global_shap_mean_abs", {}) or {}
    feature_importance = model_runner.feature_importance(top_n=15)
    metrics_raw = meta.get("metrics_test") or {}

    historical_metrics = None
    if isinstance(metrics_raw, dict):
        historical_metrics = {
            "mae": float(metrics_raw["mae"]) if "mae" in metrics_raw else None,
            "rmse": float(metrics_raw["rmse"]) if "rmse" in metrics_raw else None,
            "r2": float(metrics_raw["r2"]) if "r2" in metrics_raw else None,
            "mbe": float(metrics_raw["mbe"]) if "mbe" in metrics_raw else None,
            "directional_acc_pct": float(metrics_raw["directional_acc_pct"]) if "directional_acc_pct" in metrics_raw else None,
        }

    preview = {}
    preprocessing = meta.get("preprocessing", {}) or {}
    scaling_applied = bool(preprocessing.get("scaling_applied", False))
    scaling_note = str(
        preprocessing.get(
            "scaling_note",
            "No feature scaling is applied. This is intentional because the runtime model is tree-based XGBoost, "
            "which is generally scale-insensitive for split decisions.",
        )
    )
    outlier_note = (
        "Training clips PM2.5 at q01-q99, stores feature quantiles, and runtime scenarios/custom edits are "
        "quantile-bounded. Current exogenous extremes are flagged in health diagnostics."
    )
    outlier_policy = preprocessing.get("outlier_policy")
    if isinstance(outlier_policy, dict) and outlier_policy:
        parts = []
        if outlier_policy.get("pm25_training_clip"):
            parts.append(f"PM2.5 training clip: {outlier_policy['pm25_training_clip']}")
        if outlier_policy.get("feature_quantiles"):
            parts.append(f"feature quantiles: {outlier_policy['feature_quantiles']}")
        if outlier_policy.get("runtime_clamping"):
            parts.append(f"runtime clamping: {outlier_policy['runtime_clamping']}")
        if parts:
            outlier_note = "; ".join(parts)

    preview_features = ["PM10", "NO2", "CO", "temperature", "humidity", "wind_speed", "pressure", "O3", "SO2"]
    preview_keys = ["q01", "q05", "q25", "q50", "q75", "q95", "q99"]
    for f in preview_features:
        q = feature_quantiles.get(f)
        if not isinstance(q, dict):
            continue
        row = {}
        for k in preview_keys:
            if k in q:
                try:
                    row[k] = float(q[k])
                except Exception:
                    continue
        if row:
            preview[f] = row

    runtime_status = runtime_status_payload()
    logger_state = _run_logger_status_payload()

    payload = {
        "model_version": model_runner.model_version(),
        "schema_version": int(meta.get("schema_version", 1)),
        "target_type": str(meta.get("target_type", "delta")),
        "features_count": int(len(meta.get("features", []))),
        "scaling_applied": scaling_applied,
        "scaling_note": scaling_note,
        "outlier_note": outlier_note,
        "has_feature_quantiles": bool(feature_quantiles),
        "has_global_shap": bool(global_shap_mean_abs),
        "bounds_preview": preview,
        "feature_importance": feature_importance,
        "historical_test_metrics": historical_metrics,
        "run_logging_enabled": bool(logger_state["enabled"]),
        "run_logging_available": bool(logger_state["available"]),
        "run_logging_status": str(logger_state["status"]),
        "run_logging_note": logger_state.get("note"),
        "model_ready": bool(runtime_status.get("ready", False)),
        "training_lineage": meta.get("training_lineage"),
    }
    return _model_validate(ModelInfoResponse, payload)


@router.get("/runtime-status", response_model=RuntimeStatusResponse)
def get_runtime_status():
    return _model_validate(RuntimeStatusResponse, runtime_status_payload())


@router.post("/forecast/interactive", response_model=InteractiveForecastResponse)
def forecast_interactive(req: InteractiveForecastRequest):
    try:
        meta = model_runner.get_meta()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model metadata: {e}")

    forecast_mode = str(req.forecast_mode or "live")
    baseline_source = "live_api"
    live_data_used = True
    mode_notes = []
    req_lat = float(req.location.lat)
    req_lon = float(req.location.lon)
    req_name = req.location.name

    location_id = str(req.location.location_id or "").strip()
    if location_id:
        row = location_catalog.get_by_id(location_id)
        if row is None:
            raise HTTPException(status_code=400, detail=f"Unknown location_id: {location_id}")
        req_lat = float(row["lat"])
        req_lon = float(row["lon"])
        req_name = str(row.get("name") or req_name or "")
        mode_notes.append(f"Location resolved from catalog ID '{location_id}'.")
        if location_id != "haikou_cn":
            mode_notes.append(
                "Model was trained on Haikou-stage data; non-Haikou runs are exploratory and not externally validated."
            )
    else:
        # Manual coordinates are allowed for demo exploration.
        mode_notes.append(
            "Model was trained on Haikou-stage data; manual-coordinate runs are exploratory and not externally validated."
        )

    if forecast_mode == "live":
        try:
            aq_hist, w_hist = data_client.fetch_history(
                lat=req_lat,
                lon=req_lon,
                hours=req.options.history_hours_target,
                timezone="auto",
            )
            aq_cur, w_cur = data_client.fetch_current(
                lat=req_lat,
                lon=req_lon,
                timezone="auto",
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Live data fetch failed: {e}. "
                    "You can continue with Custom What-If Forecast, which uses a baseline history context plus your edited current conditions."
                ),
            )

        hist_df, hist_stats = history_assembler.assemble(
            aq_hist=aq_hist,
            w_hist=w_hist,
            target_hours=req.options.history_hours_target,
        )
        mode_notes.append("Live forecast uses Open-Meteo history and current conditions.")
    else:
        ctx = baseline_context_service.get_custom_context(
            lat=req_lat,
            lon=req_lon,
            target_hours=req.options.history_hours_target,
            meta=meta,
            timezone="auto",
        )
        hist_df = ctx.history_df
        hist_stats = ctx.history_stats
        aq_cur = ctx.aq_cur
        w_cur = ctx.w_cur
        baseline_source = ctx.source
        live_data_used = bool(ctx.live_data_used)
        mode_notes.extend(list(ctx.notes))

    if hist_df.empty:
        raise HTTPException(status_code=400, detail="No usable historical data available for this forecast mode.")

    feat = feature_builder.build(meta=meta, history_df=hist_df, aq_cur=aq_cur, w_cur=w_cur)

    baseline_pred = model_runner.predict(
        X=feat.X,
        base_lag1=feat.base_lag1,
        current_pm25=feat.current_pm25,
    )

    if forecast_mode == "custom":
        requested_impact_mode = str(getattr(req, "custom_impact_mode", "conservative") or "conservative")
        custom_overrides = _model_dump(req.custom_overrides) if req.custom_overrides else {}
        custom_overrides = {k: v for k, v in (custom_overrides or {}).items() if v is not None}
        custom_overrides_model, pressure_converted = _normalize_custom_overrides_for_model(custom_overrides)
        scenario_X, applied_overrides, ood_events, ood_context, impact_preview, resolved_impact_mode = scenario_engine.apply_value_overrides(
            overrides=custom_overrides_model,
            baseline_X=feat.X,
            meta=meta,
            impact_mode=requested_impact_mode,
            ood_opts=_model_dump(req.options.ood),
            return_context=True,
            include_preview=True,
        )
        scenario_id = "custom_what_if"
        if not custom_overrides:
            mode_notes.append("No override values were provided; scenario forecast matches baseline context.")
        else:
            mode_notes.append(
                f"Custom What-If forecast used baseline context with user-edited current conditions ({resolved_impact_mode.replace('_', ' ')} mode)."
            )
            if pressure_converted:
                mode_notes.append("Pressure override entered in hPa was converted internally to model pressure representation.")
    else:
        impact_preview = None
        resolved_impact_mode = None
        try:
            scenario_X, applied_overrides, ood_events, scenario_id, ood_context = scenario_engine.apply(
                scenario=req.scenario,
                baseline_X=feat.X,
                meta=meta,
                ood_opts=_model_dump(req.options.ood),
                return_context=True,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    direction_limited_count = int(sum(1 for row in applied_overrides if bool(row.get("direction_limited"))))
    if direction_limited_count > 0:
        mode_notes.append(
            f"{direction_limited_count} intervention(s) were direction-limited by plausibility bounds; requested intent was not silently inverted."
        )

    scenario_pred = model_runner.predict(
        X=scenario_X,
        base_lag1=feat.base_lag1,
        current_pm25=feat.current_pm25,
    )

    base_top = []
    scn_top = []
    delta_rows = []
    base_summary_text = ""
    scenario_summary_text = ""
    base_plain_lines = []
    scenario_plain_lines = []
    delta_plain_lines = []
    delta_summary = ""
    target_type = str(meta.get("target_type", "delta")).lower()
    if target_type == "delta":
        explain_target_space = "delta_pm25_t_plus_1"
        explain_target_note = (
            "Contribution scores explain the model's next-hour delta forecast signal. "
            "Final PM2.5 is reconstructed from baseline lag + predicted delta (+ bias correction)."
        )
    else:
        explain_target_space = "pm25_t_plus_1"
        explain_target_note = "Contribution scores explain the model output signal used directly as next-hour PM2.5."
    base_explain_meta = {
        "method": None,
        "base_value": None,
        "contrib_sum": None,
        "reconstructed_signal": None,
        "prediction_signal": None,
        "additivity_error": None,
        "additivity_ok": None,
        "additivity_tolerance": None,
        "prediction_alignment_error": None,
        "prediction_alignment_ok": None,
    }
    scenario_explain_meta = {
        "method": None,
        "base_value": None,
        "contrib_sum": None,
        "reconstructed_signal": None,
        "prediction_signal": None,
        "additivity_error": None,
        "additivity_ok": None,
        "additivity_tolerance": None,
        "prediction_alignment_error": None,
        "prediction_alignment_ok": None,
    }

    try:
        model_obj = meta.get("model")
        base_shap_s = explainer.shap_series(model_obj, feat.X)
        scn_shap_s = explainer.shap_series(model_obj, scenario_X)
        base_explain_meta = explainer.explanation_meta(
            model_obj,
            feat.X,
            base_shap_s,
            prediction_signal=float(baseline_pred.get("pred_space", baseline_pred.get("delta_pm25_t_plus_1", 0.0))),
        )
        scenario_explain_meta = explainer.explanation_meta(
            model_obj,
            scenario_X,
            scn_shap_s,
            prediction_signal=float(scenario_pred.get("pred_space", scenario_pred.get("delta_pm25_t_plus_1", 0.0))),
        )

        base_top = explainer.top_drivers(base_shap_s, feat.X, req.options.top_k_drivers)
        scn_top = explainer.top_drivers(scn_shap_s, scenario_X, req.options.top_k_drivers)
        delta_rows, delta_summary, delta_plain_lines = explainer.delta_shap(
            base_shap_s, scn_shap_s, req.options.top_k_drivers
        )
        base_summary_text = explainer.summary_text(base_top, "Baseline")
        scenario_summary_text = explainer.summary_text(scn_top, "Scenario")
        base_plain_lines = explainer.plain_language_from_top(base_top)
        scenario_plain_lines = explainer.plain_language_from_top(scn_top)
    except Exception as exc:
        fallback_text = (
            "Explainability is temporarily unavailable for this run. "
            f"Forecast values are still valid. ({exc.__class__.__name__})"
        )
        base_explain_meta = {
            "method": "unavailable",
            "base_value": None,
            "contrib_sum": None,
            "reconstructed_signal": None,
            "prediction_signal": None,
            "additivity_error": None,
            "additivity_ok": None,
            "additivity_tolerance": None,
            "prediction_alignment_error": None,
            "prediction_alignment_ok": None,
        }
        scenario_explain_meta = {
            "method": "unavailable",
            "base_value": None,
            "contrib_sum": None,
            "reconstructed_signal": None,
            "prediction_signal": None,
            "additivity_error": None,
            "additivity_ok": None,
            "additivity_tolerance": None,
            "prediction_alignment_error": None,
            "prediction_alignment_ok": None,
        }
        base_summary_text = fallback_text
        scenario_summary_text = fallback_text
        delta_summary = fallback_text
        base_plain_lines = [fallback_text]
        scenario_plain_lines = [fallback_text]
        delta_plain_lines = [fallback_text]

    request_id = req.request_id or str(uuid4())
    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    baseline_timestamp = None
    try:
        if getattr(feat, "observed_timestamp", None) is not None:
            baseline_timestamp = feat.observed_timestamp.isoformat()
    except Exception:
        baseline_timestamp = None

    baseline_block = _model_validate(
        BaselineResponse,
        {
            "inputs_snapshot": {
                "pm25_current": float(feat.current_pm25),
                "wind_speed": float(feat.current_aux.get("wind_speed", 0.0)),
                "humidity": float(feat.current_aux.get("humidity", 0.0)),
                "temperature": float(feat.current_aux.get("temperature", 0.0)),
            },
            "prediction": {
                "delta_pm25_t_plus_1": float(baseline_pred["delta_pm25_t_plus_1"]),
                "pm25_t_plus_1": float(baseline_pred["pm25_t_plus_1"]),
            },
            "shap": {
                "top_drivers": base_top,
                "summary_text": base_summary_text,
                "plain_language": base_plain_lines,
                "method": base_explain_meta.get("method"),
                "target_space": explain_target_space,
                "target_space_note": explain_target_note,
                "base_value": base_explain_meta.get("base_value"),
                "contrib_sum": base_explain_meta.get("contrib_sum"),
                "reconstructed_signal": base_explain_meta.get("reconstructed_signal"),
                "prediction_signal": base_explain_meta.get("prediction_signal"),
                "additivity_error": base_explain_meta.get("additivity_error"),
                "additivity_ok": base_explain_meta.get("additivity_ok"),
                "additivity_tolerance": base_explain_meta.get("additivity_tolerance"),
                "prediction_alignment_error": base_explain_meta.get("prediction_alignment_error"),
                "prediction_alignment_ok": base_explain_meta.get("prediction_alignment_ok"),
            },
        },
    )

    # Custom What-If uses direct manual overrides, so macro-style intensity is not
    # a meaningful scale. Keep custom intensity neutral to avoid false "max" semantics.
    scenario_intensity = int(req.scenario.intensity) if forecast_mode == "live" else 0
    if forecast_mode == "custom":
        scenario_mode = "manual_custom"
    elif scenario_id == "guided_intervention":
        scenario_mode = "guided_intervention"
    elif scenario_id == "baseline":
        scenario_mode = "baseline"
    else:
        scenario_mode = "macro"

    health_payload = health_engine.build(
        history_stats=hist_stats,
        imputed_features=feat.imputed_features,
        total_features=len(meta.get("features", [])),
        ood_events=ood_events,
        ood_opts=_model_dump(req.options.ood),
        ood_context=ood_context,
        imputed_feature_names=feat.imputed_feature_names,
        extreme_current_events=feat.extreme_current_events,
        applied_overrides=applied_overrides,
        explainability_meta={
            "method": base_explain_meta.get("method"),
            "additivity_ok": base_explain_meta.get("additivity_ok"),
            "prediction_alignment_ok": base_explain_meta.get("prediction_alignment_ok"),
        },
    )
    health_payload["uncertainty"] = model_runner.uncertainty_guidance(
        baseline_pm25=float(baseline_pred["pm25_t_plus_1"]),
        scenario_pm25=float(scenario_pred["pm25_t_plus_1"]),
        reliability_score=float(health_payload.get("quality_score", 0.0)),
        scenario_mode=scenario_mode,
    )

    scenario_block = _model_validate(
        ScenarioResponse,
        {
            "scenario_id": scenario_id,
            "scenario_mode": scenario_mode,
            "intensity": scenario_intensity,
            "applied_overrides": applied_overrides,
            "custom_impact_mode": resolved_impact_mode if forecast_mode == "custom" else None,
            "impact_preview": impact_preview if forecast_mode == "custom" else None,
            "prediction": {
                "delta_pm25_t_plus_1": float(scenario_pred["delta_pm25_t_plus_1"]),
                "pm25_t_plus_1": float(scenario_pred["pm25_t_plus_1"]),
            },
            "shap": {
                "top_drivers": scn_top,
                "summary_text": scenario_summary_text,
                "plain_language": scenario_plain_lines,
                "method": scenario_explain_meta.get("method"),
                "target_space": explain_target_space,
                "target_space_note": explain_target_note,
                "base_value": scenario_explain_meta.get("base_value"),
                "contrib_sum": scenario_explain_meta.get("contrib_sum"),
                "reconstructed_signal": scenario_explain_meta.get("reconstructed_signal"),
                "prediction_signal": scenario_explain_meta.get("prediction_signal"),
                "additivity_error": scenario_explain_meta.get("additivity_error"),
                "additivity_ok": scenario_explain_meta.get("additivity_ok"),
                "additivity_tolerance": scenario_explain_meta.get("additivity_tolerance"),
                "prediction_alignment_error": scenario_explain_meta.get("prediction_alignment_error"),
                "prediction_alignment_ok": scenario_explain_meta.get("prediction_alignment_ok"),
            },
        },
    )

    delta_block = _model_validate(
        DeltaResponse,
        {
            "pm25_change": float(scenario_pred["pm25_t_plus_1"] - baseline_pred["pm25_t_plus_1"]),
            "delta_shap": {
                "top_changes": delta_rows,
                "summary_text": delta_summary,
                "plain_language": delta_plain_lines,
            },
        },
    )

    response_payload = {
        "meta": {
            "request_id": request_id,
            "generated_at": generated_at,
            "model_version": model_runner.model_version(),
            "forecast_mode": forecast_mode,
            "custom_impact_mode": resolved_impact_mode if forecast_mode == "custom" else None,
            "baseline_source": baseline_source,
            "baseline_timestamp": baseline_timestamp,
            "live_data_used": bool(live_data_used),
            "overrides_applied": bool(len(applied_overrides) > 0),
            "mode_note": " ".join([n for n in mode_notes if n]).strip() or None,
            "location_name": req_name,
            "location_lat": float(req_lat),
            "location_lon": float(req_lon),
            "location_id": location_id or None,
            "data_sources": {
                "open_meteo": {
                    "weather": bool(live_data_used),
                    "air_quality": bool(live_data_used),
                    "terms_notice": (
                        "Open-Meteo free non-commercial usage and limits apply."
                        if live_data_used
                        else "Live Open-Meteo was not used for this run; fallback baseline context was applied."
                    ),
                }
            },
        },
        "health": health_payload,
        "baseline": _model_dump(baseline_block),
        "scenario": _model_dump(scenario_block),
        "delta": _model_dump(delta_block),
        "run": {"persisted": False, "run_id": None},
    }

    if settings.ENABLE_RUN_LOGGING and run_logger is not None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        try:
            run_logger.save_run(
                run_id=run_id,
                created_at=generated_at,
                location={"lat": req_lat, "lon": req_lon, "name": req_name, "location_id": location_id or None},
                scenario_request=_model_dump(req),
                response_json=response_payload,
                model_version=model_runner.model_version(),
            )
            response_payload["run"] = {"persisted": True, "run_id": run_id}
        except Exception:
            response_payload["run"] = {"persisted": False, "run_id": None}

    return _model_validate(InteractiveForecastResponse, response_payload)


@router.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = Query(default=50, ge=1, le=200)):
    if not settings.ENABLE_RUN_LOGGING or run_logger is None:
        return []
    rows = run_logger.list_runs(limit=limit)
    return [_model_validate(RunSummary, r) for r in rows]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str):
    if not settings.ENABLE_RUN_LOGGING or run_logger is None:
        raise HTTPException(status_code=404, detail="Run logging is disabled")
    row = run_logger.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _model_validate(RunDetail, row)
