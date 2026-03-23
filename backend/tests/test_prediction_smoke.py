from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.api import routes
from backend.app.models.schema import InteractiveForecastRequest, ScenarioRequest
from backend.app.services.feature_builder import FeatureBuilder
from backend.app.services.model_runner import ModelRunner
from backend.app.services.scenario_engine import ScenarioEngine
from backend.app.services.run_logger import RunLogger


FEATURES_FOR_TEST = [
    "lag1",
    "lag24",
    "roll24",
    "NO2",
    "PM10",
    "wind_speed",
    "temperature",
    "humidity",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "is_weekend",
]


def _make_history(hours: int = 96):
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    idx = pd.date_range(end=end, periods=hours, freq="h")

    aq = pd.DataFrame(
        {
            "pm2_5": np.linspace(12, 30, hours),
            "pm10": np.linspace(18, 36, hours),
            "carbon_monoxide": np.linspace(420, 560, hours),
            "nitrogen_dioxide": np.linspace(12, 28, hours),
            "sulphur_dioxide": np.linspace(7, 10, hours),
            "ozone": np.linspace(44, 60, hours),
        },
        index=idx,
    )

    weather = pd.DataFrame(
        {
            "temperature_2m": np.linspace(25, 31, hours),
            "relative_humidity_2m": np.linspace(66, 82, hours),
            "surface_pressure": np.linspace(1007, 1011, hours),
            "wind_speed_10m": np.linspace(3.5, 8.0, hours),
        },
        index=idx,
    )
    return aq, weather


def _make_current(ts: datetime):
    aq_cur = {
        "time": ts.isoformat(),
        "pm2_5": 31.0,
        "pm10": 37.0,
        "carbon_monoxide": 575.0,
        "nitrogen_dioxide": 29.0,
        "sulphur_dioxide": 10.2,
        "ozone": 62.0,
        "us_aqi": 74,
        "european_aqi": 59,
    }
    w_cur = {
        "time": ts.isoformat(),
        "temperature_2m": 31.4,
        "relative_humidity_2m": 80.0,
        "surface_pressure": 1010.5,
        "wind_speed_10m": 4.2,
    }
    return aq_cur, w_cur


@dataclass
class _DummyModel:
    pred_value: float = 1.0

    def predict(self, _dmatrix):
        return np.array([self.pred_value], dtype=float)


class PredictionSmokeTests(unittest.TestCase):
    def test_explainer_pred_contrib_additivity_and_feature_order(self):
        row = pd.DataFrame({"wind_speed": [2.8], "PM10": [40.0], "lag1": [18.0]})
        contrib_series = pd.Series({"lag1": 0.33, "PM10": 0.47, "wind_speed": -0.20}, dtype=float)

        original_pred_contrib = routes.explainer._pred_contrib_row
        try:
            routes.explainer._pred_contrib_row = lambda _m, _x: {
                "series": contrib_series.copy(),
                "base_value": 1.0,
                "contrib_sum": 0.60,
                "reconstructed_signal": 1.60,
                "pred_margin": 1.6000001,
                "additivity_error": 1e-7,
            }

            contrib = routes.explainer.shap_series(_DummyModel(), row)
            meta = routes.explainer.explanation_meta(_DummyModel(), row, contrib, prediction_signal=1.60)

            self.assertEqual(meta.get("method"), "xgboost_pred_contribs")
            self.assertListEqual(list(contrib.index), ["lag1", "PM10", "wind_speed"])
            self.assertTrue(bool(meta.get("additivity_ok")))
            self.assertTrue(bool(meta.get("prediction_alignment_ok")))
            self.assertLessEqual(float(meta.get("additivity_error", 1.0)), float(meta.get("additivity_tolerance", 0.0)))
            self.assertLessEqual(float(meta.get("prediction_alignment_error", 1.0)), 1e-6)
        finally:
            routes.explainer._pred_contrib_row = original_pred_contrib

    def test_macro_scenarios_has_six_cards(self):
        cards = ScenarioEngine().list_cards()
        ids = {c["scenario_id"] for c in cards}
        self.assertGreaterEqual(len(cards), 6)
        self.assertIn("strong_dispersion", ids)
        self.assertIn("trapped_pollution", ids)
        self.assertIn("dust_resuspension", ids)
        self.assertIn("industrial_plume", ids)
        self.assertNotIn("heavy_rainstorm", ids)
        self.assertNotIn("stagnation", ids)
        self.assertNotIn("windy_dispersion", ids)

    def test_locations_endpoint_returns_catalog(self):
        rows = routes.get_locations()
        self.assertGreater(len(rows), 0)
        ids = {r.location_id for r in rows}
        self.assertEqual(rows[0].location_id, "haikou_cn")
        self.assertIn("colombo_lk", ids)
        self.assertIn("berlin_de", ids)
        self.assertIn("paris_fr", ids)

    def test_model_info_exposes_quantile_and_global_shap_flags(self):
        original_get_meta = routes.model_runner.get_meta
        original_feature_importance = routes.model_runner.feature_importance
        try:
            routes.model_runner.get_meta = lambda: {
                "schema_version": 2,
                "target_type": "delta",
                "features": ["lag1", "PM10"],
                "feature_quantiles": {
                    "lag1": {"q01": 1.0, "q05": 2.0, "q25": 3.0, "q50": 4.0, "q75": 5.0, "q95": 6.0, "q99": 7.0},
                    "PM10": {"q01": 10.0, "q05": 11.0, "q25": 12.0, "q50": 13.0, "q75": 14.0, "q95": 15.0, "q99": 16.0},
                },
                "global_shap_mean_abs": {"lag1": 0.2, "PM10": 0.3},
                "metrics_test": {"mae": 1.0, "rmse": 2.0, "r2": 0.5},
            }
            routes.model_runner.feature_importance = lambda top_n=15: [
                {"feature": "PM10", "feature_label": "PM10 concentration", "score": 0.3, "pct": 60.0},
                {"feature": "lag1", "feature_label": "PM2.5 one hour ago", "score": 0.2, "pct": 40.0},
            ]

            out = routes.get_model_info()
            self.assertFalse(out.scaling_applied)
            self.assertIn("tree-based XGBoost", out.scaling_note)
            self.assertIn("quantile-bounded", out.outlier_note)
            self.assertTrue(out.has_feature_quantiles)
            self.assertTrue(out.has_global_shap)
            self.assertGreaterEqual(len(out.feature_importance), 1)
            self.assertIn("PM10", out.bounds_preview)
        finally:
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.feature_importance = original_feature_importance

    def test_model_runner_delta_reconstruction(self):
        runner = ModelRunner()
        runner._meta = {
            "model": _DummyModel(pred_value=1.75),
            "target_type": "delta",
            "log_target": False,
            "bias_correction": 0.25,
            "feature_defaults": {},
            "training_pm25_mean": 20.0,
            "features": ["lag1", "NO2"],
        }
        X = pd.DataFrame([{"lag1": 30.0, "NO2": 21.0}], columns=["lag1", "NO2"])

        import backend.app.services.model_runner as model_runner_module

        original_dmatrix = model_runner_module.xgb.DMatrix
        try:
            model_runner_module.xgb.DMatrix = lambda data, feature_names=None: data
            out = runner.predict(X=X, base_lag1=30.0, current_pm25=29.0)
        finally:
            model_runner_module.xgb.DMatrix = original_dmatrix

        self.assertAlmostEqual(out["pm25_t_plus_1"], 32.0, places=6)
        self.assertAlmostEqual(out["delta_pm25_t_plus_1"], 2.0, places=6)

    def test_feature_builder_outputs_valid_row(self):
        aq_hist, w_hist = _make_history(hours=72)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)

        assembler_df, _stats = routes.history_assembler.assemble(aq_hist=aq_hist, w_hist=w_hist, target_hours=72)
        meta = {
            "features": FEATURES_FOR_TEST,
            "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
        }
        res = FeatureBuilder().build(meta=meta, history_df=assembler_df, aq_cur=aq_cur, w_cur=w_cur)
        self.assertEqual(list(res.X.columns), FEATURES_FOR_TEST)
        self.assertEqual(len(res.X), 1)
        self.assertTrue(np.isfinite(res.X.iloc[0].values).all())
        self.assertTrue(np.isfinite(res.base_lag1))
        self.assertIsInstance(res.imputed_feature_names, list)
        self.assertIsInstance(res.extreme_current_events, list)

    def test_feature_builder_flags_extreme_current_exogenous(self):
        aq_hist, w_hist = _make_history(hours=72)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)

        # Deliberately push PM10 beyond q99 to verify robustness flags.
        aq_cur["pm10"] = 250.0

        assembler_df, _stats = routes.history_assembler.assemble(aq_hist=aq_hist, w_hist=w_hist, target_hours=72)
        meta = {
            "features": FEATURES_FOR_TEST,
            "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
            "feature_quantiles": {
                "PM10": {"q01": 10.0, "q99": 80.0},
                "NO2": {"q01": 5.0, "q99": 60.0},
                "wind_speed": {"q01": 1.0, "q99": 20.0},
                "temperature": {"q01": 15.0, "q99": 40.0},
                "humidity": {"q01": 20.0, "q99": 95.0},
                "pressure": {"q01": 980.0, "q99": 1040.0},
                "CO": {"q01": 100.0, "q99": 1200.0},
                "SO2": {"q01": 1.0, "q99": 50.0},
                "O3": {"q01": 5.0, "q99": 120.0},
            },
        }

        res = FeatureBuilder().build(meta=meta, history_df=assembler_df, aq_cur=aq_cur, w_cur=w_cur)
        self.assertGreaterEqual(len(res.extreme_current_events), 1)
        self.assertTrue(any(e["feature"] == "PM10" for e in res.extreme_current_events))

    def test_forecast_interactive_route_smoke(self):
        aq_hist, w_hist = _make_history(hours=96)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)

        original_fetch_history = routes.data_client.fetch_history
        original_fetch_current = routes.data_client.fetch_current
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series

        def fake_get_meta():
            return {
                "model": _DummyModel(pred_value=0.0),
                "features": FEATURES_FOR_TEST,
                "target_type": "delta",
                "bias_correction": 0.0,
                "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
                "feature_quantiles": {
                    "NO2": {"q01": 5.0, "q05": 10.0, "q95": 35.0, "q99": 40.0},
                    "CO": {"q01": 300.0, "q05": 350.0, "q95": 650.0, "q99": 700.0},
                    "PM10": {"q01": 10.0, "q05": 15.0, "q95": 55.0, "q99": 65.0},
                    "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                    "temperature": {"q01": 20.0, "q05": 22.0, "q95": 35.0, "q99": 37.0},
                    "humidity": {"q01": 30.0, "q05": 40.0, "q95": 92.0, "q99": 95.0},
                },
                "schema_version": 1,
            }

        def fake_predict(X, base_lag1, current_pm25):
            row = X.iloc[0]
            base = float(base_lag1 if np.isfinite(base_lag1) else current_pm25)
            # A deterministic, monotonic surrogate for smoke validation.
            delta = (
                0.03 * float(row.get("NO2", 0.0))
                + 0.01 * float(row.get("PM10", 0.0))
                - 0.05 * float(row.get("wind_speed", 0.0))
            )
            pm25_next = base + delta
            return {
                "pred_space": delta,
                "delta_pm25_t_plus_1": delta,
                "pm25_t_plus_1": pm25_next,
            }

        def fake_shap_series(_model, X_row):
            row = X_row.iloc[0]
            vals = {
                "NO2": 0.02 * float(row.get("NO2", 0.0)),
                "wind_speed": -0.03 * float(row.get("wind_speed", 0.0)),
                "PM10": 0.01 * float(row.get("PM10", 0.0)),
            }
            out = pd.Series(0.0, index=X_row.columns, dtype=float)
            for k, v in vals.items():
                if k in out.index:
                    out[k] = v
            return out

        try:
            routes.data_client.fetch_history = lambda lat, lon, hours, timezone="auto": (aq_hist.copy(), w_hist.copy())
            routes.data_client.fetch_current = lambda lat, lon, timezone="auto": (dict(aq_cur), dict(w_cur))
            routes.model_runner.get_meta = fake_get_meta
            routes.model_runner.predict = fake_predict
            routes.explainer.shap_series = fake_shap_series

            req = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "smoke-test-1",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    # Legacy alias should map to canonical trapped_pollution.
                    "scenario": {"type": "macro", "scenario_id": "stagnation", "intensity": 80},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.10, "hard_q": 0.02}},
                }
            )
            resp = routes.forecast_interactive(req)

            self.assertTrue(math.isfinite(resp.baseline.prediction.pm25_t_plus_1))
            self.assertTrue(math.isfinite(resp.scenario.prediction.pm25_t_plus_1))
            self.assertTrue(math.isfinite(resp.delta.pm25_change))
            self.assertGreater(len(resp.scenario.applied_overrides), 0)
            self.assertGreaterEqual(resp.health.history.used_hours, 24)
            self.assertGreaterEqual(resp.health.imputation.total_features, len(FEATURES_FOR_TEST))
            self.assertGreaterEqual(resp.health.quality_score, 0.0)
            self.assertLessEqual(resp.health.quality_score, 1.0)
            self.assertTrue(hasattr(resp.health, "extreme_inputs"))
            self.assertIn("count", resp.health.extreme_inputs.model_dump())
            self.assertEqual(resp.scenario.scenario_id, "trapped_pollution")
            self.assertEqual(resp.baseline.shap.target_space, "delta_pm25_t_plus_1")
            self.assertIn("Contribution scores explain", resp.baseline.shap.target_space_note)
            self.assertTrue(resp.baseline.shap.method in {"xgboost_pred_contribs", "tree_shap"})
            self.assertTrue(hasattr(resp.baseline.shap, "additivity_ok"))
            self.assertTrue(hasattr(resp.baseline.shap, "prediction_alignment_ok"))
            # Requested OOD quantiles are preserved while effective ranges align to supported quantiles.
            self.assertAlmostEqual(resp.health.ood.soft_range["q_low"], 0.05, places=6)
            self.assertAlmostEqual(resp.health.ood.hard_range["q_low"], 0.01, places=6)
            self.assertAlmostEqual(resp.health.ood.requested_soft_range["q_low"], 0.10, places=6)
            self.assertAlmostEqual(resp.health.ood.requested_hard_range["q_low"], 0.02, places=6)
            self.assertTrue(any("mapped to supported q" in n for n in resp.health.ood.notes))
            # Stagnation should generally increase pollution in this surrogate model.
            self.assertGreater(resp.delta.pm25_change, 0.0)

            # Legacy windy_dispersion should map to canonical dust_resuspension.
            req_windy = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "smoke-test-2",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "macro", "scenario_id": "windy_dispersion", "intensity": 70},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp_windy = routes.forecast_interactive(req_windy)
            self.assertEqual(resp_windy.scenario.scenario_id, "dust_resuspension")
            self.assertIn("requested_soft_range", resp_windy.health.ood.model_dump())
            self.assertIn("requested_hard_range", resp_windy.health.ood.model_dump())
        finally:
            routes.data_client.fetch_history = original_fetch_history
            routes.data_client.fetch_current = original_fetch_current
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series

    def test_forecast_interactive_route_degrades_if_shap_fails(self):
        aq_hist, w_hist = _make_history(hours=96)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)

        original_fetch_history = routes.data_client.fetch_history
        original_fetch_current = routes.data_client.fetch_current
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series

        def fake_get_meta():
            return {
                "model": _DummyModel(pred_value=0.0),
                "features": FEATURES_FOR_TEST,
                "target_type": "delta",
                "bias_correction": 0.0,
                "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
                "feature_quantiles": {
                    "NO2": {"q01": 5.0, "q05": 10.0, "q95": 35.0, "q99": 40.0},
                    "CO": {"q01": 300.0, "q05": 350.0, "q95": 650.0, "q99": 700.0},
                    "PM10": {"q01": 10.0, "q05": 15.0, "q95": 55.0, "q99": 65.0},
                    "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                    "temperature": {"q01": 20.0, "q05": 22.0, "q95": 35.0, "q99": 37.0},
                    "humidity": {"q01": 30.0, "q05": 40.0, "q95": 92.0, "q99": 95.0},
                },
                "schema_version": 1,
            }

        def fake_predict(X, base_lag1, current_pm25):
            row = X.iloc[0]
            base = float(base_lag1 if np.isfinite(base_lag1) else current_pm25)
            delta = (
                0.03 * float(row.get("NO2", 0.0))
                + 0.01 * float(row.get("PM10", 0.0))
                - 0.05 * float(row.get("wind_speed", 0.0))
            )
            pm25_next = base + delta
            return {
                "pred_space": delta,
                "delta_pm25_t_plus_1": delta,
                "pm25_t_plus_1": pm25_next,
            }

        try:
            routes.data_client.fetch_history = lambda lat, lon, hours, timezone="auto": (aq_hist.copy(), w_hist.copy())
            routes.data_client.fetch_current = lambda lat, lon, timezone="auto": (dict(aq_cur), dict(w_cur))
            routes.model_runner.get_meta = fake_get_meta
            routes.model_runner.predict = fake_predict
            routes.explainer.shap_series = lambda _m, _x: (_ for _ in ()).throw(RuntimeError("shap exploded"))

            req = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "smoke-test-shap-fallback",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "macro", "scenario_id": "dust_resuspension", "intensity": 70},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp = routes.forecast_interactive(req)
            self.assertTrue(math.isfinite(resp.baseline.prediction.pm25_t_plus_1))
            self.assertTrue(math.isfinite(resp.scenario.prediction.pm25_t_plus_1))
            self.assertEqual(len(resp.baseline.shap.top_drivers), 0)
            self.assertEqual(len(resp.scenario.shap.top_drivers), 0)
            self.assertEqual(len(resp.delta.delta_shap.top_changes), 0)
            self.assertEqual(resp.baseline.shap.target_space, "delta_pm25_t_plus_1")
            self.assertEqual(resp.baseline.shap.method, "unavailable")
            self.assertIn("Explainability is temporarily unavailable", resp.baseline.shap.summary_text)
        finally:
            routes.data_client.fetch_history = original_fetch_history
            routes.data_client.fetch_current = original_fetch_current
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series

    def test_forecast_uses_location_id_when_provided(self):
        aq_hist, w_hist = _make_history(hours=96)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)

        original_fetch_history = routes.data_client.fetch_history
        original_fetch_current = routes.data_client.fetch_current
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series

        calls = {"history": None, "current": None}

        def fake_get_meta():
            return {
                "model": _DummyModel(pred_value=0.0),
                "features": FEATURES_FOR_TEST,
                "target_type": "delta",
                "bias_correction": 0.0,
                "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
                "feature_quantiles": {},
                "schema_version": 1,
            }

        def fake_predict(X, base_lag1, current_pm25):
            row = X.iloc[0]
            base = float(base_lag1 if np.isfinite(base_lag1) else current_pm25)
            delta = 0.03 * float(row.get("NO2", 0.0)) + 0.01 * float(row.get("PM10", 0.0))
            return {
                "pred_space": delta,
                "delta_pm25_t_plus_1": delta,
                "pm25_t_plus_1": base + delta,
            }

        def fake_shap_series(_model, X_row):
            out = pd.Series(0.0, index=X_row.columns, dtype=float)
            out["NO2"] = 0.03
            out["PM10"] = 0.02
            return out

        try:
            def _fake_fetch_history(lat, lon, hours, timezone="auto"):
                calls["history"] = (float(lat), float(lon))
                return aq_hist.copy(), w_hist.copy()

            def _fake_fetch_current(lat, lon, timezone="auto"):
                calls["current"] = (float(lat), float(lon))
                return dict(aq_cur), dict(w_cur)

            routes.data_client.fetch_history = _fake_fetch_history
            routes.data_client.fetch_current = _fake_fetch_current
            routes.model_runner.get_meta = fake_get_meta
            routes.model_runner.predict = fake_predict
            routes.explainer.shap_series = fake_shap_series

            req = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "location-id-test",
                    "forecast_mode": "live",
                    "location": {"lat": 0.0, "lon": 0.0, "name": "IgnoreMe", "location_id": "colombo_lk"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "macro", "scenario_id": "traffic_gridlock", "intensity": 40},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp = routes.forecast_interactive(req)
            self.assertTrue(np.isfinite(resp.baseline.prediction.pm25_t_plus_1))
            self.assertIsNotNone(calls["history"])
            self.assertIsNotNone(calls["current"])
            self.assertAlmostEqual(calls["history"][0], 6.9271, places=4)
            self.assertAlmostEqual(calls["history"][1], 79.8612, places=4)
            self.assertAlmostEqual(calls["current"][0], 6.9271, places=4)
            self.assertAlmostEqual(calls["current"][1], 79.8612, places=4)
        finally:
            routes.data_client.fetch_history = original_fetch_history
            routes.data_client.fetch_current = original_fetch_current
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series

    def test_run_logger_list_runs_handles_null_scenario_id(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "runs.sqlite"
            logger = RunLogger(db_path)
            logger.save_run(
                run_id="run_test_1",
                created_at="2026-02-27T00:00:00+00:00",
                location={"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                scenario_request={
                    "scenario": {
                        "type": "custom",
                        "scenario_id": None,
                        "intensity": 0,
                        "items": [],
                    }
                },
                response_json={
                    "scenario": {"scenario_id": "custom"},
                    "delta": {"pm25_change": 0.0},
                    "health": {"ood": {"flag": False}},
                },
                model_version="test",
            )
            rows = logger.list_runs(limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scenario_id"], "baseline")
            self.assertEqual(rows[0]["scenario_mode"], "baseline")
            self.assertEqual(rows[0]["intensity"], 0)
            self.assertEqual(rows[0]["pm25_change"], 0.0)
            self.assertEqual(rows[0]["ood_flag"], False)

    def test_run_logger_list_runs_prefers_canonical_scenario_id(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "runs.sqlite"
            logger = RunLogger(db_path)
            logger.save_run(
                run_id="run_test_alias",
                created_at="2026-03-16T00:00:00+00:00",
                location={"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                scenario_request={
                    "scenario": {
                        "type": "macro",
                        "scenario_id": "windy_dispersion",
                        "intensity": 70,
                    }
                },
                response_json={
                    "scenario": {"scenario_id": "dust_resuspension"},
                    "delta": {"pm25_change": 0.8},
                    "health": {"ood": {"flag": False}},
                },
                model_version="test",
            )
            rows = logger.list_runs(limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scenario_id"], "dust_resuspension")
            self.assertEqual(rows[0]["scenario_mode"], "macro")
            self.assertEqual(rows[0]["intensity"], 70)

            # Historical row: canonicalize from request alias when response scenario is absent.
            logger.save_run(
                run_id="run_test_alias_req_only",
                created_at="2026-03-16T00:05:00+00:00",
                location={"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                scenario_request={
                    "scenario": {
                        "type": "macro",
                        "scenario_id": "heavy_rainstorm",
                        "intensity": 65,
                    }
                },
                response_json={
                    "delta": {"pm25_change": -0.6},
                    "health": {"ood": {"flag": False}},
                },
                model_version="test",
            )
            rows2 = logger.list_runs(limit=10)
            by_id = {r["run_id"]: r for r in rows2}
            self.assertEqual(by_id["run_test_alias_req_only"]["scenario_id"], "strong_dispersion")

    def test_run_logger_get_run_canonicalizes_legacy_response_scenario_id(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "runs.sqlite"
            logger = RunLogger(db_path)
            logger.save_run(
                run_id="run_test_get_alias",
                created_at="2026-03-16T00:00:00+00:00",
                location={"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                scenario_request={
                    "scenario": {"type": "macro", "scenario_id": "windy_dispersion", "intensity": 70}
                },
                response_json={
                    "scenario": {"scenario_id": "windy_dispersion"},
                    "delta": {"pm25_change": 0.9},
                    "health": {"ood": {"flag": False}},
                },
                model_version="test",
            )
            row = logger.get_run("run_test_get_alias")
            self.assertIsNotNone(row)
            self.assertEqual(row["response_json"]["scenario"]["scenario_id"], "dust_resuspension")

    def test_macro_scenarios_apply_non_trivial_changes(self):
        engine = ScenarioEngine()

        baseline_X = pd.DataFrame(
            [
                {
                    "wind_speed": 6.0,
                    "temperature": 30.0,
                    "humidity": 70.0,
                    "pressure": 1010.0,
                    "PM10": 35.0,
                    "NO2": 22.0,
                    "SO2": 8.0,
                    "O3": 58.0,
                    "CO": 520.0,
                }
            ]
        )
        meta = {
            "feature_quantiles": {
                "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                "temperature": {"q01": 20.0, "q05": 22.0, "q95": 35.0, "q99": 37.0},
                "humidity": {"q01": 30.0, "q05": 40.0, "q95": 92.0, "q99": 96.0},
                "pressure": {"q01": 1004.0, "q05": 1006.0, "q95": 1014.0, "q99": 1016.0},
                "PM10": {"q01": 10.0, "q05": 15.0, "q95": 60.0, "q99": 70.0},
                "NO2": {"q01": 5.0, "q05": 10.0, "q95": 40.0, "q99": 45.0},
                "SO2": {"q01": 2.0, "q05": 4.0, "q95": 16.0, "q99": 18.0},
                "O3": {"q01": 15.0, "q05": 25.0, "q95": 95.0, "q99": 110.0},
                "CO": {"q01": 250.0, "q05": 320.0, "q95": 720.0, "q99": 780.0},
            }
        }

        def score(row):
            # Deterministic surrogate score used only for sign sanity in this smoke test.
            return (
                0.020 * float(row["NO2"])
                + 0.015 * float(row["PM10"])
                + 0.0008 * float(row["CO"])
                + 0.020 * float(row["SO2"])
                + 0.010 * float(row["humidity"])
                + 0.025 * float(row["pressure"])
                - 0.050 * float(row["wind_speed"])
                + 0.005 * float(row["temperature"])
            )

        expected_sign = {
            "traffic_gridlock": +1,
            "strong_dispersion": -1,
            "heatwave": +1,
            "dust_resuspension": +1,
            "trapped_pollution": +1,
            "industrial_plume": +1,
        }

        for sid, sign in expected_sign.items():
            scenario = ScenarioRequest.model_validate({"type": "macro", "scenario_id": sid, "intensity": 100})
            scenario_X, applied, _ood, out_id = engine.apply(scenario=scenario, baseline_X=baseline_X, meta=meta)

            self.assertEqual(out_id, sid)
            self.assertGreater(len(applied), 0)
            move_sum = sum(abs(float(a["to"]) - float(a["from"])) for a in applied)
            self.assertGreater(move_sum, 0.5, msg=f"{sid} move_sum too small: {move_sum}")

            delta_score = score(scenario_X.iloc[0]) - score(baseline_X.iloc[0])
            self.assertGreater(sign * delta_score, 0.0, msg=f"{sid} direction mismatch: {delta_score}")

    def test_macro_aliases_map_to_canonical(self):
        engine = ScenarioEngine()
        baseline_X = pd.DataFrame(
            [
                {
                    "wind_speed": 6.0,
                    "temperature": 30.0,
                    "humidity": 70.0,
                    "pressure": 1010.0,
                    "PM10": 35.0,
                    "NO2": 22.0,
                    "SO2": 8.0,
                    "O3": 58.0,
                    "CO": 520.0,
                }
            ]
        )
        meta = {
            "feature_quantiles": {
                "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                "temperature": {"q01": 20.0, "q05": 22.0, "q95": 35.0, "q99": 37.0},
                "humidity": {"q01": 30.0, "q05": 40.0, "q95": 92.0, "q99": 96.0},
                "pressure": {"q01": 1004.0, "q05": 1006.0, "q95": 1014.0, "q99": 1016.0},
                "PM10": {"q01": 10.0, "q05": 15.0, "q95": 60.0, "q99": 70.0},
                "NO2": {"q01": 5.0, "q05": 10.0, "q95": 40.0, "q99": 45.0},
                "SO2": {"q01": 2.0, "q05": 4.0, "q95": 16.0, "q99": 18.0},
                "O3": {"q01": 15.0, "q05": 25.0, "q95": 95.0, "q99": 110.0},
                "CO": {"q01": 250.0, "q05": 320.0, "q95": 720.0, "q99": 780.0},
            }
        }
        intensity = 80

        strong_req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "strong_dispersion", "intensity": intensity})
        legacy_rain_req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "heavy_rainstorm", "intensity": intensity})

        X_strong, applied_strong, _ood_strong, sid_strong = engine.apply(strong_req, baseline_X=baseline_X, meta=meta)
        X_legacy, applied_legacy, _ood_legacy, sid_legacy = engine.apply(legacy_rain_req, baseline_X=baseline_X, meta=meta)
        self.assertEqual(sid_strong, "strong_dispersion")
        self.assertEqual(sid_legacy, "strong_dispersion")
        self.assertEqual(len(applied_strong), len(applied_legacy))
        self.assertTrue(np.allclose(X_strong.values, X_legacy.values))

        trapped_req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "trapped_pollution", "intensity": intensity})
        legacy_stag_req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "stagnation", "intensity": intensity})
        X_trapped, _applied_trapped, _ood_trapped, sid_trapped = engine.apply(trapped_req, baseline_X=baseline_X, meta=meta)
        X_stag, _applied_stag, _ood_stag, sid_stag = engine.apply(legacy_stag_req, baseline_X=baseline_X, meta=meta)
        self.assertEqual(sid_trapped, "trapped_pollution")
        self.assertEqual(sid_stag, "trapped_pollution")
        self.assertTrue(np.allclose(X_trapped.values, X_stag.values))

        dust_req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "dust_resuspension", "intensity": intensity})
        legacy_windy_req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "windy_dispersion", "intensity": intensity})
        X_dust, _applied_dust, _ood_dust, sid_dust = engine.apply(dust_req, baseline_X=baseline_X, meta=meta)
        X_windy, _applied_windy, _ood_windy, sid_windy = engine.apply(legacy_windy_req, baseline_X=baseline_X, meta=meta)
        self.assertEqual(sid_dust, "dust_resuspension")
        self.assertEqual(sid_windy, "dust_resuspension")
        self.assertTrue(np.allclose(X_dust.values, X_windy.values))

    def test_dust_resuspension_bounded_and_non_trivial(self):
        engine = ScenarioEngine()
        baseline_X = pd.DataFrame(
            [
                {
                    "wind_speed": 6.0,
                    "temperature": 30.0,
                    "humidity": 70.0,
                    "pressure": 1010.0,
                    "PM10": 35.0,
                    "NO2": 22.0,
                    "SO2": 8.0,
                    "O3": 58.0,
                    "CO": 520.0,
                }
            ]
        )
        meta = {
            "feature_quantiles": {
                "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                "humidity": {"q01": 30.0, "q05": 40.0, "q95": 92.0, "q99": 96.0},
                "PM10": {"q01": 10.0, "q05": 15.0, "q95": 60.0, "q99": 70.0},
            }
        }
        req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "dust_resuspension", "intensity": 100})
        scenario_X, applied, _ood, sid = engine.apply(req, baseline_X=baseline_X, meta=meta)
        self.assertEqual(sid, "dust_resuspension")
        self.assertGreater(len(applied), 0)

        by_feature = {row["feature"]: row for row in applied}
        self.assertIn("PM10", by_feature)
        self.assertIn("wind_speed", by_feature)
        self.assertIn("humidity", by_feature)

        self.assertGreater(float(by_feature["PM10"]["to"]), float(by_feature["PM10"]["from"]))
        self.assertGreater(float(by_feature["wind_speed"]["to"]), float(by_feature["wind_speed"]["from"]))
        self.assertLess(float(by_feature["humidity"]["to"]), float(by_feature["humidity"]["from"]))

        # Bounded by q05/q95-style scenario safety rails.
        self.assertLessEqual(float(by_feature["PM10"]["to"]), 60.0)
        self.assertLessEqual(float(by_feature["wind_speed"]["to"]), 12.0)
        self.assertGreaterEqual(float(by_feature["humidity"]["to"]), 40.0)

        self.assertTrue(np.isfinite(scenario_X.iloc[0].values).all())

    def test_custom_stronger_realistic_mode_boosts_within_bounds(self):
        engine = ScenarioEngine()
        baseline_X = pd.DataFrame(
            [
                {
                    "PM10": 30.0,
                    "NO2": 20.0,
                    "CO": 500.0,
                    "humidity": 70.0,
                    "wind_speed": 5.0,
                    "pressure": 1010.0,
                    "temperature": 30.0,
                }
            ]
        )
        meta = {
            "feature_quantiles": {
                "PM10": {"q01": 8.0, "q05": 12.0, "q25": 18.0, "q50": 28.0, "q75": 40.0, "q95": 58.0, "q99": 66.0},
                "NO2": {"q01": 4.0, "q05": 8.0, "q25": 12.0, "q50": 18.0, "q75": 26.0, "q95": 38.0, "q99": 44.0},
            },
            "global_shap_mean_abs": {"PM10": 0.38, "NO2": 0.29, "wind_speed": 0.18},
        }
        overrides = {"PM10": 42.0, "NO2": 27.0}

        con_X, con_applied, _con_ood, _con_ctx, con_preview, con_mode = engine.apply_value_overrides(
            overrides=overrides,
            baseline_X=baseline_X,
            meta=meta,
            impact_mode="conservative",
            return_context=True,
            include_preview=True,
        )
        str_X, str_applied, _str_ood, _str_ctx, str_preview, str_mode = engine.apply_value_overrides(
            overrides=overrides,
            baseline_X=baseline_X,
            meta=meta,
            impact_mode="stronger_realistic",
            return_context=True,
            include_preview=True,
        )

        self.assertEqual(con_mode, "conservative")
        self.assertEqual(str_mode, "stronger_realistic")
        self.assertEqual(len(con_applied), 2)
        self.assertEqual(len(str_applied), 2)

        con_move = abs(float(con_X.iloc[0]["PM10"]) - 30.0) + abs(float(con_X.iloc[0]["NO2"]) - 20.0)
        str_move = abs(float(str_X.iloc[0]["PM10"]) - 30.0) + abs(float(str_X.iloc[0]["NO2"]) - 20.0)
        self.assertGreaterEqual(str_move, con_move)

        # Stronger mode remains quantile-bounded (outer hard rails q01/q99).
        self.assertGreaterEqual(float(str_X.iloc[0]["PM10"]), 8.0)
        self.assertLessEqual(float(str_X.iloc[0]["PM10"]), 66.0)
        self.assertGreaterEqual(float(str_X.iloc[0]["NO2"]), 4.0)
        self.assertLessEqual(float(str_X.iloc[0]["NO2"]), 44.0)

        self.assertIn(str_preview["level"], {"low", "medium", "high"})
        self.assertGreaterEqual(float(str_preview["score"]), float(con_preview["score"]))

    def test_ood_requested_and_effective_ranges_are_aligned(self):
        engine = ScenarioEngine()
        baseline_X = pd.DataFrame(
            [{"wind_speed": 6.0, "PM10": 35.0, "NO2": 22.0, "CO": 520.0, "humidity": 70.0, "pressure": 1010.0}]
        )
        meta = {
            "feature_quantiles": {
                "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                "PM10": {"q01": 10.0, "q05": 15.0, "q95": 60.0, "q99": 70.0},
                "NO2": {"q01": 5.0, "q05": 10.0, "q95": 40.0, "q99": 45.0},
                "CO": {"q01": 250.0, "q05": 320.0, "q95": 720.0, "q99": 780.0},
                "humidity": {"q01": 30.0, "q05": 40.0, "q95": 92.0, "q99": 96.0},
                "pressure": {"q01": 1004.0, "q05": 1006.0, "q95": 1014.0, "q99": 1016.0},
            }
        }
        req = ScenarioRequest.model_validate({"type": "macro", "scenario_id": "strong_dispersion", "intensity": 70})
        _X, _applied, _events, _sid, ood_ctx = engine.apply(
            req,
            baseline_X=baseline_X,
            meta=meta,
            ood_opts={"soft_q": 0.10, "hard_q": 0.02},
            return_context=True,
        )
        # Only q01/q05/q95/q99 exist, so effective ranges map to nearest supported levels.
        self.assertAlmostEqual(ood_ctx["requested_soft_q"], 0.10, places=6)
        self.assertAlmostEqual(ood_ctx["requested_hard_q"], 0.02, places=6)
        self.assertAlmostEqual(ood_ctx["effective_soft_q"], 0.05, places=6)
        self.assertAlmostEqual(ood_ctx["effective_hard_q"], 0.01, places=6)
        self.assertTrue(any("mapped to supported q" in n for n in ood_ctx["notes"]))

    def test_custom_mode_uses_baseline_context_plus_overrides(self):
        aq_hist, w_hist = _make_history(hours=96)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)
        hist_df, hist_stats = routes.history_assembler.assemble(aq_hist=aq_hist, w_hist=w_hist, target_hours=72)

        original_get_context = routes.baseline_context_service.get_custom_context
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series

        def fake_get_meta():
            return {
                "model": _DummyModel(pred_value=0.0),
                "features": FEATURES_FOR_TEST,
                "target_type": "delta",
                "bias_correction": 0.0,
                "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
                "feature_quantiles": {},
                "schema_version": 1,
            }

        def fake_predict(X, base_lag1, current_pm25):
            row = X.iloc[0]
            base = float(base_lag1 if np.isfinite(base_lag1) else current_pm25)
            delta = (
                0.04 * float(row.get("NO2", 0.0))
                + 0.02 * float(row.get("PM10", 0.0))
                - 0.03 * float(row.get("wind_speed", 0.0))
            )
            return {
                "pred_space": delta,
                "delta_pm25_t_plus_1": delta,
                "pm25_t_plus_1": base + delta,
            }

        def fake_shap_series(_model, X_row):
            out = pd.Series(0.0, index=X_row.columns, dtype=float)
            out["PM10"] = 0.1
            out["NO2"] = 0.08
            out["wind_speed"] = -0.05
            return out

        try:
            routes.baseline_context_service.get_custom_context = lambda **kwargs: type("Ctx", (), {
                "history_df": hist_df.copy(),
                "history_stats": dict(hist_stats),
                "aq_cur": dict(aq_cur),
                "w_cur": dict(w_cur),
                "source": "reference_profile",
                "live_data_used": False,
                "notes": ["Using dataset-derived reference baseline profile."],
            })()
            routes.model_runner.get_meta = fake_get_meta
            routes.model_runner.predict = fake_predict
            routes.explainer.shap_series = fake_shap_series

            req = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "custom-mode-test",
                    "forecast_mode": "custom",
                    "custom_impact_mode": "stronger_realistic",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "custom", "intensity": 0, "items": []},
                    "custom_overrides": {
                        "PM10": 120.0,
                        "NO2": 35.0,
                        "wind_speed": 0.2,
                    },
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp = routes.forecast_interactive(req)
            self.assertEqual(resp.meta.forecast_mode, "custom")
            self.assertEqual(resp.meta.baseline_source, "reference_profile")
            self.assertFalse(resp.meta.live_data_used)
            self.assertTrue(resp.meta.overrides_applied)
            self.assertEqual(resp.scenario.scenario_id, "custom_what_if")
            self.assertEqual(resp.scenario.intensity, 0)
            self.assertEqual(resp.meta.custom_impact_mode, "stronger_realistic")
            self.assertEqual(resp.scenario.custom_impact_mode, "stronger_realistic")
            self.assertIsNotNone(resp.scenario.impact_preview)
            self.assertGreater(len(resp.scenario.applied_overrides), 0)
            self.assertTrue(np.isfinite(resp.delta.pm25_change))
        finally:
            routes.baseline_context_service.get_custom_context = original_get_context
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series

    def test_intensity_semantics_custom_vs_macro(self):
        aq_hist, w_hist = _make_history(hours=96)
        ts = aq_hist.index[-1] + timedelta(hours=1)
        aq_cur, w_cur = _make_current(ts)
        hist_df, hist_stats = routes.history_assembler.assemble(aq_hist=aq_hist, w_hist=w_hist, target_hours=72)

        original_get_context = routes.baseline_context_service.get_custom_context
        original_fetch_history = routes.data_client.fetch_history
        original_fetch_current = routes.data_client.fetch_current
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series
        original_explain_meta = routes.explainer.explanation_meta

        def fake_get_meta():
            return {
                "model": _DummyModel(pred_value=0.0),
                "features": FEATURES_FOR_TEST,
                "target_type": "delta",
                "bias_correction": 0.0,
                "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
                "feature_quantiles": {
                    "NO2": {"q01": 5.0, "q05": 10.0, "q95": 35.0, "q99": 40.0},
                    "PM10": {"q01": 10.0, "q05": 15.0, "q95": 55.0, "q99": 65.0},
                    "wind_speed": {"q01": 1.0, "q05": 2.0, "q95": 12.0, "q99": 16.0},
                },
                "schema_version": 2,
            }

        def fake_predict(X, base_lag1, current_pm25):
            row = X.iloc[0]
            base = float(base_lag1 if np.isfinite(base_lag1) else current_pm25)
            delta = (
                0.03 * float(row.get("NO2", 0.0))
                + 0.01 * float(row.get("PM10", 0.0))
                - 0.05 * float(row.get("wind_speed", 0.0))
            )
            return {
                "pred_space": float(delta),
                "delta_pm25_t_plus_1": float(delta),
                "pm25_t_plus_1": float(base + delta),
            }

        def fake_shap_series(_model, X_row):
            out = pd.Series(0.0, index=X_row.columns, dtype=float)
            out["NO2"] = 0.12
            out["PM10"] = 0.08
            out["wind_speed"] = -0.06
            return out

        def fake_explain_meta(_model, _X, contrib_s, prediction_signal=None):
            total = float(contrib_s.sum())
            return {
                "method": "xgboost_pred_contribs",
                "base_value": 0.0,
                "contrib_sum": total,
                "reconstructed_signal": total,
                "prediction_signal": float(prediction_signal) if prediction_signal is not None else None,
                "additivity_error": 0.0,
                "additivity_ok": True,
                "additivity_tolerance": 1e-5,
                "prediction_alignment_error": 0.0,
                "prediction_alignment_ok": True,
            }

        try:
            routes.baseline_context_service.get_custom_context = lambda **kwargs: type("Ctx", (), {
                "history_df": hist_df.copy(),
                "history_stats": dict(hist_stats),
                "aq_cur": dict(aq_cur),
                "w_cur": dict(w_cur),
                "source": "reference_profile",
                "live_data_used": False,
                "notes": ["Using dataset-derived reference baseline profile."],
            })()
            routes.data_client.fetch_history = lambda lat, lon, hours, timezone="auto": (aq_hist.copy(), w_hist.copy())
            routes.data_client.fetch_current = lambda lat, lon, timezone="auto": (dict(aq_cur), dict(w_cur))
            routes.model_runner.get_meta = fake_get_meta
            routes.model_runner.predict = fake_predict
            routes.explainer.shap_series = fake_shap_series
            routes.explainer.explanation_meta = fake_explain_meta

            req_custom = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "intensity-custom",
                    "forecast_mode": "custom",
                    "custom_impact_mode": "conservative",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "custom", "intensity": 0, "items": []},
                    "custom_overrides": {"PM10": 45.0, "NO2": 25.0},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp_custom = routes.forecast_interactive(req_custom)
            self.assertEqual(resp_custom.scenario.scenario_id, "custom_what_if")
            self.assertEqual(resp_custom.scenario.intensity, 0)

            req_macro = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "intensity-macro",
                    "forecast_mode": "live",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "macro", "scenario_id": "traffic_gridlock", "intensity": 73},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp_macro = routes.forecast_interactive(req_macro)
            self.assertEqual(resp_macro.scenario.scenario_id, "traffic_gridlock")
            self.assertEqual(resp_macro.scenario.intensity, 73)

            req_guided = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "intensity-guided",
                    "forecast_mode": "live",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {
                        "type": "custom",
                        "intensity": 70,
                        "items": [{"category": "wind", "direction": "decrease", "magnitude": "large"}],
                    },
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            resp_guided = routes.forecast_interactive(req_guided)
            self.assertEqual(resp_guided.meta.forecast_mode, "live")
            self.assertEqual(resp_guided.scenario.scenario_id, "guided_intervention")
            self.assertEqual(resp_guided.scenario.intensity, 70)
        finally:
            routes.baseline_context_service.get_custom_context = original_get_context
            routes.data_client.fetch_history = original_fetch_history
            routes.data_client.fetch_current = original_fetch_current
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series
            routes.explainer.explanation_meta = original_explain_meta

    def test_live_mode_failure_guides_to_custom_mode(self):
        original_fetch_history = routes.data_client.fetch_history
        original_get_meta = routes.model_runner.get_meta
        try:
            routes.data_client.fetch_history = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("upstream unavailable"))
            routes.model_runner.get_meta = lambda: {
                "model": _DummyModel(pred_value=0.0),
                "features": FEATURES_FOR_TEST,
                "target_type": "delta",
                "bias_correction": 0.0,
                "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
                "schema_version": 1,
            }
            req = InteractiveForecastRequest.model_validate(
                {
                    "request_id": "live-fail-guidance",
                    "forecast_mode": "live",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "custom", "intensity": 0, "items": []},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
            with self.assertRaises(HTTPException) as ctx:
                routes.forecast_interactive(req)
            self.assertEqual(ctx.exception.status_code, 502)
            self.assertIn("Custom What-If Forecast", str(ctx.exception.detail))
        finally:
            routes.data_client.fetch_history = original_fetch_history
            routes.model_runner.get_meta = original_get_meta

    def test_custom_override_validation_rejects_out_of_range_values(self):
        with self.assertRaises(Exception):
            InteractiveForecastRequest.model_validate(
                {
                    "request_id": "invalid-custom-input",
                    "forecast_mode": "custom",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "custom", "intensity": 0, "items": []},
                    "custom_overrides": {"humidity": 180.0},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )

    def test_guided_intervention_rejects_duplicate_categories(self):
        with self.assertRaises(ValidationError) as ctx:
            InteractiveForecastRequest.model_validate(
                {
                    "request_id": "guided-duplicate-categories",
                    "forecast_mode": "live",
                    "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                    "time": {"mode": "now"},
                    "scenario": {
                        "type": "custom",
                        "intensity": 70,
                        "items": [
                            {"category": "wind", "direction": "increase", "magnitude": "small"},
                            {"category": "wind", "direction": "decrease", "magnitude": "large"},
                        ],
                    },
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
        self.assertIn("must not contain duplicate categories", str(ctx.exception))

    def test_location_validation_rejects_out_of_range_coordinates(self):
        with self.assertRaises(ValidationError):
            InteractiveForecastRequest.model_validate(
                {
                    "request_id": "invalid-location-input",
                    "forecast_mode": "live",
                    "location": {"lat": 150.0, "lon": 79.8612, "name": "BadLat"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "custom", "intensity": 0, "items": []},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )
        with self.assertRaises(ValidationError):
            InteractiveForecastRequest.model_validate(
                {
                    "request_id": "invalid-location-input-2",
                    "forecast_mode": "live",
                    "location": {"lat": 6.9271, "lon": 280.0, "name": "BadLon"},
                    "time": {"mode": "now"},
                    "scenario": {"type": "custom", "intensity": 0, "items": []},
                    "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
                }
            )


if __name__ == "__main__":
    unittest.main()
