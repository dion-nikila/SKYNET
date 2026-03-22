from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.api import routes
from backend.app.main import app
from backend.app.services.history_assembler import HistoryAssembler


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


class _DummyModel:
    pass


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
    }
    w_cur = {
        "time": ts.isoformat(),
        "temperature_2m": 31.4,
        "relative_humidity_2m": 80.0,
        "surface_pressure": 1010.5,
        "wind_speed_10m": 4.2,
    }
    return aq_cur, w_cur


def _fake_meta():
    return {
        "model": _DummyModel(),
        "features": FEATURES_FOR_TEST,
        "target_type": "delta",
        "bias_correction": 0.0,
        "feature_defaults": {f: 0.0 for f in FEATURES_FOR_TEST},
        "feature_quantiles": {
            "NO2": {"q01": 5.0, "q05": 10.0, "q25": 14.0, "q50": 20.0, "q75": 28.0, "q95": 35.0, "q99": 40.0},
            "PM10": {"q01": 10.0, "q05": 15.0, "q25": 20.0, "q50": 30.0, "q75": 42.0, "q95": 55.0, "q99": 65.0},
            "wind_speed": {"q01": 1.0, "q05": 2.0, "q25": 3.0, "q50": 5.0, "q75": 8.0, "q95": 12.0, "q99": 16.0},
            "temperature": {"q01": 20.0, "q05": 22.0, "q25": 25.0, "q50": 28.0, "q75": 31.0, "q95": 35.0, "q99": 37.0},
            "humidity": {"q01": 30.0, "q05": 40.0, "q25": 52.0, "q50": 66.0, "q75": 78.0, "q95": 92.0, "q99": 95.0},
            "pressure": {"q01": 1002.0, "q05": 1005.0, "q25": 1007.0, "q50": 1010.0, "q75": 1012.0, "q95": 1015.0, "q99": 1018.0},
            "CO": {"q01": 250.0, "q05": 320.0, "q25": 400.0, "q50": 500.0, "q75": 620.0, "q95": 720.0, "q99": 780.0},
            "SO2": {"q01": 2.0, "q05": 4.0, "q25": 6.0, "q50": 8.0, "q75": 10.0, "q95": 14.0, "q99": 18.0},
            "O3": {"q01": 15.0, "q05": 25.0, "q25": 35.0, "q50": 48.0, "q75": 60.0, "q95": 95.0, "q99": 110.0},
        },
        "schema_version": 2,
    }


def _fake_predict(X, base_lag1, current_pm25):
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


def _fake_shap_series(_model, X_row):
    out = pd.Series(0.0, index=X_row.columns, dtype=float)
    out["NO2"] = 0.12
    out["PM10"] = 0.08
    out["wind_speed"] = -0.06
    return out


def _fake_explain_meta(_model, _X, contrib_s, prediction_signal=None):
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


class EndToEndSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.aq_hist, self.w_hist = _make_history(hours=96)
        self.ts = self.aq_hist.index[-1] + timedelta(hours=1)
        self.aq_cur, self.w_cur = _make_current(self.ts)
        self.hist_df, self.hist_stats = HistoryAssembler().assemble(
            aq_hist=self.aq_hist,
            w_hist=self.w_hist,
            target_hours=72,
        )

    def test_custom_mode_forecast_e2e_smoke(self):
        original_get_context = routes.baseline_context_service.get_custom_context
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series
        original_explanation_meta = routes.explainer.explanation_meta

        try:
            routes.baseline_context_service.get_custom_context = lambda **kwargs: type("Ctx", (), {
                "history_df": self.hist_df.copy(),
                "history_stats": dict(self.hist_stats),
                "aq_cur": dict(self.aq_cur),
                "w_cur": dict(self.w_cur),
                "source": "reference_profile",
                "live_data_used": False,
                "notes": ["Using dataset-derived reference baseline profile."],
            })()
            routes.model_runner.get_meta = _fake_meta
            routes.model_runner.predict = _fake_predict
            routes.explainer.shap_series = _fake_shap_series
            routes.explainer.explanation_meta = _fake_explain_meta

            payload = {
                "request_id": "e2e-custom-smoke",
                "forecast_mode": "custom",
                "custom_impact_mode": "stronger_realistic",
                "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                "time": {"mode": "now"},
                "scenario": {"type": "custom", "intensity": 0, "items": []},
                "custom_overrides": {"PM10": 120.0, "NO2": 35.0, "wind_speed": 2.0},
                "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
            }
            resp = self.client.post("/api/v1/forecast/interactive", json=payload)
            self.assertEqual(resp.status_code, 200)
            body = resp.json()

            self.assertIn("baseline", body)
            self.assertIn("scenario", body)
            self.assertIn("health", body)
            self.assertIn("delta", body)
            self.assertIn("meta", body)

            self.assertEqual(body["meta"]["forecast_mode"], "custom")
            self.assertEqual(body["scenario"]["scenario_id"], "custom_what_if")
            self.assertEqual(body["scenario"]["scenario_mode"], "manual_custom")
            self.assertTrue(isinstance(body["scenario"]["applied_overrides"], list))
            self.assertGreater(len(body["scenario"]["applied_overrides"]), 0)
            self.assertTrue(np.isfinite(float(body["baseline"]["prediction"]["pm25_t_plus_1"])))
            self.assertTrue(np.isfinite(float(body["scenario"]["prediction"]["pm25_t_plus_1"])))
            self.assertIn("method", body["baseline"]["shap"])
            self.assertIn("target_space", body["baseline"]["shap"])
            self.assertIn("quality_score", body["health"])
            self.assertIn("extreme_inputs", body["health"])
            self.assertIn("reliability", body["health"])
            self.assertIn("components", body["health"]["reliability"])
            self.assertIn("uncertainty", body["health"])
            self.assertIn("available", body["health"]["uncertainty"])
        finally:
            routes.baseline_context_service.get_custom_context = original_get_context
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series
            routes.explainer.explanation_meta = original_explanation_meta

    def test_live_macro_scenario_e2e_smoke(self):
        original_fetch_history = routes.data_client.fetch_history
        original_fetch_current = routes.data_client.fetch_current
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series
        original_explanation_meta = routes.explainer.explanation_meta

        try:
            routes.data_client.fetch_history = lambda lat, lon, hours, timezone="auto": (self.aq_hist.copy(), self.w_hist.copy())
            routes.data_client.fetch_current = lambda lat, lon, timezone="auto": (dict(self.aq_cur), dict(self.w_cur))
            routes.model_runner.get_meta = _fake_meta
            routes.model_runner.predict = _fake_predict
            routes.explainer.shap_series = _fake_shap_series
            routes.explainer.explanation_meta = _fake_explain_meta

            payload = {
                "request_id": "e2e-live-scenario-smoke",
                "forecast_mode": "live",
                "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                "time": {"mode": "now"},
                "scenario": {"type": "macro", "scenario_id": "traffic_gridlock", "intensity": 72},
                "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
            }
            resp = self.client.post("/api/v1/forecast/interactive", json=payload)
            self.assertEqual(resp.status_code, 200)
            body = resp.json()

            self.assertEqual(body["meta"]["forecast_mode"], "live")
            self.assertEqual(body["scenario"]["scenario_id"], "traffic_gridlock")
            self.assertEqual(body["scenario"]["scenario_mode"], "macro")
            self.assertGreater(len(body["scenario"]["applied_overrides"]), 0)
            self.assertIn("exploratory", str(body["meta"].get("mode_note", "")).lower())
            self.assertIn("ood", body["health"])
            self.assertIn("extreme_inputs", body["health"])
            self.assertIn("reliability", body["health"])
            self.assertIn("uncertainty", body["health"])
            self.assertIn("top_drivers", body["baseline"]["shap"])
        finally:
            routes.data_client.fetch_history = original_fetch_history
            routes.data_client.fetch_current = original_fetch_current
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series
            routes.explainer.explanation_meta = original_explanation_meta

    def test_live_guided_intervention_semantics_e2e_smoke(self):
        original_fetch_history = routes.data_client.fetch_history
        original_fetch_current = routes.data_client.fetch_current
        original_get_meta = routes.model_runner.get_meta
        original_predict = routes.model_runner.predict
        original_shap_series = routes.explainer.shap_series
        original_explanation_meta = routes.explainer.explanation_meta

        try:
            routes.data_client.fetch_history = lambda lat, lon, hours, timezone="auto": (self.aq_hist.copy(), self.w_hist.copy())
            routes.data_client.fetch_current = lambda lat, lon, timezone="auto": (dict(self.aq_cur), dict(self.w_cur))
            routes.model_runner.get_meta = _fake_meta
            routes.model_runner.predict = _fake_predict
            routes.explainer.shap_series = _fake_shap_series
            routes.explainer.explanation_meta = _fake_explain_meta

            payload = {
                "request_id": "e2e-live-guided-smoke",
                "forecast_mode": "live",
                "location": {"lat": 6.9271, "lon": 79.8612, "name": "Colombo"},
                "time": {"mode": "now"},
                "scenario": {
                    "type": "custom",
                    "intensity": 70,
                    "items": [
                        {"category": "wind", "direction": "increase", "magnitude": "medium"},
                        {"category": "emission_proxy", "direction": "decrease", "magnitude": "small"},
                    ],
                },
                "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
            }
            resp = self.client.post("/api/v1/forecast/interactive", json=payload)
            self.assertEqual(resp.status_code, 200)
            body = resp.json()

            self.assertEqual(body["meta"]["forecast_mode"], "live")
            self.assertEqual(body["scenario"]["scenario_id"], "guided_intervention")
            self.assertEqual(body["scenario"]["scenario_mode"], "guided_intervention")
            self.assertEqual(int(body["scenario"]["intensity"]), 70)
            self.assertGreater(len(body["scenario"]["applied_overrides"]), 0)
        finally:
            routes.data_client.fetch_history = original_fetch_history
            routes.data_client.fetch_current = original_fetch_current
            routes.model_runner.get_meta = original_get_meta
            routes.model_runner.predict = original_predict
            routes.explainer.shap_series = original_shap_series
            routes.explainer.explanation_meta = original_explanation_meta


if __name__ == "__main__":
    unittest.main()
