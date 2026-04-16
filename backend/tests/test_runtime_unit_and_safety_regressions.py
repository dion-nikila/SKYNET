from __future__ import annotations

import os
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
from fastapi.testclient import TestClient

from backend.app.api import routes
from backend.app import main as app_main
from backend.app.services.data_client import DataClient
from backend.app.services.baseline_context import BaselineContextService
from backend.app.services.feature_builder import FeatureBuilder
from backend.app.services.history_assembler import HistoryAssembler
from backend.app.services.scenario_engine import ScenarioEngine


class _MockResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = int(status_code)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class RuntimeUnitAndSafetyRegressionTests(unittest.TestCase):
    def test_custom_probe_uses_per_call_overrides_without_mutating_client_state(self):
        class _SpyClient(DataClient):
            def __init__(self):
                super().__init__(timeout_seconds=20, cache_ttl_seconds=0, max_retries=3)
                self.calls = []

            def fetch_history(
                self,
                lat: float,
                lon: float,
                hours: int = 72,
                timezone: str = "auto",
                timeout_seconds: float | None = None,
                max_retries: int | None = None,
            ):
                self.calls.append(
                    ("history", float(self.timeout), int(self.max_retries), timeout_seconds, max_retries)
                )
                idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=max(4, int(hours)), freq="h")
                aq_hist = pd.DataFrame(
                    {
                        "pm2_5": np.linspace(12.0, 20.0, len(idx)),
                        "pm10": np.linspace(20.0, 30.0, len(idx)),
                        "carbon_monoxide": np.linspace(0.4, 0.7, len(idx)),
                        "nitrogen_dioxide": np.linspace(10.0, 18.0, len(idx)),
                        "sulphur_dioxide": np.linspace(4.0, 6.0, len(idx)),
                        "ozone": np.linspace(35.0, 42.0, len(idx)),
                    },
                    index=idx,
                )
                w_hist = pd.DataFrame(
                    {
                        "temperature_2m": np.linspace(27.0, 29.0, len(idx)),
                        "relative_humidity_2m": np.linspace(60.0, 70.0, len(idx)),
                        "surface_pressure": np.linspace(100.5, 101.1, len(idx)),
                        "wind_speed_10m": np.linspace(1.8, 3.1, len(idx)),
                    },
                    index=idx,
                )
                return aq_hist, w_hist

            def fetch_current(
                self,
                lat: float,
                lon: float,
                timezone: str = "auto",
                timeout_seconds: float | None = None,
                max_retries: int | None = None,
            ):
                self.calls.append(
                    ("current", float(self.timeout), int(self.max_retries), timeout_seconds, max_retries)
                )
                ts = pd.Timestamp.now(tz="UTC").floor("h").isoformat()
                aq_cur = {
                    "time": ts,
                    "pm2_5": 22.0,
                    "pm10": 33.0,
                    "carbon_monoxide": 0.72,
                    "nitrogen_dioxide": 20.0,
                    "sulphur_dioxide": 7.0,
                    "ozone": 45.0,
                }
                w_cur = {
                    "time": ts,
                    "temperature_2m": 29.0,
                    "relative_humidity_2m": 68.0,
                    "surface_pressure": 101.2,
                    "wind_speed_10m": 3.2,
                }
                return aq_cur, w_cur

        spy_client = _SpyClient()
        svc = BaselineContextService(data_client=spy_client, history_assembler=HistoryAssembler())
        ctx = svc.get_custom_context(
            lat=6.92,
            lon=79.86,
            target_hours=24,
            meta={"feature_defaults": {}},
            timezone="auto",
        )

        self.assertEqual(ctx.source, "live_api")
        self.assertTrue(ctx.live_data_used)
        self.assertEqual(spy_client.timeout, 20)
        self.assertEqual(spy_client.max_retries, 3)
        self.assertEqual(len(spy_client.calls), 2)
        for call in spy_client.calls:
            _kind, timeout_now, retries_now, timeout_override, retries_override = call
            self.assertEqual(timeout_now, 20.0)
            self.assertEqual(retries_now, 3)
            self.assertEqual(retries_override, 0)
            self.assertAlmostEqual(float(timeout_override), 6.0, places=6)

    def test_live_unit_normalization_feeds_feature_builder(self):
        client = DataClient(timeout_seconds=1, cache_ttl_seconds=0, max_retries=0)
        ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        aq_payload = {
            "current": {
                "time": ts.isoformat(),
                "pm2_5": 20.0,
                "pm10": 35.0,
                "carbon_monoxide": 680.0,  # Open-Meteo ug/m^3
                "nitrogen_dioxide": 18.0,
                "sulphur_dioxide": 7.0,
                "ozone": 48.0,
            }
        }
        weather_payload = {
            "current": {
                "time": ts.isoformat(),
                "temperature_2m": 29.0,
                "relative_humidity_2m": 71.0,
                "surface_pressure": 1007.4,  # Open-Meteo hPa
                "wind_speed_10m": 14.4,  # Open-Meteo km/h
            }
        }

        with patch(
            "backend.app.services.data_client.requests.get",
            side_effect=[_MockResponse(aq_payload), _MockResponse(weather_payload)],
        ):
            aq_cur, w_cur = client.fetch_current(lat=6.92, lon=79.86, timezone="auto")

        self.assertAlmostEqual(float(aq_cur["carbon_monoxide"]), 0.68, places=6)
        self.assertAlmostEqual(float(w_cur["surface_pressure"]), 100.74, places=6)
        self.assertAlmostEqual(float(w_cur["wind_speed_10m"]), 4.0, places=6)

        idx = pd.date_range(end=ts - timedelta(hours=1), periods=8, freq="h")
        history_df = pd.DataFrame({"PM2.5": np.linspace(12.0, 18.0, len(idx))}, index=idx)
        meta = {"features": ["CO", "pressure", "wind_speed"], "feature_defaults": {}}
        feat = FeatureBuilder().build(meta=meta, history_df=history_df, aq_cur=aq_cur, w_cur=w_cur)
        row = feat.X.iloc[0]

        self.assertAlmostEqual(float(row["CO"]), 0.68, places=6)
        self.assertAlmostEqual(float(row["pressure"]), 100.74, places=6)
        self.assertAlmostEqual(float(row["wind_speed"]), 4.0, places=6)

    def test_live_current_sanitizes_physically_invalid_values(self):
        client = DataClient(timeout_seconds=1, cache_ttl_seconds=0, max_retries=0)
        ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        aq_payload = {
            "current": {
                "time": ts.isoformat(),
                "pm2_5": -3.0,
                "pm10": -8.0,
                "carbon_monoxide": -10.0,
                "nitrogen_dioxide": -2.0,
                "sulphur_dioxide": -1.0,
                "ozone": -4.0,
            }
        }
        weather_payload = {
            "current": {
                "time": ts.isoformat(),
                "temperature_2m": 180.0,
                "relative_humidity_2m": 140.0,
                "surface_pressure": 1500.0,
                "wind_speed_10m": -4.0,
            }
        }

        with patch(
            "backend.app.services.data_client.requests.get",
            side_effect=[_MockResponse(aq_payload), _MockResponse(weather_payload)],
        ):
            aq_cur, w_cur = client.fetch_current(lat=6.92, lon=79.86, timezone="auto")

        self.assertIsNone(aq_cur.get("pm2_5"))
        self.assertIsNone(aq_cur.get("pm10"))
        self.assertIsNone(aq_cur.get("carbon_monoxide"))
        self.assertIsNone(w_cur.get("temperature_2m"))
        self.assertIsNone(w_cur.get("relative_humidity_2m"))
        self.assertIsNone(w_cur.get("surface_pressure"))
        self.assertIsNone(w_cur.get("wind_speed_10m"))

    def test_feature_builder_flags_pm25_current_zero_and_tail_events(self):
        ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        idx = pd.date_range(end=ts - timedelta(hours=1), periods=12, freq="h")
        history_df = pd.DataFrame({"PM2.5": np.linspace(12.0, 22.0, len(idx))}, index=idx)
        meta = {
            "features": ["lag1"],
            "feature_defaults": {},
            "feature_quantiles": {"lag1": {"q01": 8.0, "q99": 25.0}},
        }
        builder = FeatureBuilder()

        zero_feat = builder.build(
            meta=meta,
            history_df=history_df,
            aq_cur={"time": ts.isoformat(), "pm2_5": 0.0},
            w_cur={},
        )
        self.assertTrue(
            any(
                evt.get("feature") == "PM2.5_current" and evt.get("side") == "suspicious_zero"
                for evt in zero_feat.extreme_current_events
            )
        )

        high_feat = builder.build(
            meta=meta,
            history_df=history_df,
            aq_cur={"time": ts.isoformat(), "pm2_5": 40.0},
            w_cur={},
        )
        self.assertTrue(
            any(
                evt.get("feature") == "PM2.5_current" and evt.get("side") == "above_q99"
                for evt in high_feat.extreme_current_events
            )
        )

    def test_custom_pressure_override_hpa_converts_before_scenario_engine(self):
        overrides_model, pressure_converted = routes._normalize_custom_overrides_for_model({"pressure": 1013.0})
        self.assertTrue(pressure_converted)
        self.assertAlmostEqual(float(overrides_model["pressure"]), 101.3, places=6)

        baseline_X = pd.DataFrame([{"pressure": 100.9}], columns=["pressure"])
        meta = {
            "feature_quantiles": {
                "pressure": {"q01": 99.5, "q05": 100.0, "q95": 102.2, "q99": 102.8},
            }
        }
        engine = ScenarioEngine()
        scenario_X, applied, _ood_events = engine.apply_value_overrides(
            overrides=overrides_model,
            baseline_X=baseline_X,
            meta=meta,
            impact_mode="conservative",
            ood_opts={"soft_q": 0.05, "hard_q": 0.01},
        )

        self.assertEqual(len(applied), 1)
        row = applied[0]
        self.assertEqual(row["feature"], "pressure")
        self.assertFalse(bool(row["clamped"]))
        self.assertAlmostEqual(float(row["to"]), 101.3, places=6)
        self.assertAlmostEqual(float(scenario_X.iloc[0]["pressure"]), 101.3, places=6)

    def test_scenario_intent_is_not_silently_inverted_when_bounds_block_direction(self):
        engine = ScenarioEngine()
        baseline_X = pd.DataFrame([{"NO2": 34.0}], columns=["NO2"])
        meta = {
            "feature_quantiles": {
                "NO2": {"q01": 5.0, "q05": 10.0, "q95": 30.0, "q99": 35.0},
            }
        }
        scenario_req = SimpleNamespace(type="macro", scenario_id="traffic_gridlock", intensity=100, items=None)
        scenario_X, applied, ood_events, _scenario_id, _ood_ctx = engine.apply(
            scenario=scenario_req,
            baseline_X=baseline_X,
            meta=meta,
            ood_opts={"soft_q": 0.05, "hard_q": 0.01},
            return_context=True,
        )

        self.assertEqual(len(applied), 1)
        row = applied[0]
        self.assertEqual(row["feature"], "NO2")
        self.assertEqual(row["requested_direction"], "increase")
        self.assertEqual(row["effective_direction"], "unchanged")
        self.assertTrue(bool(row["direction_limited"]))
        self.assertGreaterEqual(float(row["to"]), float(row["from"]))
        self.assertIn("could not be applied", str(row["reason"]))
        self.assertGreaterEqual(len(ood_events), 1)
        self.assertAlmostEqual(float(scenario_X.iloc[0]["NO2"]), 34.0, places=6)

    def test_logger_disabled_startup_does_not_depend_on_sqlite_path(self):
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["SKYNET_ENABLE_RUN_LOGGING"] = "0"
        env["SKYNET_SQLITE_PATH"] = "/root/not_writable/skynet_runs.db"

        proc = subprocess.run(
            [sys.executable, "-c", "from backend.app.main import app; print('startup-ok')"],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("startup-ok", proc.stdout)

    def test_readyz_returns_503_when_runtime_status_not_ready(self):
        original_runtime_status = app_main.runtime_status_payload
        try:
            app_main.runtime_status_payload = lambda: {
                "ready": False,
                "checks": {"model_meta_loaded": False},
                "details": {"model_meta_error": "FileNotFoundError: missing model"},
            }
            client = TestClient(app_main.app)
            resp = client.get("/readyz")
            self.assertEqual(resp.status_code, 503)
            body = resp.json()
            self.assertFalse(bool(body.get("ready")))
            self.assertIn("checks", body)
        finally:
            app_main.runtime_status_payload = original_runtime_status

    def test_readyz_returns_200_when_runtime_status_ready(self):
        original_runtime_status = app_main.runtime_status_payload
        try:
            app_main.runtime_status_payload = lambda: {
                "ready": True,
                "checks": {"model_meta_loaded": True, "model_artifact_present": True},
                "details": {"features_count": 36},
            }
            client = TestClient(app_main.app)
            resp = client.get("/readyz")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(bool(resp.json().get("ready")))
        finally:
            app_main.runtime_status_payload = original_runtime_status

    def test_runtime_status_exposes_run_logging_degraded_state(self):
        original_enabled = routes.settings.ENABLE_RUN_LOGGING
        original_logger = routes.run_logger
        original_init_error = routes.run_logger_init_error
        original_get_meta = routes.model_runner.get_meta
        try:
            routes.settings.ENABLE_RUN_LOGGING = True
            routes.run_logger = None
            routes.run_logger_init_error = "RuntimeError: sqlite open failed"
            routes.model_runner.get_meta = lambda: {"features": ["lag1"], "model": object()}

            payload = routes.runtime_status_payload()
            self.assertIn("details", payload)
            self.assertEqual(payload["details"].get("run_logging_status"), "degraded")
            self.assertFalse(bool(payload["checks"].get("run_logging_ready_or_optional")))
            self.assertIn("failed", str(payload["details"].get("run_logging_note", "")).lower())
        finally:
            routes.settings.ENABLE_RUN_LOGGING = original_enabled
            routes.run_logger = original_logger
            routes.run_logger_init_error = original_init_error
            routes.model_runner.get_meta = original_get_meta

    def test_live_like_converted_values_align_with_training_quantile_band(self):
        raw_co = 680.0
        raw_pressure = 1007.4
        raw_wind = 14.7
        aq_norm = DataClient._normalize_current_air_quality_units({"carbon_monoxide": 680.0})
        w_norm = DataClient._normalize_current_weather_units(
            {"surface_pressure": 1007.4, "wind_speed_10m": 14.7}
        )

        # Training quantiles observed in current SKYNET metadata/audit snapshots.
        co_q99 = 1.949
        pressure_q99 = 102.454
        wind_q99 = 2.6
        self.assertLess(abs(float(aq_norm["carbon_monoxide"]) - co_q99), abs(raw_co - co_q99))
        self.assertLess(abs(float(w_norm["surface_pressure"]) - pressure_q99), abs(raw_pressure - pressure_q99))
        self.assertLess(abs(float(w_norm["wind_speed_10m"]) - wind_q99), abs(raw_wind - wind_q99))


if __name__ == "__main__":
    unittest.main()
