from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.api import routes
from backend.app.core.settings import settings
from backend.app.main import app
from backend.app.services.history_assembler import HistoryAssembler


def _make_history(hours: int = 96):
    end = datetime(2025, 3, 1, 11, 0, 0, tzinfo=timezone.utc)
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


def _payloads() -> Dict[str, Dict]:
    common = {
        "location": {"lat": 20.0440, "lon": 110.1983, "name": "Haikou", "location_id": "haikou_cn"},
        "time": {"mode": "now"},
        "options": {"history_hours_target": 72, "top_k_drivers": 6, "ood": {"soft_q": 0.05, "hard_q": 0.01}},
    }
    return {
        "baseline_live_forecast": {
            **common,
            "request_id": "timing-baseline-live",
            "forecast_mode": "live",
            "scenario": {"type": "custom", "intensity": 0, "items": []},
        },
        "macro_scenario_forecast": {
            **common,
            "request_id": "timing-macro",
            "forecast_mode": "live",
            "scenario": {"type": "macro", "scenario_id": "traffic_gridlock", "intensity": 70},
        },
        "guided_intervention_forecast": {
            **common,
            "request_id": "timing-guided",
            "forecast_mode": "live",
            "scenario": {
                "type": "custom",
                "intensity": 70,
                "items": [
                    {"category": "wind", "direction": "increase", "magnitude": "medium"},
                    {"category": "emission_proxy", "direction": "decrease", "magnitude": "small"},
                ],
            },
        },
        "manual_custom_forecast": {
            **common,
            "request_id": "timing-manual-custom",
            "forecast_mode": "custom",
            "custom_impact_mode": "conservative",
            "scenario": {"type": "custom", "intensity": 0, "items": []},
            "custom_overrides": {"PM10": 45.0, "NO2": 24.0, "wind_speed": 3.2},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Measure lightweight SKYNET prototype response times.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--out-csv", default=str(Path(__file__).resolve().parent / "prototype_response_time_results.csv"))
    args = parser.parse_args()

    runs = max(1, int(args.runs))
    warmup_runs = max(0, int(args.warmup_runs))
    out_csv = Path(args.out_csv).resolve()

    aq_hist, w_hist = _make_history(hours=96)
    ts = aq_hist.index[-1] + timedelta(hours=1)
    aq_cur, w_cur = _make_current(ts)
    hist_df, hist_stats = HistoryAssembler().assemble(aq_hist=aq_hist, w_hist=w_hist, target_hours=72)

    original_fetch_history = routes.data_client.fetch_history
    original_fetch_current = routes.data_client.fetch_current
    original_get_custom_context = routes.baseline_context_service.get_custom_context
    original_run_logging = bool(settings.ENABLE_RUN_LOGGING)

    rows = []
    client = TestClient(app)

    try:
        # Keep the timing run deterministic and avoid SQLite side effects.
        settings.ENABLE_RUN_LOGGING = False

        routes.data_client.fetch_history = lambda lat, lon, hours, timezone="auto": (aq_hist.copy(), w_hist.copy())
        routes.data_client.fetch_current = lambda lat, lon, timezone="auto": (dict(aq_cur), dict(w_cur))
        routes.baseline_context_service.get_custom_context = lambda **kwargs: SimpleNamespace(
            history_df=hist_df.copy(),
            history_stats=dict(hist_stats),
            aq_cur=dict(aq_cur),
            w_cur=dict(w_cur),
            source="reference_profile",
            live_data_used=False,
            notes=["Using deterministic benchmark baseline context for prototype timing."],
        )

        for case_name, payload in _payloads().items():
            for _ in range(warmup_runs):
                warm = client.post("/api/v1/forecast/interactive", json=payload)
                if warm.status_code != 200:
                    raise RuntimeError(
                        f"Warm-up failed for case '{case_name}': status={warm.status_code}, body={warm.text[:400]}"
                    )

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                resp = client.post("/api/v1/forecast/interactive", json=payload)
                elapsed = time.perf_counter() - t0
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Timing run failed for case '{case_name}': status={resp.status_code}, body={resp.text[:400]}"
                    )
                times.append(float(elapsed))

            arr = np.asarray(times, dtype=float)
            rows.append(
                {
                    "Test Case": case_name,
                    "Runs": int(runs),
                    "Mean (s)": float(arr.mean()),
                    "Median (s)": float(np.median(arr)),
                    "Min (s)": float(arr.min()),
                    "Max (s)": float(arr.max()),
                    "Notes": "In-process FastAPI timing with deterministic mocked Open-Meteo payloads (prototype-level).",
                }
            )
    finally:
        routes.data_client.fetch_history = original_fetch_history
        routes.data_client.fetch_current = original_fetch_current
        routes.baseline_context_service.get_custom_context = original_get_custom_context
        settings.ENABLE_RUN_LOGGING = original_run_logging

    out_df = pd.DataFrame(rows)
    out_df = out_df[
        ["Test Case", "Runs", "Mean (s)", "Median (s)", "Min (s)", "Max (s)", "Notes"]
    ].sort_values("Test Case")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Response time results written: {out_csv}")


if __name__ == "__main__":
    main()
