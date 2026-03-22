from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


RENAME_MAP = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
}


class HistoryAssembler:
    def assemble(self, aq_hist: pd.DataFrame, w_hist: pd.DataFrame, target_hours: int) -> Tuple[pd.DataFrame, Dict]:
        # aq_hist/w_hist are expected to already be normalized to SKYNET model-facing
        # representation by DataClient (CO mg/m^3-eq, pressure kPa-eq, wind_speed m/s-eq).
        if aq_hist.empty or w_hist.empty:
            return pd.DataFrame(), {
                "target_hours": int(target_hours),
                "available_hours": 0,
                "used_hours": 0,
                "gap_count": 0,
                "largest_gap_hours": 0,
            }

        df = aq_hist.join(w_hist, how="inner").rename(columns=RENAME_MAP)
        df = df.sort_index()

        available_hours = int(len(df))
        used_hours = int(min(target_hours, available_hours))
        used = df.iloc[-used_hours:].copy() if used_hours else pd.DataFrame()

        gap_count = 0
        largest_gap_hours = 0
        if available_hours >= 2:
            diffs = used.index.to_series().diff().dropna().dt.total_seconds().div(3600)
            if not diffs.empty:
                gap_count = int((diffs > 1).sum())
                largest_gap_hours = int(diffs.max()) if len(diffs) else 0

        stats = {
            "target_hours": int(target_hours),
            "available_hours": available_hours,
            "used_hours": used_hours,
            "gap_count": gap_count,
            "largest_gap_hours": largest_gap_hours,
        }
        return used, stats
