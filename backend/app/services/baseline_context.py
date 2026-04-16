from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.settings import settings


HISTORY_COLS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "CO",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
]


@dataclass
class BaselineContext:
    history_df: pd.DataFrame
    history_stats: Dict
    aq_cur: Dict
    w_cur: Dict
    source: str
    live_data_used: bool
    notes: List[str]


class BaselineContextService:
    def __init__(self, data_client, history_assembler):
        self.data_client = data_client
        self.history_assembler = history_assembler
        self._reference_profile: Optional[pd.DataFrame] = None

    @staticmethod
    def _compute_stats(df: pd.DataFrame, target_hours: int) -> Dict:
        available_hours = int(len(df))
        used_hours = int(min(max(0, int(target_hours)), available_hours))
        used = df.iloc[-used_hours:].copy() if used_hours else pd.DataFrame(columns=df.columns)

        gap_count = 0
        largest_gap_hours = 0
        if used_hours >= 2:
            diffs = used.index.to_series().diff().dropna().dt.total_seconds().div(3600)
            if not diffs.empty:
                gap_count = int((diffs > 1).sum())
                largest_gap_hours = int(diffs.max())

        return {
            "target_hours": int(target_hours),
            "available_hours": int(available_hours),
            "used_hours": int(used_hours),
            "gap_count": int(gap_count),
            "largest_gap_hours": int(largest_gap_hours),
        }

    @staticmethod
    def _row_to_current_payload(ts: pd.Timestamp, row: pd.Series):
        aq_cur = {
            "time": ts.isoformat(),
            "pm2_5": float(row.get("PM2.5", np.nan)),
            "pm10": float(row.get("PM10", np.nan)),
            "carbon_monoxide": float(row.get("CO", np.nan)),
            "nitrogen_dioxide": float(row.get("NO2", np.nan)),
            "sulphur_dioxide": float(row.get("SO2", np.nan)),
            "ozone": float(row.get("O3", np.nan)),
        }
        w_cur = {
            "time": ts.isoformat(),
            "temperature_2m": float(row.get("temperature", np.nan)),
            "relative_humidity_2m": float(row.get("humidity", np.nan)),
            "surface_pressure": float(row.get("pressure", np.nan)),
            "wind_speed_10m": float(row.get("wind_speed", np.nan)),
        }
        return aq_cur, w_cur

    def _load_reference_profile(self) -> Optional[pd.DataFrame]:
        if self._reference_profile is not None:
            return self._reference_profile.copy()

        candidate_sources = [
            settings.ROOT_DIR / "data" / "processed" / "final_dataset" / "final.csv",
            settings.ROOT_DIR / "data" / "processed" / "final_dataset",
            settings.ROOT_DIR / "data" / "processed" / "raw_cleaned_audit" / "cleaned_station_files",
        ]

        csv_files: List[str] = []
        for source in candidate_sources:
            if source.is_file() and source.suffix.lower() == ".csv":
                csv_files = [str(source)]
                break
            if source.is_dir():
                files = sorted(glob.glob(str(source / "*.csv")))
                if files:
                    csv_files = files
                    break

        if not csv_files:
            return None

        frames = []
        for p in csv_files:
            try:
                frames.append(pd.read_csv(p, on_bad_lines="skip"))
            except TypeError:
                frames.append(pd.read_csv(p))
            except Exception:
                continue
        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        if "hours" not in df.columns:
            return None

        df["datetime"] = pd.to_datetime(df["hours"], errors="coerce")
        df = df.dropna(subset=["datetime"]).copy()
        if df.empty:
            return None

        for c in HISTORY_COLS:
            if c not in df.columns:
                return None

        num_df = df[["datetime"] + HISTORY_COLS].copy()
        for c in HISTORY_COLS:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce")

        profile = (
            num_df.groupby("datetime", as_index=True)[HISTORY_COLS]
            .median()
            .sort_index()
        )
        profile = profile.resample("h").ffill().bfill()
        profile = profile.dropna(how="any")
        if profile.empty:
            return None

        self._reference_profile = profile
        return profile.copy()

    def _reference_context(self, target_hours: int) -> Optional[BaselineContext]:
        profile = self._load_reference_profile()
        if profile is None or profile.empty:
            return None

        stats = self._compute_stats(profile, target_hours=target_hours)
        used = profile.iloc[-stats["used_hours"] :].copy()
        if used.empty:
            return None

        ts = pd.Timestamp(used.index[-1])
        row = used.iloc[-1]
        aq_cur, w_cur = self._row_to_current_payload(ts, row)
        return BaselineContext(
            history_df=used,
            history_stats=stats,
            aq_cur=aq_cur,
            w_cur=w_cur,
            source="reference_profile",
            live_data_used=False,
            notes=[
                "Using dataset-derived global reference baseline profile (not location-specific).",
            ],
        )

    @staticmethod
    def _default_value(meta: Dict, feature: str, fallback: float) -> float:
        try:
            defaults = meta.get("feature_defaults", {}) or {}
            val = float(defaults.get(feature, fallback))
            if np.isfinite(val):
                return float(val)
        except Exception:
            pass
        return float(fallback)

    def _demo_context(self, target_hours: int, meta: Dict) -> BaselineContext:
        end = pd.Timestamp.utcnow().floor("h") - pd.Timedelta(hours=1)
        n_hours = max(24, int(target_hours))
        idx = pd.date_range(end=end, periods=n_hours, freq="h")

        pm25_default = self._default_value(
            meta,
            feature="lag1",
            fallback=float(meta.get("training_pm25_mean", 10.0) or 10.0),
        )
        row = {
            "PM2.5": pm25_default,
            "PM10": self._default_value(meta, "PM10", pm25_default * 1.8),
            "NO2": self._default_value(meta, "NO2", 10.0),
            "SO2": self._default_value(meta, "SO2", 5.0),
            "O3": self._default_value(meta, "O3", 40.0),
            "CO": self._default_value(meta, "CO", 0.5),
            "temperature": self._default_value(meta, "temperature", 28.0),
            "humidity": self._default_value(meta, "humidity", 65.0),
            # Pressure fallback is model-facing kPa-equivalent (not raw hPa).
            "pressure": self._default_value(meta, "pressure", 101.0),
            "wind_speed": self._default_value(meta, "wind_speed", 2.5),
        }

        profile = pd.DataFrame([row] * len(idx), index=idx, columns=HISTORY_COLS)
        stats = self._compute_stats(profile, target_hours=target_hours)
        used = profile.iloc[-stats["used_hours"] :].copy()
        ts = pd.Timestamp(used.index[-1])
        aq_cur, w_cur = self._row_to_current_payload(ts, used.iloc[-1])
        return BaselineContext(
            history_df=used,
            history_stats=stats,
            aq_cur=aq_cur,
            w_cur=w_cur,
            source="demo_default",
            live_data_used=False,
            notes=["Using demo default baseline profile because live/reference baselines are unavailable."],
        )

    def get_custom_context(self, lat: float, lon: float, target_hours: int, meta: Dict, timezone: str = "auto") -> BaselineContext:
        notes: List[str] = []
        try:
            hist_key = ("history", round(float(lat), 4), round(float(lon), 4), int(target_hours), timezone)
            cur_key = ("current", round(float(lat), 4), round(float(lon), 4), timezone)
            cached_hist = self.data_client._get_cached(hist_key)
            cached_cur = self.data_client._get_cached(cur_key)

            if cached_hist is not None and cached_cur is not None:
                aq_hist, w_hist = cached_hist
                aq_cur, w_cur = cached_cur
                hist_df, hist_stats = self.history_assembler.assemble(
                    aq_hist=aq_hist,
                    w_hist=w_hist,
                    target_hours=target_hours,
                )
                if not hist_df.empty:
                    return BaselineContext(
                        history_df=hist_df,
                        history_stats=hist_stats,
                        aq_cur=aq_cur,
                        w_cur=w_cur,
                        source="live_api",
                        live_data_used=True,
                        notes=["Using recent cached live baseline context."],
                    )

            # Keep custom-mode fallback responsive: use per-call probe settings
            # without mutating shared DataClient singleton state.
            probe_timeout = min(float(self.data_client.timeout), 6.0)
            aq_hist, w_hist = self.data_client.fetch_history(
                lat=lat,
                lon=lon,
                hours=target_hours,
                timezone=timezone,
                timeout_seconds=probe_timeout,
                max_retries=0,
            )
            aq_cur, w_cur = self.data_client.fetch_current(
                lat=lat,
                lon=lon,
                timezone=timezone,
                timeout_seconds=probe_timeout,
                max_retries=0,
            )

            hist_df, hist_stats = self.history_assembler.assemble(
                aq_hist=aq_hist,
                w_hist=w_hist,
                target_hours=target_hours,
            )
            if not hist_df.empty:
                return BaselineContext(
                    history_df=hist_df,
                    history_stats=hist_stats,
                    aq_cur=aq_cur,
                    w_cur=w_cur,
                    source="live_api",
                    live_data_used=True,
                    notes=["Using live Open-Meteo baseline context."],
                )
            notes.append("Live history probe returned no usable rows.")
        except Exception as exc:
            notes.append(f"Live baseline unavailable: {exc}")

        reference = self._reference_context(target_hours=target_hours)
        if reference is not None:
            reference.notes.extend(notes)
            return reference

        demo = self._demo_context(target_hours=target_hours, meta=meta)
        demo.notes.extend(notes)
        return demo
