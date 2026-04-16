from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests


AQ_ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Open-Meteo current/history weather-air quality payloads use:
# - carbon_monoxide: ug/m^3
# - surface_pressure: hPa
# - wind_speed_10m: km/h (default)
# SKYNET model features in the current final pipeline (trained from
# data/processed/final_dataset/final.csv) use the following effective units:
# these effectively appear as:
# - CO: mg/m^3-equivalent
# - pressure: kPa-equivalent
# - wind_speed: m/s-equivalent
# Convert explicitly at ingest so runtime features are unit-consistent.
CO_UG_PER_M3_TO_MG_PER_M3 = 1.0 / 1000.0
PRESSURE_HPA_TO_KPA = 0.1
WIND_KMH_TO_MS = 1.0 / 3.6
CO_LIKELY_UG_THRESHOLD = 20.0
PRESSURE_LIKELY_HPA_THRESHOLD = 200.0
PRESSURE_KPA_MIN = 80.0
PRESSURE_KPA_MAX = 120.0
WIND_MS_MAX = 75.0
TEMPERATURE_C_MIN = -90.0
TEMPERATURE_C_MAX = 65.0


class DataClient:
    def __init__(
        self,
        timeout_seconds: int = 20,
        cache_ttl_seconds: int = 300,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.35,
    ):
        self.timeout = timeout_seconds
        self.cache_ttl = cache_ttl_seconds
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_seconds = max(0.0, float(backoff_base_seconds))
        self._cache: Dict[Tuple, Tuple[float, object]] = {}

    def _get_cached(self, key):
        now = datetime.now(timezone.utc).timestamp()
        if key in self._cache:
            ts, value = self._cache[key]
            if (now - ts) <= self.cache_ttl:
                return value
        return None

    def _set_cached(self, key, value):
        self._cache[key] = (datetime.now(timezone.utc).timestamp(), value)

    @staticmethod
    def _iso_date(dt):
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return status in RETRYABLE_STATUS_CODES
        return False

    @staticmethod
    def _safe_number(v):
        try:
            x = float(v)
        except Exception:
            return None
        return x if pd.notna(x) else None

    @classmethod
    def _nonnegative_or_none(cls, raw_value):
        value = cls._safe_number(raw_value)
        if value is None:
            return None
        return float(value) if value >= 0.0 else None

    @classmethod
    def _in_range_or_none(cls, raw_value, low: float, high: float):
        value = cls._safe_number(raw_value)
        if value is None:
            return None
        if value < float(low) or value > float(high):
            return None
        return float(value)

    @classmethod
    def _normalize_co_value(cls, raw_value):
        """
        Convert provider CO payload to SKYNET model representation.
        If upstream payload is already mg/m^3-like, keep value unchanged.
        """
        value = cls._safe_number(raw_value)
        if value is None:
            return None
        if abs(value) > CO_LIKELY_UG_THRESHOLD:
            return float(value * CO_UG_PER_M3_TO_MG_PER_M3)
        return float(value)

    @classmethod
    def _normalize_pressure_value(cls, raw_value):
        """
        Convert provider pressure payload to SKYNET model representation.
        If payload is already kPa-like, keep value unchanged.
        """
        value = cls._safe_number(raw_value)
        if value is None:
            return None
        if abs(value) > PRESSURE_LIKELY_HPA_THRESHOLD:
            return float(value * PRESSURE_HPA_TO_KPA)
        return float(value)

    @classmethod
    def _normalize_current_air_quality_units(cls, aq_current: Dict):
        out = dict(aq_current or {})
        co_norm = cls._normalize_co_value(out.get("carbon_monoxide"))
        if co_norm is not None:
            out["carbon_monoxide"] = co_norm

        for key in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]:
            sanitized = cls._nonnegative_or_none(out.get(key))
            if sanitized is None:
                out[key] = None
            else:
                out[key] = sanitized

        for key in ["us_aqi", "european_aqi"]:
            sanitized = cls._nonnegative_or_none(out.get(key))
            if sanitized is not None:
                out[key] = sanitized
        return out

    @classmethod
    def _normalize_current_weather_units(cls, weather_current: Dict):
        out = dict(weather_current or {})
        pressure_norm = cls._normalize_pressure_value(out.get("surface_pressure"))
        if pressure_norm is not None:
            out["surface_pressure"] = pressure_norm

        wind_kmh = cls._safe_number(out.get("wind_speed_10m"))
        if wind_kmh is not None:
            out["wind_speed_10m"] = float(wind_kmh * WIND_KMH_TO_MS)

        out["relative_humidity_2m"] = cls._in_range_or_none(out.get("relative_humidity_2m"), 0.0, 100.0)
        out["surface_pressure"] = cls._in_range_or_none(out.get("surface_pressure"), PRESSURE_KPA_MIN, PRESSURE_KPA_MAX)
        out["wind_speed_10m"] = cls._in_range_or_none(out.get("wind_speed_10m"), 0.0, WIND_MS_MAX)
        out["temperature_2m"] = cls._in_range_or_none(out.get("temperature_2m"), TEMPERATURE_C_MIN, TEMPERATURE_C_MAX)

        return out

    @classmethod
    def _normalize_history_air_quality_units(cls, aq_df: pd.DataFrame) -> pd.DataFrame:
        if aq_df is None or aq_df.empty:
            return aq_df
        out = aq_df.copy()
        if "carbon_monoxide" in out.columns:
            co = pd.to_numeric(out["carbon_monoxide"], errors="coerce")
            out["carbon_monoxide"] = np.where(
                co.abs() > CO_LIKELY_UG_THRESHOLD,
                co * CO_UG_PER_M3_TO_MG_PER_M3,
                co,
            )
        for key in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]:
            if key in out.columns:
                values = pd.to_numeric(out[key], errors="coerce")
                out[key] = np.where(values >= 0.0, values, np.nan)
        return out

    @classmethod
    def _normalize_history_weather_units(cls, weather_df: pd.DataFrame) -> pd.DataFrame:
        if weather_df is None or weather_df.empty:
            return weather_df
        out = weather_df.copy()
        if "surface_pressure" in out.columns:
            pressure = pd.to_numeric(out["surface_pressure"], errors="coerce")
            out["surface_pressure"] = np.where(
                pressure.abs() > PRESSURE_LIKELY_HPA_THRESHOLD,
                pressure * PRESSURE_HPA_TO_KPA,
                pressure,
            )
        if "wind_speed_10m" in out.columns:
            wind = pd.to_numeric(out["wind_speed_10m"], errors="coerce")
            out["wind_speed_10m"] = wind * WIND_KMH_TO_MS
        if "relative_humidity_2m" in out.columns:
            humidity = pd.to_numeric(out["relative_humidity_2m"], errors="coerce")
            out["relative_humidity_2m"] = np.where(
                (humidity >= 0.0) & (humidity <= 100.0),
                humidity,
                np.nan,
            )
        if "surface_pressure" in out.columns:
            pressure = pd.to_numeric(out["surface_pressure"], errors="coerce")
            out["surface_pressure"] = np.where(
                (pressure >= PRESSURE_KPA_MIN) & (pressure <= PRESSURE_KPA_MAX),
                pressure,
                np.nan,
            )
        if "wind_speed_10m" in out.columns:
            wind = pd.to_numeric(out["wind_speed_10m"], errors="coerce")
            out["wind_speed_10m"] = np.where(
                (wind >= 0.0) & (wind <= WIND_MS_MAX),
                wind,
                np.nan,
            )
        if "temperature_2m" in out.columns:
            temp = pd.to_numeric(out["temperature_2m"], errors="coerce")
            out["temperature_2m"] = np.where(
                (temp >= TEMPERATURE_C_MIN) & (temp <= TEMPERATURE_C_MAX),
                temp,
                np.nan,
            )
        return out

    def _request_json(
        self,
        url: str,
        params: Dict,
        source: str,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        timeout = float(self.timeout if timeout_seconds is None else timeout_seconds)
        retries = int(self.max_retries if max_retries is None else max_retries)
        attempts = max(0, retries) + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                status = int(getattr(resp, "status_code", 0) or 0)
                if status in RETRYABLE_STATUS_CODES:
                    raise requests.HTTPError(
                        f"{source} returned retryable status {status}",
                        response=resp,
                    )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= (attempts - 1) or not self._is_retryable_error(exc):
                    break
                sleep_seconds = min(2.0, self.backoff_base_seconds * (2 ** attempt))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except ValueError as exc:
                last_exc = exc
                break

        raise RuntimeError(f"{source} request failed after {attempts} attempt(s): {last_exc}")

    def fetch_history(
        self,
        lat: float,
        lon: float,
        hours: int = 72,
        timezone: str = "auto",
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        key = ("history", round(lat, 4), round(lon, 4), int(hours), timezone)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        end = datetime.utcnow()
        start = end - timedelta(hours=max(hours - 1, 1))
        start_date = self._iso_date(start)
        end_date = self._iso_date(end)

        aq_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ],
            "timezone": timezone,
        }
        aq_json = self._request_json(
            AQ_ENDPOINT,
            aq_params,
            source="Open-Meteo air-quality history",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        aq_df = pd.DataFrame((aq_json or {}).get("hourly", {}))
        if not aq_df.empty and "time" in aq_df.columns:
            aq_df["time"] = pd.to_datetime(aq_df["time"])
            aq_df = aq_df.set_index("time")
        aq_df = self._normalize_history_air_quality_units(aq_df)

        w_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"],
            "timezone": timezone,
            "wind_speed_unit": "kmh",
        }
        w_json = self._request_json(
            WEATHER_ARCHIVE,
            w_params,
            source="Open-Meteo weather history",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        w_df = pd.DataFrame((w_json or {}).get("hourly", {}))
        if not w_df.empty and "time" in w_df.columns:
            w_df["time"] = pd.to_datetime(w_df["time"])
            w_df = w_df.set_index("time")
        w_df = self._normalize_history_weather_units(w_df)

        if aq_df.empty or w_df.empty:
            result = (aq_df, w_df)
            self._set_cached(key, result)
            return result

        common_index = aq_df.index.intersection(w_df.index).sort_values()
        if len(common_index) > hours:
            common_index = common_index[-hours:]

        result = (aq_df.loc[common_index].copy(), w_df.loc[common_index].copy())
        self._set_cached(key, result)
        return result

    def fetch_current(
        self,
        lat: float,
        lon: float,
        timezone: str = "auto",
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        key = ("current", round(lat, 4), round(lon, 4), timezone)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        aq_params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "us_aqi",
                "european_aqi",
            ],
            "timezone": timezone,
        }
        aq_json = self._request_json(
            AQ_ENDPOINT,
            aq_params,
            source="Open-Meteo air-quality current",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        aq_cur = self._normalize_current_air_quality_units((aq_json or {}).get("current", {}))

        w_params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"],
            "timezone": timezone,
            "wind_speed_unit": "kmh",
        }
        w_json = self._request_json(
            WEATHER_FORECAST,
            w_params,
            source="Open-Meteo weather current",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        w_cur = self._normalize_current_weather_units((w_json or {}).get("current", {}))

        result = (aq_cur, w_cur)
        self._set_cached(key, result)
        return result
