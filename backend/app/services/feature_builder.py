from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


AUX_COLS = [
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

EXOGENOUS_LAGROLL_COLS = ["PM10", "humidity", "wind_speed", "pressure"]
INTERACTION_FEATURE_RECIPES = {
    "int_windspeedlag1_pm10lag1": ("wind_speed_lag1", "PM10_lag1"),
    "int_humiditylag1_temperaturelag1": ("humidity_lag1", "temperature_lag1"),
    "int_pressurelag1_windspeedlag1": ("pressure_lag1", "wind_speed_lag1"),
}


@dataclass
class FeatureBuildResult:
    X: pd.DataFrame
    base_lag1: float
    history_points: int
    imputed_features: int
    imputed_feature_names: List[str]
    extreme_current_events: List[Dict]
    current_pm25: float
    observed_timestamp: pd.Timestamp
    prediction_timestamp: pd.Timestamp
    current_aux: Dict[str, float]


class FeatureBuilder:
    @staticmethod
    def _safe_float(v):
        try:
            out = float(v)
        except Exception:
            return np.nan
        return out if np.isfinite(out) else np.nan

    @staticmethod
    def _time_features(ts: pd.Timestamp):
        hour = int(ts.hour)
        day = int(ts.dayofweek)
        return {
            "sin_hour": np.sin(2 * np.pi * hour / 24),
            "cos_hour": np.cos(2 * np.pi * hour / 24),
            "sin_day": np.sin(2 * np.pi * day / 7),
            "cos_day": np.cos(2 * np.pi * day / 7),
            "is_weekend": int(day >= 5),
        }

    @staticmethod
    def _feature_fallback(name: str, base_lag1: float, feat_defaults: Dict):
        if name in feat_defaults and pd.notna(feat_defaults[name]):
            return float(feat_defaults[name])

        # Exogenous lag/rolling fallback should remain in the same feature domain.
        if "_" in str(name):
            base_feature = str(name).split("_", 1)[0]
            if base_feature in AUX_COLS:
                if base_feature in feat_defaults and pd.notna(feat_defaults[base_feature]):
                    return float(feat_defaults[base_feature])
                return 0.0

        if name.startswith("std"):
            return 0.0

        if name.startswith("lag") or name.startswith("roll") or name.startswith("ewm"):
            return float(base_lag1) if pd.notna(base_lag1) else 0.0

        if name in ("min24", "max24", "trend_24", "trend_168", "roll_diff_3_24", "roll_diff_24_168"):
            return float(base_lag1) if pd.notna(base_lag1) else 0.0

        if name in AUX_COLS:
            if name in feat_defaults and pd.notna(feat_defaults[name]):
                return float(feat_defaults[name])
            return 0.0

        return 0.0

    @staticmethod
    def _quantile_bounds(meta: Dict, feature: str):
        q = (meta.get("feature_quantiles", {}) or {}).get(feature, {})
        if not isinstance(q, dict):
            return None
        try:
            q01 = float(q.get("q01"))
            q99 = float(q.get("q99"))
        except Exception:
            return None
        if not np.isfinite(q01) or not np.isfinite(q99) or q01 >= q99:
            return None
        return q01, q99

    def build(self, meta: Dict, history_df: pd.DataFrame, aq_cur: Dict, w_cur: Dict) -> FeatureBuildResult:
        feats: List[str] = list(meta.get("features", []))
        feat_defaults = meta.get("feature_defaults", {})

        ts = pd.to_datetime(aq_cur.get("time"), errors="coerce")
        if pd.isna(ts) and not history_df.empty:
            ts = pd.to_datetime(history_df.index.max(), errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp.utcnow().floor("h")
        pred_ts = ts + pd.Timedelta(hours=1)

        current_pm25 = self._safe_float(aq_cur.get("pm2_5", np.nan))

        history = history_df.copy()
        if not history.empty:
            history.index = pd.to_datetime(history.index, errors="coerce")
            history = history[history.index.notna()].copy()
            history = history.sort_index()
            history = history[~history.index.duplicated(keep="last")]

        idx = pd.DatetimeIndex([])
        if not history.empty:
            idx = idx.append(pd.DatetimeIndex(history.index))
        idx = idx.append(pd.DatetimeIndex([ts, pred_ts])).unique().sort_values()
        tmp = pd.DataFrame(index=idx)

        if "PM2.5" in history.columns:
            tmp["PM2.5"] = pd.to_numeric(history["PM2.5"], errors="coerce").reindex(tmp.index)
        else:
            tmp["PM2.5"] = np.nan
        if pd.notna(current_pm25):
            tmp.loc[ts, "PM2.5"] = float(current_pm25)
        tmp.loc[pred_ts, "PM2.5"] = np.nan
        pm = tmp["PM2.5"].astype(float)
        s = pm.dropna()

        for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
            tmp[f"lag{lag}"] = pm.shift(lag)

        pm_past = pm.shift(1)
        # Match training semantics (default min_periods=window).
        tmp["roll3"] = pm_past.rolling(3).mean()
        tmp["roll6"] = pm_past.rolling(6).mean()
        tmp["roll24"] = pm_past.rolling(24).mean()
        tmp["roll48"] = pm_past.rolling(48).mean()
        tmp["roll168"] = pm_past.rolling(168).mean()

        tmp["std6"] = pm_past.rolling(6).std()
        tmp["std24"] = pm_past.rolling(24).std()
        tmp["min24"] = pm_past.rolling(24).min()
        tmp["max24"] = pm_past.rolling(24).max()

        tmp["ewm6"] = pm_past.ewm(span=6, adjust=False).mean()
        tmp["ewm24"] = pm_past.ewm(span=24, adjust=False).mean()

        tmp["trend_24"] = tmp["lag1"] - tmp["lag24"]
        tmp["trend_168"] = tmp["lag1"] - tmp["lag168"]
        tmp["roll_diff_3_24"] = tmp["roll3"] - tmp["roll24"]
        tmp["roll_diff_24_168"] = tmp["roll24"] - tmp["roll168"]

        for k, v in self._time_features(pred_ts).items():
            tmp.loc[pred_ts, k] = v

        # These current values are consumed directly as model features.
        # DataClient/baseline context must provide SKYNET model representation
        # (not raw provider units) for CO/pressure/wind_speed.
        current_aux = {
            "PM10": aq_cur.get("pm10", np.nan),
            "NO2": aq_cur.get("nitrogen_dioxide", np.nan),
            "SO2": aq_cur.get("sulphur_dioxide", np.nan),
            "O3": aq_cur.get("ozone", np.nan),
            "CO": aq_cur.get("carbon_monoxide", np.nan),
            "temperature": w_cur.get("temperature_2m", np.nan),
            "humidity": w_cur.get("relative_humidity_2m", np.nan),
            "pressure": w_cur.get("surface_pressure", np.nan),
            "wind_speed": w_cur.get("wind_speed_10m", np.nan),
        }

        # Add exogenous history + current values so lag/rolling exogenous features
        # are computed with the same past-only semantics as training.
        for c in AUX_COLS:
            if c in history.columns:
                tmp[c] = pd.to_numeric(history[c], errors="coerce").reindex(tmp.index)
            else:
                tmp[c] = np.nan

        for c in AUX_COLS:
            current_value = self._safe_float(current_aux.get(c, np.nan))
            if pd.notna(current_value):
                tmp.loc[ts, c] = float(current_value)
                tmp.loc[pred_ts, c] = float(current_value)
            else:
                tmp.loc[pred_ts, c] = np.nan

        for col in EXOGENOUS_LAGROLL_COLS:
            s_col = pd.to_numeric(tmp[col], errors="coerce")
            s_past = s_col.shift(1)
            tmp[f"{col}_lag1"] = s_past
            tmp[f"{col}_roll3"] = s_past.rolling(3).mean()
            tmp[f"{col}_roll24"] = s_past.rolling(24).mean()
        if "PM10" in tmp.columns:
            tmp["PM10_lag3"] = pd.to_numeric(tmp["PM10"], errors="coerce").shift(3)
        if "temperature" in tmp.columns:
            tmp["temperature_lag1"] = pd.to_numeric(tmp["temperature"], errors="coerce").shift(1)

        # Keep backward compatibility with any metadata that includes lagged interactions.
        for interaction_feature, (left_col, right_col) in INTERACTION_FEATURE_RECIPES.items():
            left = pd.to_numeric(tmp.get(left_col, np.nan), errors="coerce")
            right = pd.to_numeric(tmp.get(right_col, np.nan), errors="coerce")
            tmp[interaction_feature] = left * right

        extreme_current_events: List[Dict] = []
        for feature in AUX_COLS:
            value = self._safe_float(current_aux.get(feature, np.nan))
            if pd.isna(value):
                continue
            bounds = self._quantile_bounds(meta=meta, feature=feature)
            if bounds is None:
                continue
            q01, q99 = bounds
            if value < q01:
                extreme_current_events.append(
                    {
                        "feature": str(feature),
                        "value": float(value),
                        "q01": float(q01),
                        "q99": float(q99),
                        "side": "below_q01",
                    }
                )
            elif value > q99:
                extreme_current_events.append(
                    {
                        "feature": str(feature),
                        "value": float(value),
                        "q01": float(q01),
                        "q99": float(q99),
                        "side": "above_q99",
                    }
                )

        pm25_bounds = self._quantile_bounds(meta=meta, feature="lag1")
        if pd.notna(current_pm25):
            q01 = q99 = np.nan
            if pm25_bounds is not None:
                q01, q99 = pm25_bounds
            if float(current_pm25) == 0.0:
                extreme_current_events.append(
                    {
                        "feature": "PM2.5_current",
                        "value": float(current_pm25),
                        "q01": float(q01) if np.isfinite(q01) else np.nan,
                        "q99": float(q99) if np.isfinite(q99) else np.nan,
                        "side": "suspicious_zero",
                    }
                )
            elif pm25_bounds is not None:
                if current_pm25 < q01:
                    extreme_current_events.append(
                        {
                            "feature": "PM2.5_current",
                            "value": float(current_pm25),
                            "q01": float(q01),
                            "q99": float(q99),
                            "side": "below_q01",
                        }
                    )
                elif current_pm25 > q99:
                    extreme_current_events.append(
                        {
                            "feature": "PM2.5_current",
                            "value": float(current_pm25),
                            "q01": float(q01),
                            "q99": float(q99),
                            "side": "above_q99",
                        }
                    )

        feat_row = tmp.loc[pred_ts]
        base_lag1 = self._safe_float(feat_row.get("lag1", np.nan))
        if pd.isna(base_lag1):
            base_lag1 = self._safe_float(current_pm25)
        if pd.isna(base_lag1) and not s.empty:
            base_lag1 = self._safe_float(s.iloc[-1])

        x_row = {}
        imputed = 0
        imputed_names = []
        for f in feats:
            v = self._safe_float(feat_row.get(f, np.nan))
            if pd.isna(v):
                v = self._feature_fallback(f, base_lag1, feat_defaults)
                imputed += 1
                imputed_names.append(str(f))
            x_row[f] = float(v)

        X = pd.DataFrame([x_row], columns=feats)

        return FeatureBuildResult(
            X=X,
            base_lag1=float(base_lag1) if pd.notna(base_lag1) else np.nan,
            history_points=int(len(s)),
            imputed_features=int(imputed),
            imputed_feature_names=imputed_names,
            extreme_current_events=extreme_current_events,
            current_pm25=float(current_pm25) if pd.notna(current_pm25) else np.nan,
            observed_timestamp=ts,
            prediction_timestamp=pred_ts,
            current_aux={k: self._safe_float(v) for k, v in current_aux.items()},
        )
