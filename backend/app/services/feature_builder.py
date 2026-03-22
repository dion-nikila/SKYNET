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

        if name.startswith("std"):
            return 0.0

        if name.startswith("lag") or name.startswith("roll") or name.startswith("ewm"):
            return float(base_lag1) if pd.notna(base_lag1) else 0.0

        if name in ("min24", "max24", "trend_24", "trend_168", "roll_diff_3_24", "roll_diff_24_168"):
            return float(base_lag1) if pd.notna(base_lag1) else 0.0

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

        ts = pd.to_datetime(aq_cur.get("time"))
        pred_ts = ts + pd.Timedelta(hours=1)

        current_pm25 = self._safe_float(aq_cur.get("pm2_5", np.nan))

        pm_history = history_df["PM2.5"].copy() if "PM2.5" in history_df.columns else pd.Series(dtype=float)
        if pd.notna(current_pm25):
            pm_history.loc[ts] = current_pm25

        s = pm_history.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.dropna().astype(float)

        tmp = pd.DataFrame(index=s.index)
        tmp["PM2.5"] = s
        tmp.loc[pred_ts, "PM2.5"] = np.nan
        tmp = tmp.sort_index()
        pm = tmp["PM2.5"]

        for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
            tmp[f"lag{lag}"] = pm.shift(lag)

        pm_past = pm.shift(1)
        tmp["roll3"] = pm_past.rolling(3, min_periods=1).mean()
        tmp["roll6"] = pm_past.rolling(6, min_periods=1).mean()
        tmp["roll24"] = pm_past.rolling(24, min_periods=1).mean()
        tmp["roll48"] = pm_past.rolling(48, min_periods=1).mean()
        tmp["roll168"] = pm_past.rolling(168, min_periods=1).mean()

        tmp["std6"] = pm_past.rolling(6, min_periods=2).std()
        tmp["std24"] = pm_past.rolling(24, min_periods=2).std()
        tmp["min24"] = pm_past.rolling(24, min_periods=1).min()
        tmp["max24"] = pm_past.rolling(24, min_periods=1).max()

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

        for c in AUX_COLS:
            tmp.loc[pred_ts, c] = self._safe_float(current_aux.get(c, np.nan))

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

        feat_row = tmp.loc[pred_ts]
        base_lag1 = self._safe_float(feat_row.get("lag1", np.nan))

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
