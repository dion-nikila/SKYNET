from __future__ import annotations

import os
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


HUMAN_MAP = {
    "lag1": "recent PM2.5 patterns from the previous hour are associated with this forecast",
    "lag3": "recent PM2.5 in the last few hours is associated with current forecast behavior",
    "lag6": "earlier PM2.5 conditions from today still align with the model signal",
    "lag12": "half-day PM2.5 history is associated with the next-hour forecast signal",
    "lag24": "yesterday's PM2.5 pattern is associated with current forecast behavior",
    "lag48": "PM2.5 behavior from two days ago still contributes to this pattern",
    "lag72": "PM2.5 behavior from three days ago still contributes to this pattern",
    "lag168": "last week's PM2.5 pattern remains associated with this forecast",
    "roll3": "the short-term PM2.5 average is associated with model direction",
    "roll6": "the recent PM2.5 average is associated with model direction",
    "roll24": "the 24-hour PM2.5 level is associated with this forecast pattern",
    "roll48": "the 48-hour PM2.5 trend is associated with forecast behavior",
    "roll168": "the weekly PM2.5 trend is associated with forecast behavior",
    "std6": "recent PM2.5 volatility is associated with the model signal",
    "std24": "day-level PM2.5 volatility is associated with the model signal",
    "min24": "recent low PM2.5 values help characterize local background conditions",
    "max24": "recent PM2.5 peaks are associated with current risk patterns",
    "ewm6": "very recent PM2.5 trend is weighted strongly by the model",
    "ewm24": "day-scale PM2.5 trend contributes to the model signal",
    "trend_24": "change versus yesterday helps the model track short-term shifts",
    "trend_168": "change versus last week helps the model track broader shifts",
    "roll_diff_3_24": "short- vs day-scale trend differences affect model direction",
    "roll_diff_24_168": "day- vs week-scale trend differences affect model direction",
    "wind_speed": "wind conditions are associated with pollutant dispersion behavior",
    "temperature": "temperature patterns are associated with mixing conditions",
    "humidity": "humidity patterns are associated with particle suspension behavior",
    "PM10": "coarse particles and dust levels are associated with PM2.5 dynamics",
    "NO2": "traffic/combustion gas levels are associated with local pollution pressure",
    "CO": "combustion signal is associated with particulate-risk conditions",
    "SO2": "sulfur-emission signal is associated with local pollution conditions",
    "O3": "photochemical conditions are associated with particulate behavior",
    "is_weekend": "activity-pattern differences (weekend effect) are associated with emissions behavior",
}

FEATURE_LABELS = {
    "lag1": "PM2.5 one hour ago",
    "lag3": "PM2.5 three hours ago",
    "lag6": "PM2.5 six hours ago",
    "lag12": "PM2.5 twelve hours ago",
    "lag24": "PM2.5 one day ago",
    "lag48": "PM2.5 two days ago",
    "lag72": "PM2.5 three days ago",
    "lag168": "PM2.5 one week ago",
    "roll3": "3-hour PM2.5 average",
    "roll6": "6-hour PM2.5 average",
    "roll24": "24-hour PM2.5 average",
    "roll48": "48-hour PM2.5 average",
    "roll168": "7-day PM2.5 average",
    "std6": "PM2.5 variability (6h)",
    "std24": "PM2.5 variability (24h)",
    "min24": "Lowest PM2.5 in last 24h",
    "max24": "Highest PM2.5 in last 24h",
    "ewm6": "Recent PM2.5 trend (6h)",
    "ewm24": "Recent PM2.5 trend (24h)",
    "trend_24": "Change since yesterday",
    "trend_168": "Change since last week",
    "roll_diff_3_24": "Short vs daily PM2.5 trend",
    "roll_diff_24_168": "Daily vs weekly PM2.5 trend",
    "PM10": "PM10 concentration",
    "NO2": "NO2 concentration",
    "SO2": "SO2 concentration",
    "O3": "Ozone concentration",
    "CO": "Carbon monoxide concentration",
    "temperature": "Air temperature",
    "humidity": "Relative humidity",
    "pressure": "Surface pressure",
    "wind_speed": "Wind speed",
    "sin_hour": "Hour-of-day pattern",
    "cos_hour": "Hour-of-day pattern",
    "sin_day": "Day-of-week pattern",
    "cos_day": "Day-of-week pattern",
    "is_weekend": "Weekend effect",
}


class ExplainerService:
    ADDITIVITY_TOLERANCE = 1e-5
    ALIGNMENT_TOLERANCE = 1e-4

    def __init__(self):
        self._cache = {}
        self._shap = None

    def _get_shap_module(self):
        if self._shap is None:
            # Keep matplotlib/shap cache writable in constrained local environments.
            os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".mplconfig"))
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            try:
                import shap  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised only when dependency is missing.
                raise RuntimeError(
                    "SHAP is required for explainability. Install backend dependencies (including shap)."
                ) from exc
            self._shap = shap
        return self._shap

    def _get_explainer(self, model):
        key = id(model)
        if key not in self._cache:
            shap_mod = self._get_shap_module()
            self._cache[key] = shap_mod.TreeExplainer(model)
        return self._cache[key]

    @staticmethod
    def _as_booster(model):
        if isinstance(model, xgb.Booster):
            return model
        if hasattr(model, "get_booster"):
            try:
                return model.get_booster()
            except Exception:
                return None
        return None

    def _prepare_row_for_model(self, model, X_row: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X_row, pd.DataFrame) or X_row.empty:
            raise ValueError("Explanation row must be a non-empty DataFrame.")
        if len(X_row) != 1:
            raise ValueError("Explanation requires exactly one feature row.")

        booster = self._as_booster(model)
        if booster is None or not booster.feature_names:
            return X_row.copy()

        expected = [str(f) for f in booster.feature_names]
        current_cols = [str(c) for c in X_row.columns]
        missing = [f for f in expected if f not in current_cols]
        if missing:
            raise ValueError(f"Explanation row is missing required model features: {missing[:5]}")
        # Keep only model-known features in exact model order.
        return X_row.loc[:, expected].copy()

    def _pred_contrib_row(self, model, X_row: pd.DataFrame):
        booster = self._as_booster(model)
        if booster is None:
            return None

        aligned_row = self._prepare_row_for_model(model, X_row)
        drow = xgb.DMatrix(aligned_row, feature_names=list(aligned_row.columns))
        contrib = booster.predict(drow, pred_contribs=True)
        arr = np.asarray(contrib, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] < 1:
            return None

        row = arr[0]
        if row.shape[0] == len(aligned_row.columns) + 1:
            base_value = float(row[-1])
            contrib_vals = row[:-1]
        elif row.shape[0] == len(aligned_row.columns):
            base_value = 0.0
            contrib_vals = row
        else:
            return None

        contrib_series = pd.Series(contrib_vals, index=aligned_row.columns, dtype=float)
        contrib_sum = float(contrib_series.sum())
        reconstructed_signal = float(base_value + contrib_sum)
        pred_margin = float(booster.predict(drow)[0])
        add_error = float(abs(pred_margin - reconstructed_signal))
        return {
            "series": contrib_series,
            "base_value": float(base_value),
            "contrib_sum": contrib_sum,
            "reconstructed_signal": reconstructed_signal,
            "pred_margin": float(pred_margin),
            "additivity_error": add_error,
        }

    def shap_series(self, model, X_row: pd.DataFrame) -> pd.Series:
        # Prefer XGBoost pred_contribs for exact additive local contributions.
        pred_contrib = self._pred_contrib_row(model, X_row)
        if pred_contrib is not None:
            return pred_contrib["series"]

        aligned_row = self._prepare_row_for_model(model, X_row)
        explainer = self._get_explainer(model)
        vals = explainer.shap_values(aligned_row)
        if hasattr(vals, "ndim") and vals.ndim == 2:
            vals = vals[0]
        return pd.Series(vals, index=aligned_row.columns, dtype=float)

    def explanation_meta(self, model, X_row: pd.DataFrame, contrib_s: pd.Series, prediction_signal: float | None = None):
        prediction_signal_value = None
        if prediction_signal is not None:
            try:
                p = float(prediction_signal)
                if np.isfinite(p):
                    prediction_signal_value = p
            except Exception:
                prediction_signal_value = None

        pred_contrib = self._pred_contrib_row(model, X_row)
        if pred_contrib is not None:
            add_error = float(pred_contrib["additivity_error"])
            add_ok = bool(add_error <= self.ADDITIVITY_TOLERANCE)
            align_error = None
            align_ok = None
            if prediction_signal_value is not None:
                align_error = float(abs(prediction_signal_value - float(pred_contrib["reconstructed_signal"])))
                align_ok = bool(align_error <= self.ALIGNMENT_TOLERANCE)
            return {
                "method": "xgboost_pred_contribs",
                "base_value": float(pred_contrib["base_value"]),
                "contrib_sum": float(pred_contrib["contrib_sum"]),
                "reconstructed_signal": float(pred_contrib["reconstructed_signal"]),
                "prediction_signal": prediction_signal_value,
                "additivity_error": add_error,
                "additivity_ok": add_ok,
                "additivity_tolerance": float(self.ADDITIVITY_TOLERANCE),
                "prediction_alignment_error": align_error,
                "prediction_alignment_ok": align_ok,
            }

        base_value = None
        add_error = None
        add_ok = None
        reconstructed_signal = None
        align_error = None
        align_ok = None
        try:
            aligned_row = self._prepare_row_for_model(model, X_row)
            explainer = self._get_explainer(model)
            expected = explainer.expected_value
            if hasattr(expected, "__len__") and not isinstance(expected, (str, bytes)):
                expected = expected[0]
            base_value = float(expected)
            booster = self._as_booster(model)
            contrib_sum = float(contrib_s.sum())
            reconstructed_signal = float(base_value + contrib_sum)
            if booster is not None:
                drow = xgb.DMatrix(aligned_row, feature_names=list(aligned_row.columns))
                pred_margin = float(booster.predict(drow)[0])
                add_error = float(abs(pred_margin - reconstructed_signal))
                add_ok = bool(add_error <= self.ADDITIVITY_TOLERANCE)
            if prediction_signal_value is not None and reconstructed_signal is not None:
                align_error = float(abs(prediction_signal_value - reconstructed_signal))
                align_ok = bool(align_error <= self.ALIGNMENT_TOLERANCE)
        except Exception:
            pass

        return {
            "method": "tree_shap",
            "base_value": base_value,
            "contrib_sum": float(contrib_s.sum()) if contrib_s is not None else None,
            "reconstructed_signal": reconstructed_signal,
            "prediction_signal": prediction_signal_value,
            "additivity_error": add_error,
            "additivity_ok": add_ok,
            "additivity_tolerance": float(self.ADDITIVITY_TOLERANCE),
            "prediction_alignment_error": align_error,
            "prediction_alignment_ok": align_ok,
        }

    @staticmethod
    def _reason_for_feature(feature: str) -> str:
        return HUMAN_MAP.get(feature, f"{feature} is part of the current model pattern")

    @staticmethod
    def _label_for_feature(feature: str) -> str:
        return FEATURE_LABELS.get(feature, feature)

    @staticmethod
    def top_drivers(shap_s: pd.Series, X_row: pd.DataFrame, top_k: int) -> List[Dict]:
        top = shap_s.abs().sort_values(ascending=False).head(top_k)
        out = []
        up_templates = [
            "{label} is a strong upward signal in this run; {reason}.",
            "Upward pressure is concentrated in {label}; {reason}.",
            "{label} aligns with a higher next-hour tendency; {reason}.",
        ]
        down_templates = [
            "{label} is a strong downward signal in this run; {reason}.",
            "Downward pressure is concentrated in {label}; {reason}.",
            "{label} aligns with a lower next-hour tendency; {reason}.",
        ]

        for idx, f in enumerate(top.index):
            v = float(X_row.iloc[0][f])
            s = float(shap_s[f])
            label = ExplainerService._label_for_feature(f)
            reason = ExplainerService._reason_for_feature(f)
            if s >= 0:
                plain = up_templates[idx % len(up_templates)].format(label=label, reason=reason)
            else:
                plain = down_templates[idx % len(down_templates)].format(label=label, reason=reason)
            out.append(
                {
                    "feature": f,
                    "feature_label": label,
                    "value": v,
                    "shap": s,
                    "direction": "up" if s >= 0 else "down",
                    "plain_text": plain,
                }
            )
        return out

    @staticmethod
    def summary_text(top_items: List[Dict], prefix: str) -> str:
        if not top_items:
            return f"{prefix}: no dominant drivers identified."
        labels = [str(x.get("feature_label") or x.get("feature")) for x in top_items[:3]]
        feats = ", ".join(labels)
        net = float(sum(float(x.get("shap", 0.0)) for x in top_items))
        if net > 0:
            trend = "upward"
        elif net < 0:
            trend = "downward"
        else:
            trend = "mixed"
        return (
            f"{prefix}: strongest influence comes from {feats}, "
            f"with a {trend} overall push in model tendency."
        )

    @staticmethod
    def plain_language_from_top(top_items: List[Dict]) -> List[str]:
        if not top_items:
            return ["No dominant explanation signal was found for this run."]
        return [str(item.get("plain_text", "")) for item in top_items if item.get("plain_text")]

    @staticmethod
    def delta_shap(base_shap: pd.Series, scenario_shap: pd.Series, top_k: int):
        d = (scenario_shap - base_shap).astype(float)
        top = d.abs().sort_values(ascending=False).head(top_k)
        rows = []
        up_templates = [
            "{label} gains influence versus baseline (Δ contribution {delta:+.2f}), strengthening the upward run tendency.",
            "Compared with baseline, {label} shifts upward by {delta:+.2f} and now leans more toward a higher forecast tendency.",
            "{label} shows a stronger upward contribution than baseline (Δ contribution {delta:+.2f}).",
        ]
        down_templates = [
            "{label} loses influence versus baseline (Δ contribution {delta:+.2f}), strengthening the downward run tendency.",
            "Compared with baseline, {label} shifts downward by {delta:+.2f} and now leans more toward a lower forecast tendency.",
            "{label} shows a stronger downward contribution than baseline (Δ contribution {delta:+.2f}).",
        ]

        for idx, f in enumerate(top.index):
            b = float(base_shap[f])
            s = float(scenario_shap[f])
            delta = float(s - b)
            label = ExplainerService._label_for_feature(f)
            if delta >= 0:
                plain = up_templates[idx % len(up_templates)].format(label=label, delta=delta)
            else:
                plain = down_templates[idx % len(down_templates)].format(label=label, delta=delta)
            rows.append(
                {
                    "feature": f,
                    "feature_label": label,
                    "baseline_shap": b,
                    "scenario_shap": s,
                    "delta_shap": delta,
                    "sign_flip": (b == 0 and s != 0) or (b != 0 and s != 0 and (b > 0) != (s > 0)),
                    "plain_text": plain,
                }
            )

        if not rows:
            summary = "No meaningful contribution shift between baseline and scenario."
            plain_lines = ["Scenario produced no material reasoning shift from baseline."]
        else:
            strongest = rows[0]
            summary = (
                f"Largest reasoning shift: {strongest.get('feature_label') or strongest['feature']} "
                f"(Δ contribution {strongest['delta_shap']:+.2f})."
            )
            plain_lines = [r["plain_text"] for r in rows]

        return rows, summary, plain_lines
