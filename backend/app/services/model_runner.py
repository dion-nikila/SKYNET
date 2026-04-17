from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import xgboost as xgb

from ..core.settings import settings

xgb.set_config(verbosity=0)


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
    "PM10_lag1": "PM10 one hour ago",
    "PM10_lag3": "PM10 three hours ago",
    "PM10_roll3": "3-hour PM10 average",
    "PM10_roll24": "24-hour PM10 average",
    "NO2": "NO2 concentration",
    "SO2": "SO2 concentration",
    "O3": "Ozone concentration",
    "CO": "Carbon monoxide concentration",
    "temperature": "Air temperature",
    "humidity": "Relative humidity",
    "humidity_lag1": "Humidity one hour ago",
    "humidity_roll3": "3-hour humidity average",
    "humidity_roll24": "24-hour humidity average",
    "pressure": "Surface pressure",
    "pressure_lag1": "Pressure one hour ago",
    "pressure_roll3": "3-hour pressure average",
    "pressure_roll24": "24-hour pressure average",
    "wind_speed": "Wind speed",
    "wind_speed_lag1": "Wind speed one hour ago",
    "wind_speed_roll3": "3-hour wind speed average",
    "wind_speed_roll24": "24-hour wind speed average",
    "int_windspeedlag1_pm10lag1": "Wind-speed x PM10 interaction (lag1)",
    "int_humiditylag1_temperaturelag1": "Humidity x temperature interaction (lag1)",
    "int_pressurelag1_windspeedlag1": "Pressure x wind-speed interaction (lag1)",
    "sin_hour": "Hour-of-day pattern",
    "cos_hour": "Hour-of-day pattern",
    "sin_day": "Day-of-week pattern",
    "cos_day": "Day-of-week pattern",
    "is_weekend": "Weekend effect",
}


class ModelRunner:
    def __init__(self, model_meta_path: Path | None = None):
        self.model_meta_path = model_meta_path or settings.MODEL_META_PATH
        self._meta = None
        self._uncertainty_profile = None

    def get_meta(self) -> Dict:
        if self._meta is None:
            if not self.model_meta_path.exists():
                raise FileNotFoundError(f"Model metadata not found at {self.model_meta_path}")
            meta = joblib.load(self.model_meta_path)
            meta.setdefault("log_target", False)
            meta.setdefault("target_type", "delta")
            meta.setdefault("bias_correction", 0.0)
            meta.setdefault("feature_defaults", {})
            meta.setdefault("feature_quantiles", {})
            meta.setdefault("training_pm25_mean", None)
            meta.setdefault("global_shap_mean_abs", {})
            meta.setdefault("schema_version", 1)

            model = meta.get("model")
            model_path = meta.get("model_path")

            # Prefer loading from XGBoost native model format if available.
            if model is None and model_path:
                p = Path(model_path)
                if not p.is_absolute():
                    p = self.model_meta_path.parent / p
                booster = xgb.Booster()
                booster.load_model(str(p))
                meta["model"] = self._configure_inference_model(booster)

            # One-time migration path: persist model as JSON and store path in metadata.
            if model is not None and not model_path:
                try:
                    json_path = self.model_meta_path.parent / "xgb_model.json"
                    if hasattr(model, "save_model"):
                        model.save_model(str(json_path))

                        # Save a slim metadata file to avoid repeated pickle-version warnings.
                        disk_meta = dict(meta)
                        disk_meta.pop("model", None)
                        disk_meta["model_path"] = str(json_path.relative_to(self.model_meta_path.parent))
                        joblib.dump(disk_meta, self.model_meta_path)

                        booster = xgb.Booster()
                        booster.load_model(str(json_path))
                        meta["model"] = self._configure_inference_model(booster)
                        meta["model_path"] = disk_meta["model_path"]
                except Exception:
                    # Keep runtime resilient, fall back to the loaded model object.
                    pass

            meta["model"] = self._configure_inference_model(meta.get("model"))

            self._meta = meta
        return self._meta

    @staticmethod
    def _configure_inference_model(model):
        """
        Keep inference deterministic and lightweight.
        A single-thread setting avoids OpenMP/shared-memory issues in constrained
        environments without changing model semantics.
        """
        if model is None:
            return None
        try:
            if hasattr(model, "set_param"):
                model.set_param({"nthread": 1})
        except Exception:
            pass
        return model

    @staticmethod
    def _safe_float(v):
        try:
            out = float(v)
        except Exception:
            return np.nan
        return out if np.isfinite(out) else np.nan

    def model_version(self) -> str:
        meta = self.get_meta()
        return f"xgb_delta_schema_{meta.get('schema_version', 1)}"

    def feature_importance(self, top_n: int = 12):
        meta = self.get_meta()
        model = meta.get("model")
        features = list(meta.get("features", []))
        if model is None:
            return []

        shap_raw = meta.get("global_shap_mean_abs", {})
        shap_scores = {}
        if isinstance(shap_raw, dict):
            for k, v in shap_raw.items():
                val = self._safe_float(v)
                if np.isfinite(val) and val >= 0:
                    shap_scores[str(k)] = float(val)
        elif isinstance(shap_raw, list):
            for row in shap_raw:
                if not isinstance(row, dict):
                    continue
                feat = str(row.get("feature", ""))
                val = self._safe_float(row.get("score", np.nan))
                if feat and np.isfinite(val) and val >= 0:
                    shap_scores[feat] = float(val)

        if shap_scores:
            if features:
                ranked_all = [(f, float(shap_scores.get(f, 0.0))) for f in features]
            else:
                ranked_all = sorted(shap_scores.items(), key=lambda kv: float(kv[1]), reverse=True)
            ranked_all = sorted(ranked_all, key=lambda kv: float(kv[1]), reverse=True)
        else:
            booster = model
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
            gain_scores = booster.get_score(importance_type="gain") if hasattr(booster, "get_score") else {}
            if not gain_scores:
                gain_scores = {f: 0.0 for f in features}

            ranked_all = sorted(gain_scores.items(), key=lambda kv: float(kv[1]), reverse=True)
            if features:
                existing = {k for k, _ in ranked_all}
                ranked_all.extend((f, 0.0) for f in features if f not in existing)
                ranked_all = sorted(ranked_all, key=lambda kv: float(kv[1]), reverse=True)

        ranked = ranked_all[: max(1, int(top_n))]
        total = sum(max(0.0, float(v)) for _, v in ranked_all)
        total = total if total > 0 else 1.0

        out = []
        for feat, raw in ranked:
            score = float(max(0.0, float(raw)))
            pct = float((score / total) * 100.0)
            out.append(
                {
                    "feature": feat,
                    "feature_label": FEATURE_LABELS.get(feat, feat),
                    "score": score,
                    "pct": pct,
                }
            )
        return out

    def uncertainty_profile(self) -> Dict | None:
        if self._uncertainty_profile is not None:
            return self._uncertainty_profile

        meta = self.get_meta()
        profile = meta.get("uncertainty_profile")
        if isinstance(profile, dict) and profile:
            self._uncertainty_profile = profile
            return profile

        plot_data = meta.get("plot_data") or {}
        y_true = np.asarray(plot_data.get("y_true", []), dtype=float)
        preds = np.asarray(plot_data.get("preds", []), dtype=float)
        if y_true.size == 0 or preds.size == 0 or y_true.size != preds.size:
            self._uncertainty_profile = None
            return None

        residual = y_true - preds
        residual = residual[np.isfinite(residual)]
        if residual.size < 50:
            self._uncertainty_profile = None
            return None

        q = {
            "q05": float(np.quantile(residual, 0.05)),
            "q10": float(np.quantile(residual, 0.10)),
            "q20": float(np.quantile(residual, 0.20)),
            "q80": float(np.quantile(residual, 0.80)),
            "q90": float(np.quantile(residual, 0.90)),
            "q95": float(np.quantile(residual, 0.95)),
            "abs_q50": float(np.quantile(np.abs(residual), 0.50)),
            "abs_q80": float(np.quantile(np.abs(residual), 0.80)),
            "abs_q90": float(np.quantile(np.abs(residual), 0.90)),
        }
        profile = {
            "method": "empirical_residual_quantiles_from_haikou_test_split",
            "sample_size": int(residual.size),
            "residual_quantiles": q,
        }
        self._uncertainty_profile = profile
        return profile

    def _interval_from_profile(self, point_pred: float, coverage_pct: int, inflation: float = 1.0) -> Dict | None:
        profile = self.uncertainty_profile()
        if not profile:
            return None
        q = (profile.get("residual_quantiles") or {})
        if int(coverage_pct) >= 90:
            low_q = self._safe_float(q.get("q05", np.nan))
            high_q = self._safe_float(q.get("q95", np.nan))
        else:
            low_q = self._safe_float(q.get("q10", np.nan))
            high_q = self._safe_float(q.get("q90", np.nan))
        if np.isnan(low_q) or np.isnan(high_q):
            return None

        scale = max(0.75, float(inflation))
        lower = float(point_pred + (low_q * scale))
        upper = float(point_pred + (high_q * scale))
        if upper < lower:
            lower, upper = upper, lower
        return {
            "coverage_pct": int(coverage_pct),
            "lower": float(lower),
            "upper": float(upper),
            "width": float(upper - lower),
        }

    def uncertainty_guidance(
        self,
        baseline_pm25: float,
        scenario_pm25: float,
        reliability_score: float,
        scenario_mode: str,
    ) -> Dict:
        profile = self.uncertainty_profile()
        if not profile:
            return {
                "method": "empirical_residual_quantiles_from_haikou_test_split",
                "available": False,
                "note": "Uncertainty bands are unavailable because calibrated residual samples are missing from current model metadata.",
                "caveats": [
                    "Reliability guidance remains available.",
                    "When available, uncertainty bands are empirical error ranges, not probabilistic guarantees.",
                ],
                "calibration_sample_size": None,
                "baseline_bands": [],
                "scenario_bands": [],
                "scenario_inflation": None,
            }

        rel = max(0.0, min(1.0, float(reliability_score)))
        base_inflation = 1.0 + max(0.0, 0.85 - rel) * 0.90
        mode = str(scenario_mode or "").strip().lower()
        mode_extra = {
            "baseline": 0.0,
            "macro": 0.08,
            "guided_intervention": 0.12,
            "manual_custom": 0.18,
        }.get(mode, 0.08)
        scenario_inflation = float(base_inflation * (1.0 + mode_extra))

        baseline_bands = []
        scenario_bands = []
        for cov in (80, 90):
            b = self._interval_from_profile(point_pred=float(baseline_pm25), coverage_pct=cov, inflation=base_inflation)
            s = self._interval_from_profile(point_pred=float(scenario_pm25), coverage_pct=cov, inflation=scenario_inflation)
            if b:
                baseline_bands.append(b)
            if s:
                scenario_bands.append(s)

        return {
            "method": str(profile.get("method", "empirical_residual_quantiles_from_haikou_test_split")),
            "available": bool(baseline_bands and scenario_bands),
            "note": (
                "Bands reflect empirical residual spread from the Haikou historical test split, scaled by run reliability and scenario mode."
            ),
            "caveats": [
                "These are empirical error ranges for decision support, not calibrated probabilities.",
                "Coverage behavior is expected to be strongest for Haikou-like conditions.",
                "Scenario bands include additional widening because interventions can amplify domain mismatch risk.",
            ],
            "calibration_sample_size": int(profile.get("sample_size", 0)),
            "baseline_bands": baseline_bands,
            "scenario_bands": scenario_bands,
            "scenario_inflation": float(scenario_inflation),
        }

    def predict(self, X, base_lag1: float, current_pm25: float):
        meta = self.get_meta()
        model = meta.get("model")
        if model is None:
            raise RuntimeError("Loaded model metadata has no 'model' object.")

        pred_space = float(model.predict(xgb.DMatrix(X, feature_names=list(X.columns)))[0])

        target_type = str(meta.get("target_type", "delta")).lower()
        log_target = bool(meta.get("log_target", False))
        bias_raw = meta.get("bias_correction", 0.0)
        if isinstance(bias_raw, dict):
            bias_correction = self._safe_float(bias_raw.get("value", 0.0))
        else:
            bias_correction = self._safe_float(bias_raw)
        if np.isnan(bias_correction):
            bias_correction = 0.0
        training_pm25_mean = self._safe_float(meta.get("training_pm25_mean", np.nan))

        current_pm25 = self._safe_float(current_pm25)
        base_lag1 = self._safe_float(base_lag1)

        if target_type == "delta":
            base = base_lag1
            if np.isnan(base):
                base = current_pm25 if not np.isnan(current_pm25) else training_pm25_mean
            if np.isnan(base):
                base = 0.0
            pm25_next = float(base + pred_space + bias_correction)
            delta_pm25 = float(pm25_next - base)
        else:
            pm25_next = float(np.expm1(pred_space) if log_target else pred_space)
            reference = current_pm25 if not np.isnan(current_pm25) else 0.0
            delta_pm25 = float(pm25_next - reference)

        return {
            "pred_space": pred_space,
            "delta_pm25_t_plus_1": float(delta_pm25),
            "pm25_t_plus_1": float(pm25_next),
        }
