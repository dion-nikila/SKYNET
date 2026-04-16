from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd


MACRO_SCENARIOS = {
    "traffic_gridlock": {
        "title": "Traffic Gridlock",
        "description": "Traffic-linked emissions rise while local ventilation weakens.",
        "category": "emissions",
        "default_intensity": 72,
        "knobs": [
            {"feature": "NO2", "direction": "increase", "weight": 1.0, "category": "emission_proxy"},
            {"feature": "CO", "direction": "increase", "weight": 0.85, "category": "emission_proxy"},
            {"feature": "PM10", "direction": "increase", "weight": 0.75, "category": "emission_proxy"},
            {"feature": "wind_speed", "direction": "decrease", "weight": 0.25, "category": "wind"},
            {"feature": "pressure", "direction": "increase", "weight": 0.20, "category": "humidity"},
        ],
    },
    "strong_dispersion": {
        "title": "Strong Dispersion",
        "description": "Wind-driven ventilation that clears near-surface pollution; humidity/pressure shifts are secondary contextual adjustments.",
        "category": "weather",
        "default_intensity": 68,
        "knobs": [
            {"feature": "wind_speed", "direction": "increase", "weight": 0.95, "category": "wind"},
            {"feature": "humidity", "direction": "decrease", "weight": 0.20, "category": "humidity"},
            {"feature": "PM10", "direction": "decrease", "weight": 0.95, "category": "emission_proxy"},
            {"feature": "NO2", "direction": "decrease", "weight": 0.70, "category": "emission_proxy"},
            {"feature": "CO", "direction": "decrease", "weight": 0.60, "category": "emission_proxy"},
            {"feature": "pressure", "direction": "decrease", "weight": 0.15, "category": "humidity"},
        ],
    },
    "heatwave": {
        "title": "Heatwave",
        "description": "Hot and dry build-up with primary ozone stress; PM10 rise is treated as context-dependent.",
        "category": "weather",
        "default_intensity": 66,
        "knobs": [
            {"feature": "temperature", "direction": "increase", "weight": 0.95, "category": "temperature"},
            {"feature": "humidity", "direction": "decrease", "weight": 0.55, "category": "humidity"},
            {"feature": "PM10", "direction": "increase", "weight": 0.40, "category": "emission_proxy"},
            {"feature": "O3", "direction": "increase", "weight": 0.85, "category": "emission_proxy"},
            {"feature": "wind_speed", "direction": "decrease", "weight": 0.25, "category": "wind"},
        ],
    },
    "dust_resuspension": {
        "title": "Dust Resuspension",
        "description": "Wind lifts dust and road particles, increasing coarse particulate pollution under drier conditions.",
        "category": "dispersion",
        "default_intensity": 70,
        "knobs": [
            {"feature": "PM10", "direction": "increase", "weight": 0.95, "category": "emission_proxy"},
            {"feature": "wind_speed", "direction": "increase", "weight": 0.55, "category": "wind"},
            {"feature": "humidity", "direction": "decrease", "weight": 0.25, "category": "humidity"},
        ],
    },
    "trapped_pollution": {
        "title": "Trapped Pollution",
        "description": "Low wind and poor dispersion conditions that can trap pollutants near ground level.",
        "category": "dispersion",
        "default_intensity": 78,
        "knobs": [
            {"feature": "wind_speed", "direction": "decrease", "weight": 1.0, "category": "wind"},
            {"feature": "pressure", "direction": "increase", "weight": 0.95, "category": "humidity"},
            {"feature": "humidity", "direction": "increase", "weight": 0.90, "category": "humidity"},
            {"feature": "PM10", "direction": "increase", "weight": 0.65, "category": "emission_proxy"},
            {"feature": "NO2", "direction": "increase", "weight": 0.35, "category": "emission_proxy"},
        ],
    },
    "industrial_plume": {
        "title": "Industrial Source Loading",
        "description": "Industrial-combustion loading under weaker dispersion with coordinated gas and particle increases.",
        "category": "emissions",
        "default_intensity": 70,
        "knobs": [
            {"feature": "SO2", "direction": "increase", "weight": 0.95, "category": "emission_proxy"},
            {"feature": "NO2", "direction": "increase", "weight": 0.75, "category": "emission_proxy"},
            {"feature": "CO", "direction": "increase", "weight": 0.75, "category": "emission_proxy"},
            {"feature": "PM10", "direction": "increase", "weight": 0.65, "category": "emission_proxy"},
            {"feature": "pressure", "direction": "increase", "weight": 0.25, "category": "humidity"},
            {"feature": "wind_speed", "direction": "decrease", "weight": 0.30, "category": "wind"},
        ],
    },
}

SCENARIO_ALIASES = {
    "heavy_rainstorm": "strong_dispersion",
    "stagnation": "trapped_pollution",
    "windy_dispersion": "dust_resuspension",
}


CUSTOM_OVERRIDE_FEATURES = [
    "PM10",
    "NO2",
    "CO",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "O3",
    "SO2",
]

CUSTOM_IMPACT_MODES = {"conservative", "stronger_realistic"}


CATEGORY_FEATURES = {
    "wind": ["wind_speed"],
    "humidity": ["humidity"],
    "temperature": ["temperature"],
    "emission_proxy": ["NO2", "CO", "PM10", "SO2"],
}


MAG_WEIGHTS = {"small": 0.25, "medium": 0.5, "large": 0.8}
STRICT_QUANTILE_FEATURES = set(CUSTOM_OVERRIDE_FEATURES) | {
    str(knob["feature"])
    for _sid, cfg in MACRO_SCENARIOS.items()
    for knob in cfg.get("knobs", [])
}

DERIVED_EXOGENOUS_FEATURES = {
    "PM10": {"lag1": "PM10_lag1", "roll3": "PM10_roll3", "roll24": "PM10_roll24", "lag3": "PM10_lag3"},
    "humidity": {"lag1": "humidity_lag1", "roll3": "humidity_roll3", "roll24": "humidity_roll24"},
    "wind_speed": {"lag1": "wind_speed_lag1", "roll3": "wind_speed_roll3", "roll24": "wind_speed_roll24"},
    "pressure": {"lag1": "pressure_lag1", "roll3": "pressure_roll3", "roll24": "pressure_roll24"},
    "temperature": {"lag1": "temperature_lag1"},
}

INTERACTION_FEATURE_RECIPES = {
    "int_windspeedlag1_pm10lag1": ("wind_speed_lag1", "PM10_lag1"),
    "int_humiditylag1_temperaturelag1": ("humidity_lag1", "temperature_lag1"),
    "int_pressurelag1_windspeedlag1": ("pressure_lag1", "wind_speed_lag1"),
}


class ScenarioEngine:
    @staticmethod
    def _safe_float(v):
        try:
            out = float(v)
        except Exception:
            return np.nan
        return out if np.isfinite(out) else np.nan

    def list_cards(self):
        cards = []
        for sid, cfg in MACRO_SCENARIOS.items():
            cards.append(
                {
                    "scenario_id": sid,
                    "title": cfg["title"],
                    "description": cfg["description"],
                    "category": cfg["category"],
                    "default_intensity": cfg["default_intensity"],
                    "knobs": [k["feature"] for k in cfg["knobs"]],
                }
            )
        return cards

    @staticmethod
    def _movement_direction(from_value: float, to_value: float, tol: float = 1e-9) -> str:
        delta = float(to_value) - float(from_value)
        if delta > float(tol):
            return "increase"
        if delta < -float(tol):
            return "decrease"
        return "unchanged"

    @staticmethod
    def _quantile_label(q: float) -> str:
        q_int = int(round(float(q) * 100))
        return f"q{q_int:02d}"

    @staticmethod
    def _normalize_ood_opts(ood_opts: Dict | None):
        raw_soft = (ood_opts or {}).get("soft_q", 0.05)
        raw_hard = (ood_opts or {}).get("hard_q", 0.01)
        notes: List[str] = []

        try:
            soft_q = float(raw_soft)
        except Exception:
            soft_q = 0.05
            notes.append("Invalid soft_q; defaulted to 0.05.")
        try:
            hard_q = float(raw_hard)
        except Exception:
            hard_q = 0.01
            notes.append("Invalid hard_q; defaulted to 0.01.")

        soft_q = max(0.0, min(0.49, soft_q))
        hard_q = max(0.0, min(0.49, hard_q))
        if hard_q > soft_q:
            hard_q = soft_q
            notes.append("hard_q was greater than soft_q; hard_q was clamped to soft_q.")

        return {"requested_soft_q": float(soft_q), "requested_hard_q": float(hard_q), "notes": notes}

    @staticmethod
    def _extract_quantiles_for_feature(meta: Dict, feature: str):
        quantiles = (meta.get("feature_quantiles", {}) or {}).get(feature, {})
        low_map: Dict[float, float] = {}
        high_map: Dict[float, float] = {}

        for k, v in quantiles.items():
            if not isinstance(k, str) or not k.startswith("q"):
                continue
            try:
                lvl = float(k[1:]) / 100.0
                val = float(v)
            except Exception:
                continue
            if not np.isfinite(val) or lvl <= 0.0 or lvl >= 1.0:
                continue
            if lvl < 0.5:
                low_map[round(float(lvl), 4)] = val
            elif lvl > 0.5:
                high_map[round(float(1.0 - lvl), 4)] = val

        return low_map, high_map

    @staticmethod
    def _nearest_level(levels: List[float], target: float) -> float:
        if not levels:
            return float(target)
        return float(min(levels, key=lambda lvl: (abs(lvl - target), lvl)))

    @classmethod
    def _resolve_bounds_with_opts(
        cls,
        meta: Dict,
        feature: str,
        current_value: float,
        ood_cfg: Dict,
        strict_if_missing: bool = False,
    ):
        low_map, high_map = cls._extract_quantiles_for_feature(meta, feature)
        common_levels = sorted(set(low_map).intersection(set(high_map)))
        notes: List[str] = []

        requested_soft_q = float(ood_cfg["requested_soft_q"])
        requested_hard_q = float(ood_cfg["requested_hard_q"])

        if common_levels:
            soft_q = cls._nearest_level(common_levels, requested_soft_q)
            hard_candidates = [lvl for lvl in common_levels if lvl <= soft_q]
            if not hard_candidates:
                hard_candidates = common_levels
            hard_q = cls._nearest_level(hard_candidates, requested_hard_q)

            bounds = {
                "q_soft": float(soft_q),
                "q_hard": float(hard_q),
                "soft_low": float(low_map[soft_q]),
                "soft_high": float(high_map[soft_q]),
                "hard_low": float(low_map[hard_q]),
                "hard_high": float(high_map[hard_q]),
                "quantiles_missing_locked": False,
            }
            if not np.isclose(soft_q, requested_soft_q):
                notes.append(f"{feature}: requested soft_q={requested_soft_q:.2f} mapped to supported q={soft_q:.2f}.")
            if not np.isclose(hard_q, requested_hard_q):
                notes.append(f"{feature}: requested hard_q={requested_hard_q:.2f} mapped to supported q={hard_q:.2f}.")
            return bounds, notes

        defaults = meta.get("feature_defaults", {})
        center = current_value
        if not np.isfinite(center):
            center = defaults.get(feature, 0.0)
        center = float(center)
        if strict_if_missing:
            notes.append(
                f"{feature}: feature quantiles unavailable; intervention locked to baseline to avoid synthetic realism."
            )
            return (
                {
                    "q_soft": float(requested_soft_q),
                    "q_hard": float(requested_hard_q),
                    "soft_low": center,
                    "soft_high": center,
                    "hard_low": center,
                    "hard_high": center,
                    "quantiles_missing_locked": True,
                },
                notes,
            )

        spread = max(abs(center) * 0.5, 1.0)
        notes.append(f"{feature}: feature quantiles unavailable; used fallback synthetic bounds.")
        return (
            {
                "q_soft": 0.05,
                "q_hard": 0.01,
                "soft_low": center - 1.0 * spread,
                "soft_high": center + 1.0 * spread,
                "hard_low": center - 2.0 * spread,
                "hard_high": center + 2.0 * spread,
                "quantiles_missing_locked": True,
            },
            notes,
        )

    @classmethod
    def _bounds_for(cls, meta: Dict, feature: str, current_value: float, ood_cfg: Dict):
        return cls._resolve_bounds_with_opts(
            meta,
            feature,
            current_value,
            ood_cfg,
            strict_if_missing=bool(str(feature) in STRICT_QUANTILE_FEATURES),
        )

    @staticmethod
    def _normalize_impact_mode(impact_mode: str | None) -> str:
        mode = str(impact_mode or "conservative").strip().lower()
        return mode if mode in CUSTOM_IMPACT_MODES else "conservative"

    @staticmethod
    def _feature_importance_scale(meta: Dict, feature: str) -> float:
        shap_map = meta.get("global_shap_mean_abs", {}) or {}
        if not isinstance(shap_map, dict):
            return 0.5

        cleaned = {}
        for k, v in shap_map.items():
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv) and fv >= 0:
                cleaned[str(k)] = float(fv)
        if not cleaned:
            return 0.5

        max_val = max(cleaned.values())
        if max_val <= 0:
            return 0.5
        return float(np.clip(cleaned.get(feature, 0.0) / max_val, 0.0, 1.0))

    def _build_impact_preview(self, rows: List[Dict], impact_mode: str) -> Dict:
        if not rows:
            return {
                "level": "low",
                "score": 0.0,
                "note": "Impact preview is low because no custom overrides are currently selected.",
                "basis": "heuristic_estimate",
                "factors": [],
            }

        total = float(sum(max(0.0, float(r.get("contribution", 0.0))) for r in rows))
        score = total * (1.1 if impact_mode == "stronger_realistic" else 1.0)

        if score < 0.7:
            level = "low"
            note = (
                "Estimated preview only: impact appears low based on selected controls and typical model sensitivity. "
                "Final forecast may differ after full model evaluation."
            )
        elif score < 1.6:
            level = "medium"
            note = (
                "Estimated preview only: impact appears medium; selected controls should produce a visible but bounded change. "
                "Final forecast may differ after full model evaluation."
            )
        else:
            level = "high"
            note = (
                "Estimated preview only: impact appears high within realistic bounds; selected controls target stronger model-leverage factors. "
                "Final forecast may differ after full model evaluation."
            )

        top = sorted(rows, key=lambda r: float(r.get("contribution", 0.0)), reverse=True)[:3]
        factors = [str(r.get("feature", "")) for r in top if str(r.get("feature", ""))]

        return {
            "level": level,
            "score": float(round(score, 3)),
            "note": note,
            "basis": "heuristic_estimate",
            "factors": factors,
        }

    @staticmethod
    def _resolve_macro_id(scenario_id: str):
        sid = str(scenario_id or "").strip()
        if sid in MACRO_SCENARIOS:
            return sid, False
        if sid in SCENARIO_ALIASES:
            return SCENARIO_ALIASES[sid], True
        known = sorted(list(MACRO_SCENARIOS.keys()) + list(SCENARIO_ALIASES.keys()))
        raise ValueError(f"Unknown macro scenario_id: {sid}. Allowed IDs: {', '.join(known)}")

    def _apply_knob(self, current: float, direction: str, weight: float, intensity: int, bounds: Dict):
        base_intensity = max(0.0, min(1.0, intensity / 100.0))
        # Slightly non-linear scaling makes medium/high intensities move more decisively.
        scaled_intensity = base_intensity ** 0.85
        alpha = scaled_intensity * max(0.0, min(1.0, weight))
        if direction == "increase":
            feasible_span = max(0.0, float(bounds["soft_high"] - current))
            direction_limited = feasible_span <= 1e-9
            raw_target = float(current + alpha * feasible_span)
            soft_exceeded = bool(direction_limited and current > (bounds["soft_high"] + 1e-9))
            hard_exceeded = bool(direction_limited and current > (bounds["hard_high"] + 1e-9))
        else:
            feasible_span = max(0.0, float(current - bounds["soft_low"]))
            direction_limited = feasible_span <= 1e-9
            raw_target = float(current - alpha * feasible_span)
            soft_exceeded = bool(direction_limited and current < (bounds["soft_low"] - 1e-9))
            hard_exceeded = bool(direction_limited and current < (bounds["hard_low"] - 1e-9))

        # Preserve intervention intent semantics: if requested movement cannot be
        # honored inside plausibility bounds, keep baseline value unchanged.
        if direction_limited:
            clamped = float(current)
        else:
            clamped = float(np.clip(raw_target, bounds["soft_low"], bounds["soft_high"]))
        return clamped, raw_target, soft_exceeded, hard_exceeded, bool(direction_limited)

    def _sync_derived_features(
        self,
        scenario_X: pd.DataFrame,
        baseline_X: pd.DataFrame,
        changed_features: Set[str],
    ) -> pd.DataFrame:
        """
        Keep derived lag/rolling/interaction features consistent after manual exogenous edits.
        This is row-local and uses only baseline row values plus user edits (no future data).
        """
        if scenario_X.empty or baseline_X.empty or not changed_features:
            return scenario_X

        idx = scenario_X.index[0]
        base_row = baseline_X.iloc[0]

        for feature in sorted(set(changed_features)):
            spec = DERIVED_EXOGENOUS_FEATURES.get(str(feature))
            if not spec:
                continue

            old_now = self._safe_float(base_row.get(feature, np.nan))
            new_now = self._safe_float(scenario_X.iloc[0].get(feature, np.nan))
            if not np.isfinite(new_now):
                continue

            lag1_col = spec.get("lag1")
            if lag1_col and lag1_col in scenario_X.columns:
                scenario_X.loc[idx, lag1_col] = float(new_now)

            for roll_col, window in ((spec.get("roll3"), 3.0), (spec.get("roll24"), 24.0)):
                if not roll_col or roll_col not in scenario_X.columns:
                    continue
                old_roll = self._safe_float(base_row.get(roll_col, np.nan))
                if np.isfinite(old_roll) and np.isfinite(old_now):
                    scenario_X.loc[idx, roll_col] = float(old_roll + ((new_now - old_now) / window))
                else:
                    scenario_X.loc[idx, roll_col] = float(new_now)

        # Recompute interaction features only if model uses them.
        for interaction_feature, (left_col, right_col) in INTERACTION_FEATURE_RECIPES.items():
            if interaction_feature not in scenario_X.columns:
                continue

            left_val = self._safe_float(scenario_X.iloc[0].get(left_col, np.nan))
            right_val = self._safe_float(scenario_X.iloc[0].get(right_col, np.nan))

            if not np.isfinite(left_val):
                left_base = left_col.replace("_lag1", "")
                left_val = self._safe_float(scenario_X.iloc[0].get(left_base, np.nan))
            if not np.isfinite(right_val):
                right_base = right_col.replace("_lag1", "")
                right_val = self._safe_float(scenario_X.iloc[0].get(right_base, np.nan))

            if np.isfinite(left_val) and np.isfinite(right_val):
                scenario_X.loc[idx, interaction_feature] = float(left_val * right_val)

        return scenario_X

    def apply(self, scenario, baseline_X: pd.DataFrame, meta: Dict, ood_opts: Dict | None = None, return_context: bool = False):
        ood_cfg = self._normalize_ood_opts(ood_opts)
        ood_notes: List[str] = list(ood_cfg.get("notes", []))

        scenario_type = scenario.type
        intensity = int(scenario.intensity)
        scenario_X = baseline_X.copy()
        applied = []
        ood_events = []
        changed_features: Set[str] = set()

        knobs = []
        scenario_id = scenario.scenario_id or "custom"

        alias_used = False
        if scenario_type == "macro":
            scenario_id, alias_used = self._resolve_macro_id(scenario_id)
            knobs = MACRO_SCENARIOS[scenario_id]["knobs"]
            if alias_used:
                ood_notes.append(f"Legacy scenario alias mapped to canonical ID '{scenario_id}'.")
        else:
            scenario_id = "baseline"
            for item in scenario.items or []:
                features = CATEGORY_FEATURES.get(item.category, [])
                for f in features:
                    knobs.append(
                        {
                            "feature": f,
                            "direction": item.direction,
                            "weight": MAG_WEIGHTS[item.magnitude],
                            "category": item.category,
                        }
                    )
            if knobs:
                scenario_id = "guided_intervention"

        effective_soft_levels: List[float] = []
        effective_hard_levels: List[float] = []

        for knob in knobs:
            feature = knob["feature"]
            if feature not in scenario_X.columns:
                continue

            current = float(scenario_X.iloc[0][feature])
            bounds, bound_notes = self._bounds_for(meta, feature, current, ood_cfg)
            ood_notes.extend(bound_notes)
            effective_soft_levels.append(float(bounds["q_soft"]))
            effective_hard_levels.append(float(bounds["q_hard"]))
            target, raw_target, soft_exceeded, hard_exceeded, direction_limited = self._apply_knob(
                current=current,
                direction=knob["direction"],
                weight=float(knob["weight"]),
                intensity=intensity,
                bounds=bounds,
            )

            scenario_X.loc[scenario_X.index[0], feature] = float(target)
            if not np.isclose(current, target):
                changed_features.add(str(feature))
            clamped = not np.isclose(raw_target, target)
            effective_direction = self._movement_direction(current, target)
            requested_direction = str(knob["direction"])

            soft_low_label = self._quantile_label(bounds["q_soft"])
            soft_high_label = self._quantile_label(1.0 - bounds["q_soft"])
            hard_low_label = self._quantile_label(bounds["q_hard"])
            hard_high_label = self._quantile_label(1.0 - bounds["q_hard"])

            reason = "within quantile range"
            if direction_limited:
                if bool(bounds.get("quantiles_missing_locked", False)):
                    reason = "requested change was not applied because training quantiles are unavailable for this feature"
                    ood_notes.append(
                        f"{feature}: intervention skipped because training quantiles are unavailable for this controllable feature."
                    )
                else:
                    reason = (
                        f"requested {requested_direction} could not be applied because baseline was already "
                        f"outside/at the {soft_low_label}/{soft_high_label} plausibility edge in that direction"
                    )
                    ood_notes.append(
                        f"{feature}: requested {requested_direction} was limited by training {soft_low_label}/{soft_high_label} bounds."
                    )
            elif clamped:
                reason = f"clamped to training {soft_low_label}/{soft_high_label} range"

            applied.append(
                {
                    "category": knob.get("category", "custom"),
                    "feature": feature,
                    "from": float(current),
                    "to": float(target),
                    "clamped": bool(clamped),
                    "reason": reason,
                    "requested_direction": requested_direction,
                    "effective_direction": effective_direction,
                    "direction_limited": bool(direction_limited),
                }
            )

            if hard_exceeded or soft_exceeded or direction_limited:
                severity = "hard" if hard_exceeded else "soft"
                if requested_direction == "increase":
                    bound = hard_high_label if severity == "hard" else soft_high_label
                else:
                    bound = hard_low_label if severity == "hard" else soft_low_label
                ood_events.append(
                    {
                        "feature": feature,
                        "value": float(current if direction_limited else raw_target),
                        "bound": bound,
                        "q01": float(bounds["hard_low"]),
                        "q05": float(bounds["soft_low"]),
                        "q95": float(bounds["soft_high"]),
                        "q99": float(bounds["hard_high"]),
                        "severity": severity,
                    }
                )

        scenario_X = self._sync_derived_features(
            scenario_X=scenario_X,
            baseline_X=baseline_X,
            changed_features=changed_features,
        )

        effective_soft_q = float(min(effective_soft_levels)) if effective_soft_levels else 0.05
        effective_hard_q = float(min(effective_hard_levels)) if effective_hard_levels else 0.01
        if effective_hard_q > effective_soft_q:
            effective_hard_q = effective_soft_q

        ood_context = {
            "requested_soft_q": float(ood_cfg["requested_soft_q"]),
            "requested_hard_q": float(ood_cfg["requested_hard_q"]),
            "effective_soft_q": float(effective_soft_q),
            "effective_hard_q": float(effective_hard_q),
            "notes": sorted(set(ood_notes)),
        }

        if return_context:
            return scenario_X, applied, ood_events, scenario_id, ood_context
        return scenario_X, applied, ood_events, scenario_id

    def apply_value_overrides(
        self,
        overrides: Dict[str, float],
        baseline_X: pd.DataFrame,
        meta: Dict,
        impact_mode: str | None = "conservative",
        ood_opts: Dict | None = None,
        return_context: bool = False,
        include_preview: bool = False,
    ):
        resolved_mode = self._normalize_impact_mode(impact_mode)
        ood_cfg = self._normalize_ood_opts(ood_opts)
        ood_notes: List[str] = list(ood_cfg.get("notes", []))

        scenario_X = baseline_X.copy()
        applied = []
        ood_events = []
        effective_soft_levels: List[float] = []
        effective_hard_levels: List[float] = []
        preview_rows: List[Dict] = []
        changed_features: Set[str] = set()

        for feature, raw_target_in in (overrides or {}).items():
            if feature not in CUSTOM_OVERRIDE_FEATURES:
                continue
            if feature not in scenario_X.columns:
                continue

            try:
                raw_target = float(raw_target_in)
            except Exception:
                continue
            if not np.isfinite(raw_target):
                continue

            current = float(scenario_X.iloc[0][feature])
            requested_direction = self._movement_direction(current, raw_target)
            bounds, bound_notes = self._bounds_for(meta, feature, current, ood_cfg)
            ood_notes.extend(bound_notes)

            effective_soft_levels.append(float(bounds["q_soft"]))
            effective_hard_levels.append(float(bounds["q_hard"]))

            conservative_target = float(np.clip(raw_target, bounds["soft_low"], bounds["soft_high"]))
            stronger_target = float(np.clip(raw_target, bounds["hard_low"], bounds["hard_high"]))
            soft_exceeded = raw_target < bounds["soft_low"] or raw_target > bounds["soft_high"]
            hard_exceeded = raw_target < bounds["hard_low"] or raw_target > bounds["hard_high"]

            final_target = conservative_target
            boosted = False
            if resolved_mode == "stronger_realistic" and not np.isclose(stronger_target, current):
                direction = 1.0 if stronger_target > current else -1.0
                edge = float(bounds["hard_high"] if direction > 0 else bounds["hard_low"])
                edge_span = abs(edge - current)
                if edge_span > 1e-9:
                    request_ratio = abs(stronger_target - current) / edge_span
                    leverage = self._feature_importance_scale(meta, feature)
                    boost_factor = 1.1 if leverage < 0.25 else (1.3 + (0.7 * leverage))
                    boosted_ratio = float(np.clip(request_ratio * boost_factor, 0.0, 1.0))
                    final_target = float(current + (direction * boosted_ratio * edge_span))
                    final_target = float(np.clip(final_target, bounds["hard_low"], bounds["hard_high"]))
                    boosted = not np.isclose(final_target, stronger_target)
            elif resolved_mode == "stronger_realistic":
                final_target = stronger_target

            clamped = float(final_target)
            was_clamped = not np.isclose(raw_target, clamped)

            effective_direction = self._movement_direction(current, clamped)
            direction_limited = False
            if requested_direction in {"increase", "decrease"} and effective_direction not in {
                requested_direction,
                "unchanged",
            }:
                # Never let custom intent silently invert due bounds.
                clamped = float(current)
                effective_direction = "unchanged"
                direction_limited = True
                was_clamped = True

            scenario_X.loc[scenario_X.index[0], feature] = float(clamped)
            if not np.isclose(current, clamped):
                changed_features.add(str(feature))

            soft_low_label = self._quantile_label(bounds["q_soft"])
            soft_high_label = self._quantile_label(1.0 - bounds["q_soft"])
            hard_low_label = self._quantile_label(bounds["q_hard"])
            hard_high_label = self._quantile_label(1.0 - bounds["q_hard"])
            reason = "manual override within quantile range"
            if direction_limited:
                if bool(bounds.get("quantiles_missing_locked", False)):
                    reason = "manual override was not applied because training quantiles are unavailable for this feature"
                    ood_notes.append(
                        f"{feature}: manual override skipped because training quantiles are unavailable for this controllable feature."
                    )
                else:
                    reason = (
                        f"manual override requested {requested_direction}, but baseline was already outside/at "
                        f"the feasible {soft_low_label}/{soft_high_label} edge in that direction"
                    )
                    ood_notes.append(
                        f"{feature}: manual override requested {requested_direction} but was direction-limited by training bounds."
                    )
            elif was_clamped:
                reason = f"manual override clamped to training {soft_low_label}/{soft_high_label} range"
                if resolved_mode == "stronger_realistic":
                    reason = f"manual override clamped to stronger realistic {hard_low_label}/{hard_high_label} range"
            if resolved_mode == "stronger_realistic" and boosted:
                reason = (
                    f"stronger realistic mode moved override closer to {hard_low_label}/{hard_high_label} training range edge"
                )

            applied.append(
                {
                    "category": "manual_override",
                    "feature": feature,
                    "from": float(current),
                    "to": float(clamped),
                    "clamped": bool(was_clamped),
                    "reason": reason,
                    "requested_direction": requested_direction,
                    "effective_direction": effective_direction,
                    "direction_limited": bool(direction_limited),
                }
            )

            soft_span = max(abs(bounds["soft_high"] - bounds["soft_low"]), 1e-9)
            movement = min(1.0, abs(clamped - current) / soft_span)
            leverage = self._feature_importance_scale(meta, feature)
            contribution = float(max(0.0, movement) * (0.4 + (0.8 * leverage)))
            preview_rows.append({"feature": feature, "contribution": contribution})

            if hard_exceeded or soft_exceeded or direction_limited:
                severity = "hard" if hard_exceeded else "soft"
                if requested_direction == "increase":
                    bound = hard_high_label if severity == "hard" else soft_high_label
                else:
                    bound = hard_low_label if severity == "hard" else soft_low_label

                ood_events.append(
                    {
                        "feature": feature,
                        "value": float(current if direction_limited else raw_target),
                        "bound": bound,
                        "q01": float(bounds["hard_low"]),
                        "q05": float(bounds["soft_low"]),
                        "q95": float(bounds["soft_high"]),
                        "q99": float(bounds["hard_high"]),
                        "severity": severity,
                    }
                )

        scenario_X = self._sync_derived_features(
            scenario_X=scenario_X,
            baseline_X=baseline_X,
            changed_features=changed_features,
        )

        effective_soft_q = float(min(effective_soft_levels)) if effective_soft_levels else float(ood_cfg["requested_soft_q"])
        effective_hard_q = float(min(effective_hard_levels)) if effective_hard_levels else float(ood_cfg["requested_hard_q"])
        if effective_hard_q > effective_soft_q:
            effective_hard_q = effective_soft_q

        ood_context = {
            "requested_soft_q": float(ood_cfg["requested_soft_q"]),
            "requested_hard_q": float(ood_cfg["requested_hard_q"]),
            "effective_soft_q": float(effective_soft_q),
            "effective_hard_q": float(effective_hard_q),
            "notes": sorted(set(ood_notes)),
        }

        impact_preview = self._build_impact_preview(preview_rows, resolved_mode)
        if resolved_mode == "stronger_realistic":
            impact_preview["note"] += " Stronger realistic mode remains baseline-anchored and quantile-bounded."

        if return_context:
            if include_preview:
                return scenario_X, applied, ood_events, ood_context, impact_preview, resolved_mode
            return scenario_X, applied, ood_events, ood_context
        return scenario_X, applied, ood_events
