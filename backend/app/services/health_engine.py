from __future__ import annotations

from typing import Dict, List


class HealthEngine:
    @staticmethod
    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @staticmethod
    def _label_from_reliability(score: float) -> str:
        if score >= 0.80:
            return "high reliability guidance"
        if score >= 0.55:
            return "moderate reliability guidance"
        return "low reliability guidance"

    def build(
        self,
        history_stats: Dict,
        imputed_features: int,
        total_features: int,
        ood_events: List[Dict],
        ood_opts: Dict,
        ood_context: Dict | None = None,
        imputed_feature_names: List[str] | None = None,
        extreme_current_events: List[Dict] | None = None,
        applied_overrides: List[Dict] | None = None,
        explainability_meta: Dict | None = None,
    ):
        used = int(history_stats.get("used_hours", 0))
        target = int(history_stats.get("target_hours", 0))
        gaps = int(history_stats.get("gap_count", 0))
        coverage_ratio = (used / target) if target > 0 else 0.0

        imputed_ratio = (imputed_features / total_features) if total_features else 0.0
        imputed_feature_set = {str(x) for x in (imputed_feature_names or [])}
        weekly_imputed = sorted(
            [f for f in ["lag168", "roll168", "trend_168", "roll_diff_24_168"] if f in imputed_feature_set]
        )

        if used < 24:
            level = 2
            label = "short-history fallback"
            notes = "Limited history window; prediction stability may be reduced."
        elif imputed_ratio > 0.15:
            level = 2
            label = "feature-imputation fallback"
            notes = f"{imputed_features} features were imputed from defaults."
        elif gaps > 0:
            level = 1
            label = "history-gap warning"
            notes = "Historical series contains missing hourly gaps."
        else:
            level = 0
            label = "full-data path"
            notes = "No fallback was required."

        if weekly_imputed:
            weekly_note = (
                "Weekly lag-derived features were imputed from trained defaults because available history was below 168h."
            )
            notes = f"{notes} {weekly_note}".strip()

        soft = sum(1 for e in ood_events if e.get("severity") == "soft")
        hard = sum(1 for e in ood_events if e.get("severity") == "hard")
        denom = max(1, len(ood_events))
        score = min(1.0, (soft + 2 * hard) / (2 * denom))
        flag = hard > 0 or (soft / denom) > 0.25

        # Reliability guidance score for non-technical users.
        coverage = (used / target) if target > 0 else 0.0
        gap_penalty = min(0.30, gaps * 0.05)
        impute_penalty = min(0.35, imputed_ratio * 1.2)
        ood_penalty = 0.36 if hard > 0 else (0.18 if soft > 0 else 0.0)
        extreme_events = list(extreme_current_events or [])
        extreme_count = int(len(extreme_events))
        extreme_penalty = min(0.40, (0.06 * extreme_count) + (0.03 * max(0, extreme_count - 2)))

        overrides = list(applied_overrides or [])
        clamped_count = int(sum(1 for row in overrides if bool((row or {}).get("clamped"))))
        direction_limited_count = int(sum(1 for row in overrides if bool((row or {}).get("direction_limited"))))

        method = str((explainability_meta or {}).get("method") or "").strip().lower()
        add_ok = (explainability_meta or {}).get("additivity_ok")
        align_ok = (explainability_meta or {}).get("prediction_alignment_ok")

        completeness_component = self._clip01(coverage - gap_penalty)
        imputation_component = self._clip01(1.0 - impute_penalty)
        domain_component = self._clip01(1.0 - ood_penalty - min(0.35, extreme_penalty))
        fallback_component = {0: 1.0, 1: 0.78, 2: 0.48}.get(int(level), 0.48)
        scenario_component = self._clip01(
            1.0
            - min(
                0.60,
                (0.08 * max(0, clamped_count)) + (0.14 * max(0, direction_limited_count)),
            )
        )
        if method == "xgboost_pred_contribs":
            if add_ok is False or align_ok is False:
                explain_component = 0.72
            else:
                explain_component = 1.0
        elif method == "tree_shap":
            explain_component = 0.85
        elif method == "unavailable":
            explain_component = 0.42
        else:
            explain_component = 0.68

        weights = {
            "data_completeness": 0.30,
            "domain_plausibility": 0.22,
            "imputation_burden": 0.18,
            "fallback_severity": 0.12,
            "scenario_validity": 0.10,
            "explainability_integrity": 0.08,
        }
        reliability_score = float(
            (weights["data_completeness"] * completeness_component)
            + (weights["domain_plausibility"] * domain_component)
            + (weights["imputation_burden"] * imputation_component)
            + (weights["fallback_severity"] * fallback_component)
            + (weights["scenario_validity"] * scenario_component)
            + (weights["explainability_integrity"] * explain_component)
        )
        reliability_score = self._clip01(reliability_score)
        reliability_label = self._label_from_reliability(reliability_score)
        reliability_notes = [
            "Reliability guidance combines data completeness, in-range plausibility, fallback/imputation burden, scenario validity, and explainability integrity.",
            "This guidance is diagnostic and heuristic; it is not a calibrated probability of correctness.",
        ]

        requested_soft_q = float((ood_opts or {}).get("soft_q", 0.05))
        requested_hard_q = float((ood_opts or {}).get("hard_q", 0.01))
        effective_soft_q = float((ood_context or {}).get("effective_soft_q", requested_soft_q))
        effective_hard_q = float((ood_context or {}).get("effective_hard_q", requested_hard_q))
        ood_notes = list((ood_context or {}).get("notes", []))
        extreme_notes: List[str] = []
        if extreme_count > 0:
            extreme_notes.append(
                "Heuristic caution: some current exogenous inputs are outside training q01-q99 bounds; reliability guidance is reduced."
            )
        if extreme_count >= 3:
            extreme_notes.append(
                "Multiple extreme inputs detected simultaneously; treat scenario direction and magnitude as lower-reliability guidance."
            )

        return {
            "history": {
                "target_hours": target,
                "available_hours": int(history_stats.get("available_hours", 0)),
                "used_hours": used,
                "coverage_ratio": float(coverage_ratio),
            },
            "gaps": {
                "gap_count": gaps,
                "largest_gap_hours": int(history_stats.get("largest_gap_hours", 0)),
            },
            "imputation": {
                "imputed_features": int(imputed_features),
                "total_features": int(total_features),
                "ratio": float(imputed_ratio),
                "features": list(imputed_feature_names or []),
            },
            "fallback": {
                "level": level,
                "label": label,
                "notes": notes,
            },
            "ood": {
                "method": "quantile_exceedance",
                "soft_range": {"q_low": float(effective_soft_q), "q_high": 1.0 - float(effective_soft_q)},
                "hard_range": {"q_low": float(effective_hard_q), "q_high": 1.0 - float(effective_hard_q)},
                "requested_soft_range": {"q_low": float(requested_soft_q), "q_high": 1.0 - float(requested_soft_q)},
                "requested_hard_range": {"q_low": float(requested_hard_q), "q_high": 1.0 - float(requested_hard_q)},
                "notes": ood_notes,
                "flag": bool(flag),
                "score": float(score),
                "soft_count": int(soft),
                "hard_count": int(hard),
                "features_exceeded": ood_events,
            },
            "extreme_inputs": {
                "method": "current_vs_training_quantiles_q01_q99",
                "count": extreme_count,
                "flag": bool(extreme_count > 0),
                "notes": extreme_notes,
                "events": extreme_events,
            },
            "quality_score": float(reliability_score),
            "quality_label": reliability_label,
            "reliability": {
                "method": "weighted_heuristic_components",
                "score": float(reliability_score),
                "label": reliability_label,
                "components": [
                    {
                        "name": "data_completeness",
                        "score": float(completeness_component),
                        "weight": float(weights["data_completeness"]),
                        "rationale": "Recent history coverage and gap burden.",
                    },
                    {
                        "name": "domain_plausibility",
                        "score": float(domain_component),
                        "weight": float(weights["domain_plausibility"]),
                        "rationale": "Out-of-range exceedances and extreme current inputs versus training quantiles.",
                    },
                    {
                        "name": "imputation_burden",
                        "score": float(imputation_component),
                        "weight": float(weights["imputation_burden"]),
                        "rationale": "Share of model features imputed from defaults.",
                    },
                    {
                        "name": "fallback_severity",
                        "score": float(fallback_component),
                        "weight": float(weights["fallback_severity"]),
                        "rationale": "Whether fallback paths were used and their severity.",
                    },
                    {
                        "name": "scenario_validity",
                        "score": float(scenario_component),
                        "weight": float(weights["scenario_validity"]),
                        "rationale": "Intervention clamping and direction-limited constraints.",
                    },
                    {
                        "name": "explainability_integrity",
                        "score": float(explain_component),
                        "weight": float(weights["explainability_integrity"]),
                        "rationale": "Availability and integrity status of local explanation diagnostics.",
                    },
                ],
                "notes": reliability_notes,
            },
        }
