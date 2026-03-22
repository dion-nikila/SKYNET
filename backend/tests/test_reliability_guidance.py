from __future__ import annotations

import unittest

import numpy as np

from backend.app.services.health_engine import HealthEngine
from backend.app.services.model_runner import ModelRunner


class ReliabilityGuidanceTests(unittest.TestCase):
    def test_health_engine_returns_decomposed_reliability_guidance(self):
        payload = HealthEngine().build(
            history_stats={"used_hours": 72, "target_hours": 72, "available_hours": 72, "gap_count": 1, "largest_gap_hours": 1},
            imputed_features=4,
            total_features=40,
            ood_events=[{"severity": "soft"}],
            ood_opts={"soft_q": 0.05, "hard_q": 0.01},
            ood_context={"effective_soft_q": 0.05, "effective_hard_q": 0.01, "notes": ["adaptive quantile guard active"]},
            imputed_feature_names=["lag168", "trend_168"],
            extreme_current_events=[{"feature": "PM10", "value": 240.0, "q01": 8.0, "q99": 180.0, "side": "above_q99"}],
            applied_overrides=[{"feature": "NO2", "clamped": True, "direction_limited": True}],
            explainability_meta={"method": "xgboost_pred_contribs", "additivity_ok": True, "prediction_alignment_ok": True},
        )

        reliability = payload.get("reliability") or {}
        self.assertIsInstance(reliability, dict)
        self.assertAlmostEqual(float(payload["quality_score"]), float(reliability.get("score", -1.0)), places=9)
        self.assertEqual(str(payload.get("quality_label")), str(reliability.get("label")))

        components = reliability.get("components") or []
        self.assertEqual(len(components), 6)
        names = {str(c.get("name")) for c in components}
        self.assertSetEqual(
            names,
            {
                "data_completeness",
                "domain_plausibility",
                "imputation_burden",
                "fallback_severity",
                "scenario_validity",
                "explainability_integrity",
            },
        )

        notes = reliability.get("notes") or []
        joined = " ".join(str(n) for n in notes).lower()
        self.assertIn("heuristic", joined)
        self.assertIn("not", joined)
        self.assertIn("probability", joined)
        self.assertIn("168h", str(payload.get("fallback", {}).get("notes", "")))

    def test_uncertainty_guidance_uses_empirical_profile_when_residuals_exist(self):
        runner = ModelRunner()
        y_true = np.linspace(10.0, 60.0, 160)
        preds = y_true + np.sin(np.linspace(0.0, 8.0, 160))
        runner._meta = {"plot_data": {"y_true": y_true.tolist(), "preds": preds.tolist()}}
        runner._uncertainty_profile = None

        out = runner.uncertainty_guidance(
            baseline_pm25=28.5,
            scenario_pm25=31.2,
            reliability_score=0.62,
            scenario_mode="guided_intervention",
        )

        self.assertTrue(bool(out.get("available")))
        self.assertEqual(str(out.get("method")), "empirical_residual_quantiles_from_haikou_test_split")
        self.assertGreaterEqual(int(out.get("calibration_sample_size", 0)), 50)
        self.assertGreaterEqual(len(out.get("baseline_bands", [])), 2)
        self.assertGreaterEqual(len(out.get("scenario_bands", [])), 2)
        self.assertGreater(float(out.get("scenario_inflation", 0.0)), 1.0)

    def test_uncertainty_guidance_reports_unavailable_when_profile_missing(self):
        runner = ModelRunner()
        runner._meta = {"plot_data": {"y_true": [], "preds": []}}
        runner._uncertainty_profile = None

        out = runner.uncertainty_guidance(
            baseline_pm25=20.0,
            scenario_pm25=20.5,
            reliability_score=0.8,
            scenario_mode="macro",
        )

        self.assertFalse(bool(out.get("available")))
        self.assertEqual(out.get("baseline_bands"), [])
        self.assertEqual(out.get("scenario_bands"), [])
        self.assertIn("unavailable", str(out.get("note", "")).lower())


if __name__ == "__main__":
    unittest.main()
