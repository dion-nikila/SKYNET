from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.scripts.validate_scenarios import summarize_results


class ValidateScenariosHarnessTests(unittest.TestCase):
    def test_summarize_results_outputs_required_metrics(self):
        rows = []
        for sample_id in [0, 1]:
            for intensity, a, b in [
                (0, 0.0, 0.0),
                (10, 0.1, -0.2),
                (25, 0.2, -0.4),
                (50, 0.4, -0.6),
                (75, 0.6, -0.8),
                (100, 0.8, -1.0),
            ]:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": "scenario_a",
                        "intensity": intensity,
                        "delta_pm25": a,
                        "abs_delta_pm25": abs(a),
                        "ood_event_count": 0,
                    }
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": "scenario_b",
                        "intensity": intensity,
                        "delta_pm25": b,
                        "abs_delta_pm25": abs(b),
                        "ood_event_count": 1 if intensity == 100 else 0,
                    }
                )

        summary = summarize_results(pd.DataFrame(rows), intensities=[0, 10, 25, 50, 75, 100], saturation_eps=0.05)
        self.assertEqual(set(summary["scenario_id"]), {"scenario_a", "scenario_b"})

        required_cols = {
            "effect_size_mean_abs_at_100",
            "effect_size_median_abs_at_100",
            "effect_size_p90_abs_at_100",
            "monotonicity_rate_abs",
            "saturation_rate_75_to_100",
            "ood_rate_any_at_100",
            "ood_rate_any_all_intensities",
            "mean_ood_events_at_100",
            "near_zero_rate_abs_lt_0_1_at_100",
        }
        self.assertTrue(required_cols.issubset(set(summary.columns)))
        self.assertTrue(np.isfinite(summary["effect_size_mean_abs_at_100"]).all())
        self.assertTrue((summary["monotonicity_rate_abs"] >= 0.0).all())
        self.assertTrue((summary["monotonicity_rate_abs"] <= 1.0).all())

    def test_summarize_results_supports_dust_resuspension_card(self):
        rows = []
        for sample_id in [0, 1, 2]:
            for intensity, delta in [
                (0, 0.0),
                (10, 0.08),
                (25, 0.21),
                (50, 0.44),
                (75, 0.62),
                (100, 0.81),
            ]:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": "dust_resuspension",
                        "intensity": intensity,
                        "delta_pm25": delta,
                        "abs_delta_pm25": abs(delta),
                        "ood_event_count": 0,
                    }
                )

        summary = summarize_results(pd.DataFrame(rows), intensities=[0, 10, 25, 50, 75, 100], saturation_eps=0.05)
        self.assertEqual(set(summary["scenario_id"]), {"dust_resuspension"})
        row = summary.iloc[0]
        self.assertIn("effect_size_mean_abs_at_100", row.index)
        self.assertIn("monotonicity_rate_abs", row.index)
        self.assertIn("saturation_rate_75_to_100", row.index)
        self.assertGreater(float(row["effect_size_mean_abs_at_100"]), 0.0)


if __name__ == "__main__":
    unittest.main()
