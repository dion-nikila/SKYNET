# SKYNET Reliability Guidance + Empirical Uncertainty Method

## Scope
This note documents how SKYNET reports run-quality guidance and uncertainty context for one-hour-ahead PM2.5 forecasts.

## 1) Reliability guidance (heuristic, decomposed)
SKYNET computes a `reliability` block per run as a weighted heuristic composite of six component scores:
- `data_completeness`
- `domain_plausibility`
- `imputation_burden`
- `fallback_severity`
- `scenario_validity`
- `explainability_integrity`

The final score is mapped to a qualitative label (`high/moderate/low reliability guidance`).

### What this means
- It summarizes whether the run used stable inputs and well-supported model conditions.
- It is suitable for decision-support cautioning and comparison across runs.

### What this does not mean
- It is **not** a calibrated probability that the forecast is correct.
- It is **not** Bayesian/posterior uncertainty.

## 2) Empirical uncertainty guidance (when available)
If model metadata includes historical test residual traces (`plot_data.y_true` and `plot_data.preds`) or a precomputed uncertainty profile, SKYNET derives residual quantiles and builds empirical ranges around baseline/scenario predictions.

Default exported/displayed bands include 80% and 90% empirical residual ranges.

Scenario bands are widened relative to baseline using:
- lower reliability score,
- scenario mode risk adjustment (macro/guided/manual).

### Interpretation limits
- These bands are empirical error ranges from historical Haikou test behavior.
- They are strongest for Haikou-like operating conditions.
- They are **not** probabilistic guarantees.

## 3) Report-safe wording
Preferred wording:
- "heuristic reliability guidance"
- "empirical residual uncertainty ranges"
- "decision-support diagnostics"

Avoid wording that overstates calibration or certainty.
