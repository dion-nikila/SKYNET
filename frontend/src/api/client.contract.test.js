import test from 'node:test'
import assert from 'node:assert/strict'

import { validateInteractiveForecastResponse } from './client.js'
import { buildForecastExportRecord } from '../utils/exportForecast.js'

function validPayload() {
  return {
    meta: {
      request_id: 't1',
      generated_at: '2026-03-20T10:00:00Z',
      forecast_mode: 'live',
      baseline_source: 'live_api',
      live_data_used: true,
      overrides_applied: false,
    },
    health: {
      quality_score: 0.8,
      quality_label: 'moderate reliability guidance',
      history: { target_hours: 72, used_hours: 72 },
      imputation: { ratio: 0.0 },
      fallback: {},
      ood: { flag: false, score: 0.0 },
    },
    baseline: {
      prediction: {
        delta_pm25_t_plus_1: 0.7,
        pm25_t_plus_1: 24.1,
      },
    },
    scenario: {
      scenario_id: 'baseline',
      scenario_mode: 'baseline',
      intensity: 0,
      applied_overrides: [],
      prediction: {
        delta_pm25_t_plus_1: 0.7,
        pm25_t_plus_1: 25.3,
      },
    },
    delta: {
      pm25_change: 1.2,
    },
    run: { persisted: false },
  }
}

test('validateInteractiveForecastResponse accepts valid payload shape', () => {
  const payload = validPayload()
  const out = validateInteractiveForecastResponse(payload)
  assert.equal(out, payload)
})

test('validateInteractiveForecastResponse rejects missing blocks', () => {
  const payload = validPayload()
  delete payload.health
  assert.throws(() => validateInteractiveForecastResponse(payload), /expected object at health/)
})

test('validateInteractiveForecastResponse rejects non-finite key numeric fields', () => {
  const payload = validPayload()
  payload.delta.pm25_change = 'not-a-number'
  assert.throws(() => validateInteractiveForecastResponse(payload), /finite number at delta\.pm25_change/)
})

test('validateInteractiveForecastResponse rejects scenario semantic mismatch', () => {
  const payload = validPayload()
  payload.meta.forecast_mode = 'custom'
  payload.scenario.scenario_mode = 'guided_intervention'
  payload.scenario.scenario_id = 'guided_intervention'
  assert.throws(
    () => validateInteractiveForecastResponse(payload),
    /custom forecast_mode must return scenario_mode=manual_custom/
  )
})

test('validateInteractiveForecastResponse accepts explicit manual custom semantics', () => {
  const payload = validPayload()
  payload.meta.forecast_mode = 'custom'
  payload.meta.overrides_applied = true
  payload.scenario.scenario_mode = 'manual_custom'
  payload.scenario.scenario_id = 'custom_what_if'
  payload.scenario.intensity = 0
  payload.scenario.applied_overrides = [
    { feature: 'NO2', clamped: false, direction_limited: false },
  ]
  assert.doesNotThrow(() => validateInteractiveForecastResponse(payload))
})

test('validateInteractiveForecastResponse accepts reliability and uncertainty guidance blocks', () => {
  const payload = validPayload()
  payload.health.reliability = {
    method: 'weighted_heuristic_components',
    score: 0.77,
    label: 'high reliability guidance',
    components: [
      { name: 'data_completeness', score: 0.9, weight: 0.3, rationale: 'coverage and gaps' },
    ],
    notes: ['heuristic guidance only'],
  }
  payload.health.uncertainty = {
    method: 'empirical_residual_quantiles_from_haikou_test_split',
    available: true,
    note: 'Empirical residual bands',
    caveats: ['Decision-support range only'],
    calibration_sample_size: 120,
    baseline_bands: [{ coverage_pct: 90, lower: 10.1, upper: 30.2, width: 20.1 }],
    scenario_bands: [{ coverage_pct: 90, lower: 11.0, upper: 31.4, width: 20.4 }],
  }
  assert.doesNotThrow(() => validateInteractiveForecastResponse(payload))
})

test('validateInteractiveForecastResponse rejects malformed reliability block', () => {
  const payload = validPayload()
  payload.health.reliability = {
    method: 'weighted_heuristic_components',
    score: 0.8,
    label: 'high reliability guidance',
    components: [{ name: 'data_completeness', score: 'oops', weight: 0.3, rationale: 'bad score type' }],
  }
  assert.throws(
    () => validateInteractiveForecastResponse(payload),
    /health\.reliability\.components\[0\]\.score/
  )
})

test('export semantics separate manual custom, guided intervention, and macro runs', () => {
  const base = {
    baseline: { inputs_snapshot: { pm25_current: 18 }, prediction: { pm25_t_plus_1: 19, delta_pm25_t_plus_1: 1 }, shap: {} },
    delta: { pm25_change: 1, delta_shap: {} },
    health: { quality_score: 0.8, quality_label: 'moderate reliability guidance', history: {}, imputation: {}, fallback: {}, ood: {} },
    run: { persisted: false },
  }

  const manual = buildForecastExportRecord({
    ...base,
    meta: { forecast_mode: 'custom' },
    scenario: { scenario_id: 'custom_what_if', scenario_mode: 'manual_custom', intensity: 0, prediction: { pm25_t_plus_1: 20, delta_pm25_t_plus_1: 2 }, applied_overrides: [] },
  })
  assert.equal(manual.scenario_mode, 'manual_custom')
  assert.equal(manual.scenario_intensity, '')

  const guided = buildForecastExportRecord({
    ...base,
    meta: { forecast_mode: 'live' },
    scenario: { scenario_id: 'guided_intervention', scenario_mode: 'guided_intervention', intensity: 70, prediction: { pm25_t_plus_1: 20, delta_pm25_t_plus_1: 2 }, applied_overrides: [] },
  })
  assert.equal(guided.scenario_mode, 'guided_intervention')
  assert.equal(guided.scenario_intensity, 70)
  assert.match(guided.scenario_intensity_note, /fixed baseline intensity/i)

  const macro = buildForecastExportRecord({
    ...base,
    meta: { forecast_mode: 'live' },
    scenario: { scenario_id: 'traffic_gridlock', scenario_mode: 'macro', intensity: 73, prediction: { pm25_t_plus_1: 20, delta_pm25_t_plus_1: 2 }, applied_overrides: [] },
  })
  assert.equal(macro.scenario_mode, 'macro')
  assert.equal(macro.scenario_intensity, 73)
  assert.equal(macro.scenario_intensity_note, '')
})

test('export record preserves fallback source and baseline timestamp honesty', () => {
  const row = buildForecastExportRecord({
    meta: {
      generated_at: '2026-03-20T10:30:00Z',
      forecast_mode: 'custom',
      baseline_source: 'reference_profile',
      baseline_timestamp: '2023-08-17T08:00:00+00:00',
      live_data_used: false,
      overrides_applied: true,
    },
    baseline: {
      inputs_snapshot: { pm25_current: 18.4 },
      prediction: { pm25_t_plus_1: 22.1, delta_pm25_t_plus_1: 3.7 },
      shap: {},
    },
    scenario: {
      scenario_id: 'custom_what_if',
      scenario_mode: 'manual_custom',
      intensity: 0,
      prediction: { pm25_t_plus_1: 25.4, delta_pm25_t_plus_1: 7.0 },
      applied_overrides: [],
    },
    delta: { pm25_change: 3.3, delta_shap: {} },
    health: { quality_score: 0.62, quality_label: 'moderate reliability guidance', history: {}, imputation: {}, fallback: {}, ood: {} },
    run: { persisted: false },
  })

  assert.equal(row.baseline_source, 'reference_profile')
  assert.equal(row.baseline_source_label, 'Reference dataset baseline')
  assert.equal(row.baseline_timestamp, '2023-08-17T08:00:00+00:00')
  assert.equal(row.live_data_used, false)
  assert.match(row.baseline_context_note, /non-live baseline context/i)
  assert.match(row.baseline_context_note, /2023-08-17T08:00:00\+00:00/)
})
