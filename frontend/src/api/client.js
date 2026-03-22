const RUNTIME_SAME_ORIGIN_API_BASE =
  typeof window !== 'undefined' ? `${window.location.origin}/api/v1` : 'http://localhost:8000/api/v1'
const ENV = (typeof import.meta !== 'undefined' && import.meta.env) ? import.meta.env : {}
const API_BASE = ENV.VITE_API_BASE_URL || RUNTIME_SAME_ORIGIN_API_BASE
const DEFAULT_TIMEOUT_MS = Number(ENV.VITE_API_TIMEOUT_MS || 20000)

function ensureObject(value, label) {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Invalid API response: expected object at ${label}`)
  }
}

function ensureArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`Invalid API response: expected array at ${label}`)
  }
}

function ensureFiniteNumber(value, label) {
  const n = Number(value)
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid API response: expected finite number at ${label}`)
  }
}

function ensureBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`Invalid API response: expected boolean at ${label}`)
  }
}

function ensureString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Invalid API response: expected non-empty string at ${label}`)
  }
}

function ensureOneOf(value, allowed, label) {
  if (!allowed.includes(value)) {
    throw new Error(`Invalid API response: expected one of [${allowed.join(', ')}] at ${label}`)
  }
}

function validScenarioMode(mode) {
  return ['macro', 'guided_intervention', 'manual_custom', 'baseline'].includes(String(mode || ''))
}

export function validateInteractiveForecastResponse(payload) {
  ensureObject(payload, 'root')
  ensureObject(payload.meta, 'meta')
  ensureObject(payload.health, 'health')
  ensureObject(payload.baseline, 'baseline')
  ensureObject(payload.scenario, 'scenario')
  ensureObject(payload.delta, 'delta')
  ensureObject(payload.run, 'run')
  ensureObject(payload.baseline.prediction, 'baseline.prediction')
  ensureObject(payload.scenario.prediction, 'scenario.prediction')
  ensureObject(payload.health.history, 'health.history')
  ensureObject(payload.health.imputation, 'health.imputation')
  ensureObject(payload.health.fallback, 'health.fallback')
  ensureObject(payload.health.ood, 'health.ood')
  ensureArray(payload.scenario.applied_overrides, 'scenario.applied_overrides')

  ensureString(payload.meta.request_id, 'meta.request_id')
  ensureString(payload.meta.generated_at, 'meta.generated_at')
  ensureString(payload.meta.baseline_source, 'meta.baseline_source')
  ensureOneOf(payload.meta.forecast_mode, ['live', 'custom'], 'meta.forecast_mode')
  ensureBoolean(payload.meta.live_data_used, 'meta.live_data_used')
  ensureBoolean(payload.meta.overrides_applied, 'meta.overrides_applied')

  ensureString(payload.scenario.scenario_id, 'scenario.scenario_id')
  ensureString(payload.scenario.scenario_mode, 'scenario.scenario_mode')
  ensureOneOf(payload.scenario.scenario_mode, ['macro', 'guided_intervention', 'manual_custom', 'baseline'], 'scenario.scenario_mode')
  ensureFiniteNumber(payload.scenario.intensity, 'scenario.intensity')
  ensureBoolean(payload.run.persisted, 'run.persisted')

  ensureFiniteNumber(payload.health.quality_score, 'health.quality_score')
  ensureString(payload.health.quality_label, 'health.quality_label')
  ensureFiniteNumber(payload.health.history.target_hours, 'health.history.target_hours')
  ensureFiniteNumber(payload.health.history.used_hours, 'health.history.used_hours')
  ensureFiniteNumber(payload.health.imputation.ratio, 'health.imputation.ratio')
  ensureBoolean(payload.health.ood.flag, 'health.ood.flag')
  ensureFiniteNumber(payload.health.ood.score, 'health.ood.score')

  if (payload.health.reliability != null) {
    ensureObject(payload.health.reliability, 'health.reliability')
    ensureString(payload.health.reliability.method, 'health.reliability.method')
    ensureFiniteNumber(payload.health.reliability.score, 'health.reliability.score')
    ensureString(payload.health.reliability.label, 'health.reliability.label')
    ensureArray(payload.health.reliability.components, 'health.reliability.components')
    if (payload.health.reliability.notes != null) {
      ensureArray(payload.health.reliability.notes, 'health.reliability.notes')
    }
    payload.health.reliability.components.forEach((row, idx) => {
      ensureObject(row, `health.reliability.components[${idx}]`)
      ensureString(row.name, `health.reliability.components[${idx}].name`)
      ensureFiniteNumber(row.score, `health.reliability.components[${idx}].score`)
      ensureFiniteNumber(row.weight, `health.reliability.components[${idx}].weight`)
      ensureString(row.rationale, `health.reliability.components[${idx}].rationale`)
    })
  }

  if (payload.health.uncertainty != null) {
    ensureObject(payload.health.uncertainty, 'health.uncertainty')
    ensureString(payload.health.uncertainty.method, 'health.uncertainty.method')
    ensureBoolean(payload.health.uncertainty.available, 'health.uncertainty.available')
    ensureString(payload.health.uncertainty.note, 'health.uncertainty.note')
    if (payload.health.uncertainty.caveats != null) {
      ensureArray(payload.health.uncertainty.caveats, 'health.uncertainty.caveats')
    }
    if (payload.health.uncertainty.calibration_sample_size != null) {
      ensureFiniteNumber(payload.health.uncertainty.calibration_sample_size, 'health.uncertainty.calibration_sample_size')
    }
    ensureArray(payload.health.uncertainty.baseline_bands, 'health.uncertainty.baseline_bands')
    ensureArray(payload.health.uncertainty.scenario_bands, 'health.uncertainty.scenario_bands')
    payload.health.uncertainty.baseline_bands.forEach((row, idx) => {
      ensureObject(row, `health.uncertainty.baseline_bands[${idx}]`)
      ensureFiniteNumber(row.coverage_pct, `health.uncertainty.baseline_bands[${idx}].coverage_pct`)
      ensureFiniteNumber(row.lower, `health.uncertainty.baseline_bands[${idx}].lower`)
      ensureFiniteNumber(row.upper, `health.uncertainty.baseline_bands[${idx}].upper`)
      ensureFiniteNumber(row.width, `health.uncertainty.baseline_bands[${idx}].width`)
    })
    payload.health.uncertainty.scenario_bands.forEach((row, idx) => {
      ensureObject(row, `health.uncertainty.scenario_bands[${idx}]`)
      ensureFiniteNumber(row.coverage_pct, `health.uncertainty.scenario_bands[${idx}].coverage_pct`)
      ensureFiniteNumber(row.lower, `health.uncertainty.scenario_bands[${idx}].lower`)
      ensureFiniteNumber(row.upper, `health.uncertainty.scenario_bands[${idx}].upper`)
      ensureFiniteNumber(row.width, `health.uncertainty.scenario_bands[${idx}].width`)
    })
  }

  payload.scenario.applied_overrides.forEach((row, idx) => {
    ensureObject(row, `scenario.applied_overrides[${idx}]`)
    ensureString(row.feature, `scenario.applied_overrides[${idx}].feature`)
    ensureBoolean(row.clamped, `scenario.applied_overrides[${idx}].clamped`)
    ensureBoolean(row.direction_limited, `scenario.applied_overrides[${idx}].direction_limited`)
  })

  ensureFiniteNumber(payload.baseline.prediction.pm25_t_plus_1, 'baseline.prediction.pm25_t_plus_1')
  ensureFiniteNumber(payload.baseline.prediction.delta_pm25_t_plus_1, 'baseline.prediction.delta_pm25_t_plus_1')
  ensureFiniteNumber(payload.scenario.prediction.pm25_t_plus_1, 'scenario.prediction.pm25_t_plus_1')
  ensureFiniteNumber(payload.scenario.prediction.delta_pm25_t_plus_1, 'scenario.prediction.delta_pm25_t_plus_1')
  ensureFiniteNumber(payload.delta.pm25_change, 'delta.pm25_change')

  if (payload.meta.forecast_mode === 'custom' && payload.scenario.scenario_mode !== 'manual_custom') {
    throw new Error('Invalid API response: custom forecast_mode must return scenario_mode=manual_custom')
  }
  if (payload.scenario.scenario_mode === 'manual_custom' && payload.scenario.scenario_id !== 'custom_what_if') {
    throw new Error('Invalid API response: manual_custom must use scenario_id=custom_what_if')
  }
  if (payload.scenario.scenario_mode === 'guided_intervention' && payload.scenario.scenario_id !== 'guided_intervention') {
    throw new Error('Invalid API response: guided_intervention mode must use scenario_id=guided_intervention')
  }
  if (payload.scenario.scenario_mode === 'manual_custom' && Number(payload.scenario.intensity) !== 0) {
    throw new Error('Invalid API response: manual_custom intensity must be 0')
  }
  if (!validScenarioMode(payload.scenario.scenario_mode)) {
    throw new Error('Invalid API response: unknown scenario_mode')
  }

  return payload
}

async function request(path, options = {}) {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options
  const controller = new AbortController()
  const timer = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', ...(fetchOptions.headers || {}) },
      signal: controller.signal,
      ...fetchOptions
    })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(`API ${res.status}: ${text}`)
    }
    return res.json()
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new Error(`API request timed out after ${timeoutMs} ms`)
    }
    throw err
  } finally {
    if (timer) clearTimeout(timer)
  }
}

export function getScenarios() {
  return request('/scenarios')
}

export function getLocations() {
  return request('/locations')
}

export function getModelInfo() {
  return request('/model-info')
}

export function runInteractive(payload) {
  return request('/forecast/interactive', {
    method: 'POST',
    body: JSON.stringify(payload)
  }).then(validateInteractiveForecastResponse)
}

export function getRuns(limit = 50) {
  return request(`/runs?limit=${limit}`)
}

export function getRun(runId) {
  return request(`/runs/${runId}`)
}
