import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Box,
  Alert,
  CircularProgress,
  Typography,
  Chip,
  Stack,
  Tabs,
  Tab,
  Button,
  Divider,
} from '@mui/material'

import { getLocations, getModelInfo, getScenarios, runInteractive } from './api/client'
import TopBar from './components/TopBar'
import ForecastControlsPanel from './components/ForecastControlsPanel'
import ScenarioPanel from './components/ScenarioPanel'
import TrustPanel from './components/TrustPanel'
import ForecastComparison from './components/ForecastComparison'
import ExplainabilityPanel from './components/ExplainabilityPanel'
import RunHistoryPanel from './components/RunHistoryPanel'
import CustomWhatIfPanel from './components/CustomWhatIfPanel'
import { labelScenario } from './utils/labels'
import { downloadForecastCsv, openForecastPrintReport } from './utils/exportForecast'
import PredictionSpotlight from './components/PredictionSpotlight'
import QuickGuideCard from './components/QuickGuideCard'
import { sk, workspaceRowSx, workspaceLeftColSx, workspaceRightColSx } from './theme/tokens'

function formatWhen(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return String(ts)
  return d.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function validateCoordinateInput(rawValue, kind) {
  const raw = String(rawValue ?? '').trim()
  const label = kind === 'lat' ? 'Latitude' : 'Longitude'
  const min = kind === 'lat' ? -90 : -180
  const max = kind === 'lat' ? 90 : 180

  if (!raw) {
    return { valid: false, value: null, error: `${label} is required.` }
  }

  const n = Number(raw)
  if (!Number.isFinite(n)) {
    return { valid: false, value: null, error: `${label} must be a valid number.` }
  }
  if (n < min || n > max) {
    return { valid: false, value: null, error: `${label} must be between ${min} and ${max}.` }
  }
  return { valid: true, value: n, error: '' }
}

const CUSTOM_FIELD_RANGES = {
  PM10: { label: 'PM10', min: 0, max: 1000 },
  NO2: { label: 'NO2', min: 0, max: 500 },
  CO: { label: 'CO', min: 0, max: 50 },
  temperature: { label: 'Temperature', min: -50, max: 60 },
  humidity: { label: 'Humidity', min: 0, max: 100 },
  wind_speed: { label: 'Wind speed', min: 0, max: 80 },
  pressure: { label: 'Pressure', min: 850, max: 1100 },
  O3: { label: 'O3', min: 0, max: 500 },
  SO2: { label: 'SO2', min: 0, max: 500 },
}

const CUSTOM_GROUP_PRESETS = {
  dispersion_conditions: {
    title: 'Dispersion conditions',
    description: 'Wind, humidity, and pressure changes that alter pollutant dispersion.',
    targets: { wind_speed: 'decrease', humidity: 'increase', pressure: 'increase' },
  },
  traffic_urban_emissions: {
    title: 'Traffic / urban emissions',
    description: 'NO2, CO, and PM10 shifts that represent traffic-heavy urban stress.',
    targets: { NO2: 'increase', CO: 'increase', PM10: 'increase' },
  },
  dust_dry_conditions: {
    title: 'Dust / dry conditions',
    description: 'Dust and dryness pattern with PM10 rise, drier air, and wind changes.',
    targets: { PM10: 'increase', humidity: 'decrease', wind_speed: 'increase' },
  },
  heat_photochemical_stress: {
    title: 'Heat / photochemical stress',
    description: 'Temperature-humidity stress profile, with optional ozone increase.',
    targets: { temperature: 'increase', humidity: 'decrease', O3: 'increase' },
  },
}

const PRESSURE_MODEL_TO_UI_SCALE = 10

function boundsForUiFeature(feature, bounds) {
  const q = bounds || {}
  if (feature !== 'pressure') return q
  const out = { ...q }
  Object.keys(out).forEach((k) => {
    const n = Number(out[k])
    if (Number.isFinite(n)) out[k] = n * PRESSURE_MODEL_TO_UI_SCALE
  })
  return out
}

function quantileTargetForDirection(feature, bounds, direction, impactMode, fallbackRange) {
  const q = boundsForUiFeature(feature, bounds)
  const stronger = impactMode === 'stronger_realistic'
  const orderedInc = stronger ? ['q99', 'q95', 'q75'] : ['q95', 'q75', 'q50']
  const orderedDec = stronger ? ['q01', 'q05', 'q25'] : ['q05', 'q25', 'q50']
  const keys = direction === 'increase' ? orderedInc : orderedDec
  const min = Number(fallbackRange?.min)
  const max = Number(fallbackRange?.max)
  const hasFallbackRange = Number.isFinite(min) && Number.isFinite(max) && max > min

  for (const k of keys) {
    const n = Number(q?.[k])
    if (!Number.isFinite(n)) continue
    if (hasFallbackRange && (n < min || n > max)) continue
    return n
  }

  if (hasFallbackRange) {
    const alpha = stronger ? (direction === 'increase' ? 0.92 : 0.08) : (direction === 'increase' ? 0.78 : 0.22)
    return min + (max - min) * alpha
  }
  return null
}

function estimateCustomImpactPreview(overridesPayload, impactMode, modelInfo) {
  const rows = Object.entries(overridesPayload || {})
  if (!rows.length) {
    return {
      level: 'low',
      score: 0,
      note: 'Estimated preview only — add at least one override to assess likely impact; final forecast may differ after full model evaluation.',
      factors: [],
    }
  }

  const bounds = modelInfo?.bounds_preview || {}
  const importanceRows = Array.isArray(modelInfo?.feature_importance) ? modelInfo.feature_importance : []
  const importanceMap = importanceRows.reduce((acc, row) => {
    const f = String(row?.feature || '')
    const pct = Number(row?.pct || 0)
    if (f) acc[f] = Math.max(0, pct / 100)
    return acc
  }, {})

  const scored = rows.map(([feature, value]) => {
    const q = boundsForUiFeature(feature, bounds?.[feature] || {})
    const fieldRange = CUSTOM_FIELD_RANGES[feature] || null
    const rangeMin = Number(fieldRange?.min)
    const rangeMax = Number(fieldRange?.max)
    const hasRange = Number.isFinite(rangeMin) && Number.isFinite(rangeMax) && rangeMax > rangeMin
    const inRange = (n) => Number.isFinite(n) && (!hasRange || (n >= rangeMin && n <= rangeMax))

    const v = Number(value)
    const q05 = Number(q?.q05)
    const q95 = Number(q?.q95)
    const q25 = Number(q?.q25)
    const q75 = Number(q?.q75)
    const q50 = Number(q?.q50)

    const low = inRange(q05) ? q05 : inRange(q25) ? q25 : hasRange ? rangeMin : v
    const high = inRange(q95) ? q95 : inRange(q75) ? q75 : hasRange ? rangeMax : v
    const mid = inRange(q50)
      ? q50
      : (Number.isFinite(low) && Number.isFinite(high) ? (low + high) / 2 : v)

    const span = Number.isFinite(low) && Number.isFinite(high) ? Math.max(Math.abs(high - low), 1e-6) : 1.0
    const dist = Number.isFinite(mid) ? Math.min(1.0, Math.abs(v - mid) / span) : 0.5
    const edgeBoost = Number.isFinite(low) && Number.isFinite(high) && (v < low || v > high) ? 1.1 : 1.0
    const leverage = Number(importanceMap[feature] ?? 0.35)
    const contribution = (0.35 + dist) * (0.45 + leverage) * edgeBoost
    return { feature, contribution: Number.isFinite(contribution) ? contribution : 0 }
  })

  let score = scored.reduce((sum, r) => sum + Math.max(0, r.contribution), 0)
  if (impactMode === 'stronger_realistic') score *= 1.2

  let level = 'low'
  let note = 'Estimated preview only — impact appears low based on selected controls and typical model sensitivity; final forecast may differ after full model evaluation.'
  if (score >= 2.4) {
    level = 'high'
    note = 'Estimated preview only — impact appears high while still constrained by baseline anchoring and training-range bounds; final forecast may differ after full model evaluation.'
  } else if (score >= 1.2) {
    level = 'medium'
    note = 'Estimated preview only — impact appears medium and should produce a visible but bounded forecast shift; final forecast may differ after full model evaluation.'
  }

  const factors = scored
    .sort((a, b) => b.contribution - a.contribution)
    .slice(0, 3)
    .map((x) => x.feature)

  return {
    level,
    score: Number(score.toFixed(2)),
    note,
    factors,
  }
}

function validateCustomField(rawValue, key) {
  const cfg = CUSTOM_FIELD_RANGES[key]
  const raw = String(rawValue ?? '').trim()
  if (!raw) return { valid: true, value: null, error: '' }
  const n = Number(raw)
  if (!Number.isFinite(n)) {
    return { valid: false, value: null, error: `${cfg.label} must be a valid number.` }
  }
  if (n < cfg.min || n > cfg.max) {
    return { valid: false, value: null, error: `${cfg.label} must be between ${cfg.min} and ${cfg.max}.` }
  }
  return { valid: true, value: n, error: '' }
}

function parseApiErrorMessage(err) {
  const raw = String(err?.message || err || '')
  const jsonStart = raw.indexOf('{')
  if (jsonStart >= 0) {
    try {
      const parsed = JSON.parse(raw.slice(jsonStart))
      if (parsed?.detail?.message) return String(parsed.detail.message)
      if (typeof parsed?.detail === 'string') return String(parsed.detail)
    } catch (_ignore) {
      // Keep original message.
    }
  }
  return raw
}

const LOCAL_RUNS_KEY = 'skynet_local_runs_v1'
const LOCAL_RUNS_MAX = 40

function makeLocalRunId() {
  return `local_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

function loadLocalRuns() {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(LOCAL_RUNS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch (_err) {
    return []
  }
}

function saveLocalRuns(runs) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(LOCAL_RUNS_KEY, JSON.stringify(runs))
  } catch (_err) {
    // Ignore storage errors; app still works without persisted local history.
  }
}

function buildLocalRunEntry(response, requestPayload) {
  const forecastMode = requestPayload?.forecast_mode || response?.meta?.forecast_mode || 'live'
  const scenarioIdRaw = response?.scenario?.scenario_id || requestPayload?.scenario?.scenario_id || 'baseline'
  const intensity = Number(response?.scenario?.intensity ?? requestPayload?.scenario?.intensity ?? 0)
  const scenarioMode = classifyScenarioRun({
    forecastMode,
    scenarioId: scenarioIdRaw,
    intensity,
    scenarioMode: response?.scenario?.scenario_mode,
  })
  const scenarioId = scenarioMode === 'guided_intervention'
    ? 'guided_intervention'
    : scenarioMode === 'baseline'
      ? 'baseline'
      : scenarioMode === 'manual_custom'
        ? 'custom_what_if'
        : scenarioIdRaw
  return {
    run_id: response?.run?.run_id || makeLocalRunId(),
    created_at: response?.meta?.generated_at || new Date().toISOString(),
    scenario_id: scenarioId,
    scenario_mode: scenarioMode,
    intensity,
    pm25_change: Number(response?.delta?.pm25_change || 0),
    pm25_t_plus_1: Number(response?.scenario?.prediction?.pm25_t_plus_1 || 0),
    ood_flag: Boolean(response?.health?.ood?.flag),
    forecast_mode: forecastMode,
    location_name: String(requestPayload?.location?.name || ''),
    location_lat: Number(requestPayload?.location?.lat),
    location_lon: Number(requestPayload?.location?.lon),
    request_payload: requestPayload,
    response_json: response,
  }
}

function classifyScenarioRun({ forecastMode, scenarioId, intensity, scenarioMode }) {
  const explicitMode = String(scenarioMode || '').toLowerCase()
  if (['macro', 'guided_intervention', 'manual_custom', 'baseline'].includes(explicitMode)) {
    return explicitMode
  }
  const mode = String(forecastMode || '').toLowerCase()
  const sid = String(scenarioId || '')
  const level = Number(intensity || 0)

  if (mode === 'custom' || sid === 'custom_what_if') return 'manual_custom'
  if (sid === 'guided_intervention') return 'guided_intervention'
  if (sid === 'baseline') return 'baseline'
  if (mode === 'live' && sid === 'custom' && level > 0) return 'guided_intervention'
  if (mode === 'live' && sid === 'custom' && level <= 0) return 'baseline'
  if (!sid) return 'baseline'
  return 'macro'
}

export default function App() {
  const initialCustomDraft = useMemo(
    () =>
      Object.keys(CUSTOM_FIELD_RANGES).reduce((acc, key) => {
        acc[key] = ''
        return acc
      }, {}),
    []
  )

  const [location, setLocation] = useState({ lat: 20.0442, lon: 110.1999, name: 'Haikou' })
  const [selectedLocationId, setSelectedLocationId] = useState('haikou_cn')
  const [useManualLocation, setUseManualLocation] = useState(false)
  const [forecastMode, setForecastMode] = useState('live')
  const [locationDraft, setLocationDraft] = useState({
    lat: String(20.0442),
    lon: String(110.1999)
  })
  const [customDraft, setCustomDraft] = useState(initialCustomDraft)
  const [customImpactMode, setCustomImpactMode] = useState('conservative')
  const [customTouched, setCustomTouched] = useState({})
  const [liveFetchUnavailable, setLiveFetchUnavailable] = useState(false)
  const [options, setOptions] = useState({
    history_hours_target: 72,
    top_k_drivers: 6,
    ood: { soft_q: 0.05, hard_q: 0.01 }
  })
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [loadingRunId, setLoadingRunId] = useState(null)
  const [activeRunId, setActiveRunId] = useState(null)
  const [localRuns, setLocalRuns] = useState(() => loadLocalRuns())
  const [viewTab, setViewTab] = useState('overview')
  const latCheck = useMemo(() => validateCoordinateInput(locationDraft.lat, 'lat'), [locationDraft.lat])
  const lonCheck = useMemo(() => validateCoordinateInput(locationDraft.lon, 'lon'), [locationDraft.lon])
  const presetCoordinatesValid = Number.isFinite(Number(location.lat)) && Number.isFinite(Number(location.lon))
  const coordinatesValid = useManualLocation ? Boolean(latCheck.valid && lonCheck.valid) : presetCoordinatesValid
  const coordinateErrors = {
    lat: useManualLocation ? latCheck.error : '',
    lon: useManualLocation ? lonCheck.error : ''
  }
  const customValidation = useMemo(
    () =>
      Object.keys(CUSTOM_FIELD_RANGES).reduce((acc, key) => {
        acc[key] = validateCustomField(customDraft[key], key)
        return acc
      }, {}),
    [customDraft]
  )
  const customErrors = useMemo(
    () =>
      Object.keys(customValidation).reduce((acc, key) => {
        acc[key] = customValidation[key].error
        return acc
      }, {}),
    [customValidation]
  )
  const customAllValid = useMemo(
    () => Object.values(customValidation).every((v) => v.valid),
    [customValidation]
  )
  const customHasAnyValue = useMemo(
    () => Object.values(customDraft).some((v) => String(v ?? '').trim() !== ''),
    [customDraft]
  )
  const customOverridesPayload = useMemo(
    () =>
      Object.keys(customValidation).reduce((acc, key) => {
        const v = customValidation[key]
        if (v.valid && v.value !== null) acc[key] = v.value
        return acc
      }, {}),
    [customValidation]
  )
  const scenarioId = result?.scenario?.scenario_id || ''
  const scenarioMode = useMemo(
    () =>
      classifyScenarioRun({
        forecastMode: result?.meta?.forecast_mode,
        scenarioId: result?.scenario?.scenario_id,
        intensity: result?.scenario?.intensity,
        scenarioMode: result?.scenario?.scenario_mode,
      }),
    [result]
  )
  const scenarioOverrideCount = Array.isArray(result?.scenario?.applied_overrides) ? result.scenario.applied_overrides.length : 0
  const hasScenario = Boolean(
    result && (
      scenarioOverrideCount > 0 ||
      scenarioMode === 'macro' ||
      scenarioMode === 'guided_intervention'
    )
  )
  const customAppliedOverrides = useMemo(() => {
    const rows = Array.isArray(result?.scenario?.applied_overrides) ? result.scenario.applied_overrides : []
    if (!rows.length) return []
    const mode = String(result?.meta?.forecast_mode || '')
    const sid = String(result?.scenario?.scenario_id || '')
    if (mode !== 'custom' && sid !== 'custom_what_if') return []
    return rows
  }, [result])
  const customRunImpactPreview = useMemo(() => {
    const mode = String(result?.meta?.forecast_mode || '')
    const sid = String(result?.scenario?.scenario_id || '')
    if (mode !== 'custom' && sid !== 'custom_what_if') return null
    return result?.scenario?.impact_preview || null
  }, [result])

  const scenariosQuery = useQuery({ queryKey: ['scenarios'], queryFn: getScenarios })
  const locationsQuery = useQuery({ queryKey: ['locations'], queryFn: getLocations })
  const modelInfoQuery = useQuery({ queryKey: ['model-info'], queryFn: getModelInfo })
  const customImpactPreviewEstimate = useMemo(
    () => estimateCustomImpactPreview(customOverridesPayload, customImpactMode, modelInfoQuery.data),
    [customOverridesPayload, customImpactMode, modelInfoQuery.data]
  )

  const runMutation = useMutation({
    mutationFn: runInteractive,
    onSuccess: (data, variables) => {
      setErrorMsg('')
      setLiveFetchUnavailable(false)
      setResult(data)
      const localEntry = buildLocalRunEntry(data, variables)
      setActiveRunId(localEntry.run_id)
      setLocalRuns((prev) => {
        const rows = Array.isArray(prev) ? prev : []
        const dedup = rows.filter((r) => r.run_id !== localEntry.run_id)
        return [localEntry, ...dedup].slice(0, LOCAL_RUNS_MAX)
      })
    },
    onError: (e) => {
      const msg = parseApiErrorMessage(e)
      setErrorMsg(msg)
      if (msg.toLowerCase().includes('live data fetch failed')) {
        setLiveFetchUnavailable(true)
      }
    }
  })

  const bootstrapRef = useRef(false)

  useEffect(() => {
    if (!useManualLocation) return
    setLocation((prev) => {
      const nextLat = latCheck.valid ? latCheck.value : prev.lat
      const nextLon = lonCheck.valid ? lonCheck.value : prev.lon
      if (nextLat === prev.lat && nextLon === prev.lon) return prev
      return { ...prev, lat: nextLat, lon: nextLon }
    })
  }, [latCheck.valid, latCheck.value, lonCheck.valid, lonCheck.value, useManualLocation])

  useEffect(() => {
    if (!Array.isArray(locationsQuery.data) || !locationsQuery.data.length || useManualLocation) return
    const selected = locationsQuery.data.find((x) => x.location_id === selectedLocationId) || locationsQuery.data[0]
    if (!selected) return
    setSelectedLocationId(selected.location_id)
    setLocation({
      lat: Number(selected.lat),
      lon: Number(selected.lon),
      name: String(selected.name || ''),
    })
    setLocationDraft({
      lat: String(selected.lat),
      lon: String(selected.lon),
    })
  }, [locationsQuery.data, selectedLocationId, useManualLocation])

  useEffect(() => {
    saveLocalRuns(localRuns)
  }, [localRuns])

  const applyScenario = (scenario) => {
    if (!coordinatesValid) {
      setErrorMsg('Enter valid latitude and longitude before running forecast scenarios.')
      return
    }
    setErrorMsg('')
    const requestLocation = useManualLocation
      ? {
          ...location,
          lat: latCheck.value,
          lon: lonCheck.value,
          location_id: null,
        }
      : {
          ...location,
          location_id: selectedLocationId || null,
        }
    runMutation.mutate({
      request_id: crypto.randomUUID(),
      forecast_mode: 'live',
      location: requestLocation,
      time: { mode: 'now' },
      scenario,
      options
    })
  }

  const applyCustomGroupPreset = (groupId) => {
    const preset = CUSTOM_GROUP_PRESETS[groupId]
    if (!preset) return

    const bounds = modelInfoQuery.data?.bounds_preview || {}
    setCustomDraft((prev) => {
      const next = { ...prev }
      Object.entries(preset.targets).forEach(([feature, direction]) => {
        const fallbackRange = CUSTOM_FIELD_RANGES[feature]
        const target = quantileTargetForDirection(feature, bounds?.[feature], direction, customImpactMode, fallbackRange)
        if (Number.isFinite(target)) {
          const min = Number(fallbackRange?.min)
          const max = Number(fallbackRange?.max)
          const bounded = Number.isFinite(min) && Number.isFinite(max)
            ? Math.min(max, Math.max(min, Number(target)))
            : Number(target)
          next[feature] = String(Number(bounded).toFixed(2))
        }
      })
      return next
    })
    setCustomTouched({})
    setErrorMsg('')
  }

  const runCustomWhatIf = ({ allowEmpty = false } = {}) => {
    if (!coordinatesValid) {
      setErrorMsg('Enter valid latitude and longitude before running custom what-if forecasts.')
      return
    }
    if (!customAllValid) {
      setErrorMsg('Fix invalid custom values before running the forecast.')
      return
    }
    if (!allowEmpty && !customHasAnyValue) {
      setErrorMsg('Provide at least one custom override value before running a custom what-if forecast.')
      return
    }
    setErrorMsg('')
    const requestLocation = useManualLocation
      ? {
          ...location,
          lat: latCheck.value,
          lon: lonCheck.value,
          location_id: null,
        }
      : {
          ...location,
          location_id: selectedLocationId || null,
        }
    runMutation.mutate({
      request_id: crypto.randomUUID(),
      forecast_mode: 'custom',
      custom_impact_mode: customImpactMode,
      location: requestLocation,
      time: { mode: 'now' },
      scenario: { type: 'custom', intensity: 0, items: [] },
      custom_overrides: Object.keys(customOverridesPayload).length ? customOverridesPayload : undefined,
      options,
    })
  }

  const clearCustomInputs = () => {
    setCustomDraft(initialCustomDraft)
    setCustomTouched({})
  }

  const runBaseline = () => {
    if (forecastMode === 'custom') {
      runCustomWhatIf({ allowEmpty: true })
      return
    }
    applyScenario({ type: 'custom', intensity: 0, items: [] })
  }

  useEffect(() => {
    if (!bootstrapRef.current && scenariosQuery.data?.length && forecastMode === 'live') {
      bootstrapRef.current = true
      runBaseline()
    }
  }, [scenariosQuery.data, forecastMode])

  const loadRun = (runId) => {
    setLoadingRunId(runId)
    try {
      const row = (localRuns || []).find((r) => r.run_id === runId)
      const payload = row?.response_json
      const compatible =
        payload &&
        payload.meta &&
        payload.baseline &&
        payload.scenario &&
        payload.delta
      if (!compatible) {
        setErrorMsg('This saved local run is not compatible with the current dashboard format.')
        return
      }
      const requestPayload = row?.request_payload || {}
      const replayMode = String(requestPayload.forecast_mode || payload?.meta?.forecast_mode || 'live')
      const replayImpactMode = String(requestPayload.custom_impact_mode || payload?.meta?.custom_impact_mode || 'conservative')
      const replayLocation = requestPayload.location || {}
      setForecastMode(replayMode === 'custom' ? 'custom' : 'live')
      setCustomImpactMode(replayMode === 'custom' ? replayImpactMode : 'conservative')
      if (replayLocation?.location_id) {
        setUseManualLocation(false)
        setSelectedLocationId(String(replayLocation.location_id))
      } else if (Number.isFinite(Number(replayLocation?.lat)) && Number.isFinite(Number(replayLocation?.lon))) {
        setUseManualLocation(true)
      }
      if (Number.isFinite(Number(replayLocation?.lat)) && Number.isFinite(Number(replayLocation?.lon))) {
        setLocation((prev) => ({
          ...prev,
          lat: Number(replayLocation.lat),
          lon: Number(replayLocation.lon),
          name: String(replayLocation?.name || prev?.name || ''),
        }))
        setLocationDraft({
          lat: String(replayLocation.lat),
          lon: String(replayLocation.lon),
        })
      }
      setResult(payload)
      setActiveRunId(runId)
      setViewTab('overview')
      setErrorMsg('')
    } catch (e) {
      setErrorMsg(String(e.message || e))
    } finally {
      setLoadingRunId(null)
    }
  }

  const rerunFromHistory = (runId) => {
    const row = (localRuns || []).find((r) => r.run_id === runId)
    const requestPayload = row?.request_payload
    if (!requestPayload) {
      setErrorMsg('This run does not include a replayable request payload.')
      return
    }
    const replayMode = String(requestPayload.forecast_mode || 'live')
    const replayImpactMode = String(requestPayload.custom_impact_mode || 'conservative')
    const replayLocation = requestPayload.location || {}
    setForecastMode(replayMode === 'custom' ? 'custom' : 'live')
    setCustomImpactMode(replayMode === 'custom' ? replayImpactMode : 'conservative')
    if (replayLocation?.location_id) {
      setUseManualLocation(false)
      setSelectedLocationId(String(replayLocation.location_id))
    } else {
      setUseManualLocation(true)
    }
    if (Number.isFinite(Number(replayLocation?.lat)) && Number.isFinite(Number(replayLocation?.lon))) {
      setLocation((prev) => ({
        ...prev,
        lat: Number(replayLocation.lat),
        lon: Number(replayLocation.lon),
        name: String(replayLocation?.name || prev?.name || ''),
      }))
      setLocationDraft({
        lat: String(replayLocation.lat),
        lon: String(replayLocation.lon),
      })
    }
    setErrorMsg('')
    runMutation.mutate({
      ...requestPayload,
      request_id: crypto.randomUUID(),
    })
  }

  const deleteLocalRun = (runId) => {
    setLocalRuns((prev) => (Array.isArray(prev) ? prev.filter((r) => r.run_id !== runId) : []))
    if (activeRunId === runId) setActiveRunId(null)
  }

  const clearLocalRuns = () => {
    setLocalRuns([])
    setActiveRunId(null)
  }

  const exportCsv = () => {
    if (!result) return
    try {
      downloadForecastCsv(result)
    } catch (err) {
      setErrorMsg(`CSV export failed: ${String(err?.message || err)}`)
    }
  }

  const exportPdf = () => {
    if (!result) return
    try {
      openForecastPrintReport(result)
    } catch (err) {
      setErrorMsg(`PDF export failed: ${String(err?.message || err)}`)
    }
  }

  const activeRunLabel = result
    ? hasScenario
      ? scenarioMode === 'manual_custom'
        ? `${labelScenario('custom_what_if')} (manual custom overrides)`
        : scenarioMode === 'guided_intervention'
          ? `${labelScenario('guided_intervention')} (guided mode, calibrated intensity ${result.scenario.intensity})`
          : `${labelScenario(result.scenario.scenario_id)} (intensity ${result.scenario.intensity})`
      : 'Baseline forecast'
    : 'No run yet'

  return (
    <Box sx={{ minHeight: '100vh' }} className="skynet-shell">
      <TopBar
        forecastMode={forecastMode}
        locationName={location?.name || ''}
        modelInfo={modelInfoQuery.data}
      />

      <Box
        sx={{
          px: { xs: 0.75, sm: 1.8, md: 2.6, xl: 3.2 },
          py: { xs: 0.95, sm: 1.6, md: 1.9 },
          width: '100%',
          maxWidth: '100%',
          mx: 'auto',
        }}
      >
        <Stack spacing={{ xs: 1.35, sm: 2.1, md: 2.45 }}>
          <ForecastControlsPanel
            location={location}
            onLocationNameChange={(name) => setLocation((s) => ({ ...s, name }))}
            forecastMode={forecastMode}
            onForecastModeChange={setForecastMode}
            locations={locationsQuery.data || []}
            selectedLocationId={selectedLocationId}
            onLocationSelect={(locationId) => {
              setSelectedLocationId(locationId)
              const row = (locationsQuery.data || []).find((x) => x.location_id === locationId)
              if (row) {
                setLocation({
                  lat: Number(row.lat),
                  lon: Number(row.lon),
                  name: String(row.name || ''),
                })
                setLocationDraft({
                  lat: String(row.lat),
                  lon: String(row.lon),
                })
              }
            }}
            useManualLocation={useManualLocation}
            onUseManualLocationChange={setUseManualLocation}
            locationDraft={locationDraft}
            setLocationDraft={setLocationDraft}
            coordinateErrors={coordinateErrors}
            coordinatesValid={coordinatesValid}
            onRunBaseline={runBaseline}
            loading={runMutation.isPending}
          />

          <Stack spacing={1.15}>
            {errorMsg ? <Alert severity="error">{errorMsg}</Alert> : null}

            {liveFetchUnavailable && forecastMode === 'live' ? (
              <Alert
                severity="warning"
                action={
                  <Button color="inherit" size="small" onClick={() => setForecastMode('custom')}>
                    Switch to Custom What-If
                  </Button>
                }
              >
                Live data is unavailable right now. Continue with Custom What-If using baseline history context.
              </Alert>
            ) : null}

            {scenariosQuery.isError ? (
              <Alert
                severity="error"
                action={
                  <Button color="inherit" size="small" onClick={() => scenariosQuery.refetch()}>
                    Retry
                  </Button>
                }
              >
                Scenario templates could not be loaded from backend. Check API status and retry.
              </Alert>
            ) : null}

            {locationsQuery.isError ? (
              <Alert
                severity="warning"
                action={
                  <Button color="inherit" size="small" onClick={() => locationsQuery.refetch()}>
                    Retry
                  </Button>
                }
              >
                Preset location list failed to load. You can still run forecasts using manual coordinates.
              </Alert>
            ) : null}

            {modelInfoQuery.isError ? (
              <Alert
                severity="warning"
                action={
                  <Button color="inherit" size="small" onClick={() => modelInfoQuery.refetch()}>
                    Retry
                  </Button>
                }
              >
                Model metadata failed to load. Forecast runs can still execute, but model-information details are currently limited.
              </Alert>
            ) : null}

            {!modelInfoQuery.isError && modelInfoQuery.data?.run_logging_status === 'degraded' ? (
              <Alert severity="warning">
                Backend run logging is degraded. Forecasting remains available, but backend `/runs` persistence is currently unavailable.
              </Alert>
            ) : null}
          </Stack>

          {(scenariosQuery.isLoading || modelInfoQuery.isLoading) && !result ? (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <CircularProgress size={20} />
              <Typography variant="body2">Loading dashboard metadata...</Typography>
            </Box>
          ) : null}

          <Box className="prediction-star-shell">
            <PredictionSpotlight
              data={result}
              hasScenario={hasScenario}
              onRefreshBaseline={runBaseline}
              loading={runMutation.isPending}
              modelInfo={modelInfoQuery.data}
            />
          </Box>

          <Box sx={workspaceRowSx}>
            <Box sx={workspaceLeftColSx}>
                <Box className="analysis-shell" sx={{ pb: 0.85 }}>
                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    spacing={0.75}
                    alignItems={{ xs: 'flex-start', sm: 'center' }}
                    justifyContent="space-between"
                  >
                    <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title, lineHeight: 1.3, mb: 0.25 }}>
                      Scenario intelligence, explanation evidence, and local run memory
                    </Typography>
                    {viewTab !== 'explain' && result ? (
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => setViewTab('explain')}
                        sx={{ fontWeight: 700 }}
                      >
                        View Explainability
                      </Button>
                    ) : null}
                  </Stack>

                  <Tabs
                    value={viewTab}
                    onChange={(_, v) => setViewTab(v)}
                    variant="fullWidth"
                    TabIndicatorProps={{ children: <span className="tab-indicator-inner" /> }}
                    sx={{
                      minHeight: 44,
                      mt: 1.15,
                      pt: 0.65,
                      borderTop: `1px solid ${sk.divider}`,
                      py: 0.15,
                      bgcolor: 'transparent',
                      '& .MuiTabs-indicator': {
                        height: 3,
                        display: 'flex',
                        justifyContent: 'center',
                        backgroundColor: 'transparent',
                      },
                      '& .tab-indicator-inner': {
                        width: '100%',
                        maxWidth: 128,
                        borderRadius: 999,
                        backgroundColor: 'primary.main',
                      },
                      '& .MuiTab-root': {
                        minHeight: 38,
                        borderRadius: 1,
                        fontWeight: 600,
                        fontSize: { xs: '0.79rem', sm: '0.875rem' },
                      },
                      '& .Mui-selected': {
                        color: 'primary.dark',
                        backgroundColor: 'rgba(255,255,255,0.82)',
                        boxShadow: '0 4px 14px rgba(10, 68, 99, 0.12)',
                      },
                    }}
                  >
                    <Tab value="overview" label="Overview" />
                    <Tab value="explain" label="Explainability" />
                    <Tab value="history" label="Local Run History" />
                  </Tabs>
                </Box>

                <Box sx={{ pb: { xs: 1.25, sm: 1.45 }, pt: 0.35 }}>
                {viewTab === 'overview' ? (
                  <Stack spacing={{ xs: 1.65, sm: 1.95 }}>
                    <ForecastComparison data={result} hasScenario={hasScenario} />
                    {forecastMode === 'live' ? (
                      <ScenarioPanel
                        scenarios={scenariosQuery.data || []}
                        loading={runMutation.isPending}
                        canRunForecast={coordinatesValid}
                        onApply={applyScenario}
                      />
                    ) : (
                      <CustomWhatIfPanel
                        values={customDraft}
                        touched={customTouched}
                        errors={customErrors}
                        impactMode={customImpactMode}
                        onImpactModeChange={setCustomImpactMode}
                        groupPresets={CUSTOM_GROUP_PRESETS}
                        onApplyGroupPreset={applyCustomGroupPreset}
                        impactPreviewEstimate={customImpactPreviewEstimate}
                        runImpactPreview={customRunImpactPreview}
                        onValueChange={(key, value) => setCustomDraft((s) => ({ ...s, [key]: value }))}
                        onFieldBlur={(key) => setCustomTouched((s) => ({ ...s, [key]: true }))}
                        onRun={() => runCustomWhatIf()}
                        onClear={clearCustomInputs}
                        loading={runMutation.isPending}
                        canRunForecast={coordinatesValid}
                        allValid={customAllValid}
                        hasAnyValue={customHasAnyValue}
                        baselineSource={result?.meta?.baseline_source || 'live_api'}
                        liveUnavailable={liveFetchUnavailable}
                        appliedOverrides={customAppliedOverrides}
                      />
                    )}
                  </Stack>
                ) : null}

                {viewTab === 'explain' ? (
                  <Box className="explainability-stage-shell">
                    <ExplainabilityPanel data={result} modelInfo={modelInfoQuery.data} hasScenario={hasScenario} />
                  </Box>
                ) : null}

                {viewTab === 'history' ? (
                  <Stack spacing={1.35}>
                    <Alert severity="info">
                      Local history is stored in this browser/device and is separate from backend system run logs.
                    </Alert>
                    <RunHistoryPanel
                      runs={localRuns}
                      onLoadRun={loadRun}
                      onRerun={rerunFromHistory}
                      onDelete={deleteLocalRun}
                      onClearAll={clearLocalRuns}
                      loading={runMutation.isPending || Boolean(loadingRunId)}
                      loadingRunId={loadingRunId}
                      activeRunId={activeRunId}
                    />
                  </Stack>
                ) : null}
                </Box>
            </Box>

            <Box className="studio-right-rail" sx={{ ...workspaceRightColSx }}>
                <Stack className="side-rail-shell" spacing={1.45} sx={{ minWidth: 0 }}>
                  <TrustPanel health={result?.health} meta={result?.meta} scenario={result?.scenario} />

                  <Divider sx={{ borderColor: sk.divider }} />

                  <Box className="rail-section" sx={{ py: 0.25 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 800, color: sk.ink.title }}>
                      Run context and export
                    </Typography>
                    <Stack direction="row" spacing={0.65} useFlexGap flexWrap="wrap" sx={{ mt: 0.65 }}>
                      <Chip size="small" label={result ? `Generated ${formatWhen(result?.meta?.generated_at)}` : 'No run yet'} variant="outlined" />
                      {result ? (
                        <Chip size="small" label={result.run.persisted ? `Saved ${result.run.run_id}` : 'Backend run not saved'} variant="outlined" color={result.run.persisted ? 'success' : 'warning'} />
                      ) : null}
                    </Stack>

                    <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: sk.ink.body }}>
                      <strong>Mode summary:</strong> {activeRunLabel}
                    </Typography>
                    {result?.meta?.mode_note ? (
                      <Typography variant="caption" sx={{ display: 'block', mt: 0.24, color: sk.ink.body }}>
                        {result.meta.mode_note}
                      </Typography>
                    ) : null}
                    <Typography variant="caption" sx={{ display: 'block', mt: 0.22, color: sk.ink.body }}>
                      {(result?.meta?.forecast_mode || forecastMode || 'live')} mode · baseline source:{' '}
                      {result?.meta?.baseline_source || 'live_api'}
                    </Typography>

                    <Stack direction="row" spacing={0.75} sx={{ mt: 0.85, flexWrap: 'wrap' }}>
                      <Button size="small" variant="contained" onClick={exportCsv} disabled={!result}>
                        Export CSV
                      </Button>
                      <Button size="small" variant="outlined" onClick={exportPdf} disabled={!result}>
                        Export PDF
                      </Button>
                    </Stack>

                    <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: sk.ink.body }}>
                      Local run history in this app is browser-only and separate from backend system logs.
                    </Typography>
                    {result?.run?.persisted === false ? (
                      <Typography variant="caption" sx={{ display: 'block', mt: 0.3, color: '#8d2e2e', fontWeight: 700 }}>
                        Forecast output is valid, but this run was not written to backend persistence.
                      </Typography>
                    ) : null}
                  </Box>

                  <Divider sx={{ borderColor: sk.divider }} />
                  <Box className="method-caveat-rail">
                    <Box component="details">
                      <Box component="summary" sx={{ cursor: 'pointer', fontWeight: 800, color: sk.ink.title }}>
                        Methodology caveats
                      </Box>
                      <Stack spacing={0.75} sx={{ mt: 0.75 }}>
                        <Typography variant="caption" className="rail-caveat-item">
                          Runtime uses a 72h history window by default. Weekly lag-derived features may be imputed when &lt;168h history is available; Prediction Health reports imputation context.
                        </Typography>
                        <Typography variant="caption" className="rail-caveat-item">
                          Model training source is Haikou-stage data. Non-Haikou cities are provided for exploratory demo checks only and should not be presented as externally validated generalization.
                        </Typography>
                        <Typography variant="caption" className="rail-caveat-item">
                          Demo city list is intentionally small: Haikou (training-aligned), Colombo (user-relevant), and selected Europe cities where Open-Meteo air-quality coverage is typically stronger.
                        </Typography>
                        <Typography variant="caption" className="rail-caveat-item">
                          Reliability guidance and uncertainty bands are decision-support diagnostics. They are not guarantees and should not be read as probabilistic certainty.
                        </Typography>
                      </Stack>
                    </Box>
                  </Box>

                  <Divider sx={{ borderColor: sk.divider }} />
                  <QuickGuideCard forecastMode={forecastMode} embedded />
                </Stack>
            </Box>
          </Box>
        </Stack>
      </Box>
    </Box>
  )
}
