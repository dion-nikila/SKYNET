import { labelFeature, labelScenario } from './labels.js'

function num(v, fallback = null) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function text(v) {
  if (v === null || v === undefined) return ''
  return String(v)
}

function escapeCsvCell(v) {
  const raw = text(v)
  if (raw.includes('"') || raw.includes(',') || raw.includes('\n')) {
    return `"${raw.replaceAll('"', '""')}"`
  }
  return raw
}

function toIsoNow() {
  return new Date().toISOString()
}

function mapTopDrivers(rows, limit = 5) {
  if (!Array.isArray(rows) || !rows.length) return ''
  return rows
    .slice(0, limit)
    .map((r) => {
      const name = r?.feature_label || labelFeature(r?.feature || '')
      const shap = num(r?.shap, 0)
      return `${name} (${shap >= 0 ? '+' : ''}${shap.toFixed(3)})`
    })
    .join('; ')
}

function mapDeltaChanges(rows, limit = 5) {
  if (!Array.isArray(rows) || !rows.length) return ''
  return rows
    .slice(0, limit)
    .map((r) => {
      const name = r?.feature_label || labelFeature(r?.feature || '')
      const delta = num(r?.delta_shap, 0)
      return `${name} (${delta >= 0 ? '+' : ''}${delta.toFixed(3)})`
    })
    .join('; ')
}

function mapOverrides(overrides) {
  if (!Array.isArray(overrides) || !overrides.length) return ''
  return overrides
    .map((o) => {
      const f = o?.feature || ''
      const scale = f === 'pressure' ? 10 : 1
      const from = num(o?.from, null)
      const to = num(o?.to, null)
      const fromScaled = from === null ? null : from * scale
      const toScaled = to === null ? null : to * scale
      const fromText = fromScaled === null ? text(o?.from) : fromScaled.toFixed(3)
      const toText = toScaled === null ? text(o?.to) : toScaled.toFixed(3)
      const unit = f === 'pressure' ? ' hPa' : ''
      const limitTag = o?.direction_limited ? ' [direction-limited]' : ''
      return `${f}: ${fromText}${unit} -> ${toText}${unit}${o?.clamped ? ' [clamped]' : ''}${limitTag}`
    })
    .join('; ')
}

function listOodFeatures(rows) {
  if (!Array.isArray(rows) || !rows.length) return ''
  return rows
    .map((r) => {
      const name = r?.feature_label || labelFeature(r?.feature || '')
      return `${name} (${r?.severity || 'soft'})`
    })
    .join('; ')
}

function scenarioRunMode(meta, scenario) {
  const explicitMode = String(scenario?.scenario_mode || '').toLowerCase()
  if (['macro', 'guided_intervention', 'manual_custom', 'baseline'].includes(explicitMode)) {
    return explicitMode
  }
  const forecastMode = String(meta?.forecast_mode || '').toLowerCase()
  const scenarioId = String(scenario?.scenario_id || '')
  const intensity = num(scenario?.intensity, 0)

  if (forecastMode === 'custom' || scenarioId === 'custom_what_if') return 'manual_custom'
  if (scenarioId === 'guided_intervention') return 'guided_intervention'
  if (scenarioId === 'baseline') return 'baseline'
  if (forecastMode === 'live' && scenarioId === 'custom' && intensity > 0) return 'guided_intervention'
  if (forecastMode === 'live' && scenarioId === 'custom' && intensity <= 0) return 'baseline'
  if (scenarioId) return 'macro'
  return 'baseline'
}

function baselineSourceLabel(source) {
  const key = text(source).trim().toLowerCase()
  if (key === 'live_api') return 'Live API baseline'
  if (key === 'reference_profile') return 'Reference dataset baseline'
  if (key === 'demo_default') return 'Demo default baseline'
  return key || 'Baseline context'
}

function baselineContextNote({ baselineSource, liveDataUsed, baselineTimestamp }) {
  const sourceLabel = baselineSourceLabel(baselineSource)
  const ts = text(baselineTimestamp)
  if (liveDataUsed) {
    return `Live observed baseline context from ${sourceLabel}${ts ? ` at ${ts}` : ''}.`
  }
  return `Non-live baseline context from ${sourceLabel}${ts ? ` at ${ts}` : ''}; treat absolute PM2.5 level as fallback/reference context, not a live observed snapshot.`
}

export function buildForecastExportRecord(result) {
  const meta = result?.meta || {}
  const baseline = result?.baseline || {}
  const scenario = result?.scenario || {}
  const delta = result?.delta || {}
  const health = result?.health || {}
  const reliability = health?.reliability || {}
  const uncertainty = health?.uncertainty || {}
  const run = result?.run || {}

  const basePred = baseline?.prediction || {}
  const scenarioPred = scenario?.prediction || {}
  const history = health?.history || {}
  const imputation = health?.imputation || {}
  const fallback = health?.fallback || {}
  const ood = health?.ood || {}
  const scenarioId = text(scenario.scenario_id)
  const scenarioMode = scenarioRunMode(meta, scenario)
  const scenarioLabel = scenarioMode === 'guided_intervention'
    ? labelScenario('guided_intervention')
    : labelScenario(scenario.scenario_id || '')
  const includeIntensity = scenarioMode === 'macro' || scenarioMode === 'guided_intervention'
  const scenarioIntensityNote = scenarioMode === 'guided_intervention'
    ? 'Guided mode uses a fixed baseline intensity; per-row strength choices are the primary control.'
    : ''

  return {
    export_timestamp_utc: toIsoNow(),
    generated_at: text(meta.generated_at),
    request_id: text(meta.request_id),
    run_id: text(run.run_id),
    run_persisted: Boolean(run.persisted),
    forecast_mode: text(meta.forecast_mode),
    baseline_source: text(meta.baseline_source),
    baseline_source_label: baselineSourceLabel(meta.baseline_source),
    baseline_timestamp: text(meta.baseline_timestamp),
    baseline_context_note: baselineContextNote({
      baselineSource: meta.baseline_source,
      liveDataUsed: Boolean(meta.live_data_used),
      baselineTimestamp: meta.baseline_timestamp,
    }),
    live_data_used: Boolean(meta.live_data_used),
    overrides_applied: Boolean(meta.overrides_applied),
    mode_note: text(meta.mode_note),
    location_id: text(meta.location_id),
    location_name: text(meta.location_name),
    location_lat: num(meta.location_lat, ''),
    location_lon: num(meta.location_lon, ''),
    scenario_mode: scenarioMode,
    scenario_id: scenarioId,
    scenario_label: scenarioLabel,
    scenario_intensity: includeIntensity ? num(scenario.intensity, '') : '',
    scenario_intensity_note: scenarioIntensityNote,
    baseline_pm25_now: num(baseline?.inputs_snapshot?.pm25_current, ''),
    baseline_pred_pm25_t_plus_1: num(basePred.pm25_t_plus_1, ''),
    baseline_pred_delta_pm25: num(basePred.delta_pm25_t_plus_1, ''),
    scenario_pred_pm25_t_plus_1: num(scenarioPred.pm25_t_plus_1, ''),
    scenario_pred_delta_pm25: num(scenarioPred.delta_pm25_t_plus_1, ''),
    pm25_change_vs_baseline: num(delta.pm25_change, ''),
    quality_score: num(health.quality_score, ''),
    quality_label: text(health.quality_label),
    reliability_method: text(reliability.method),
    reliability_score: num(reliability.score, ''),
    reliability_label: text(reliability.label),
    ood_flag: Boolean(ood.flag),
    ood_score: num(ood.score, ''),
    ood_soft_count: num(ood.soft_count, ''),
    ood_hard_count: num(ood.hard_count, ''),
    ood_features_exceeded: listOodFeatures(ood.features_exceeded),
    history_target_hours: num(history.target_hours, ''),
    history_available_hours: num(history.available_hours, ''),
    history_used_hours: num(history.used_hours, ''),
    history_coverage_ratio: num(history.coverage_ratio, ''),
    imputed_features: num(imputation.imputed_features, ''),
    total_features: num(imputation.total_features, ''),
    imputation_ratio: num(imputation.ratio, ''),
    fallback_level: num(fallback.level, ''),
    fallback_label: text(fallback.label),
    fallback_notes: text(fallback.notes),
    uncertainty_method: text(uncertainty.method),
    uncertainty_available: Boolean(uncertainty.available),
    uncertainty_note: text(uncertainty.note),
    baseline_band_90_low: num((uncertainty.baseline_bands || []).find((x) => Number(x?.coverage_pct) === 90)?.lower, ''),
    baseline_band_90_high: num((uncertainty.baseline_bands || []).find((x) => Number(x?.coverage_pct) === 90)?.upper, ''),
    scenario_band_90_low: num((uncertainty.scenario_bands || []).find((x) => Number(x?.coverage_pct) === 90)?.lower, ''),
    scenario_band_90_high: num((uncertainty.scenario_bands || []).find((x) => Number(x?.coverage_pct) === 90)?.upper, ''),
    baseline_summary_text: text(baseline?.shap?.summary_text),
    scenario_summary_text: text(scenario?.shap?.summary_text),
    delta_summary_text: text(delta?.delta_shap?.summary_text),
    baseline_top_drivers: mapTopDrivers(baseline?.shap?.top_drivers),
    scenario_top_drivers: mapTopDrivers(scenario?.shap?.top_drivers),
    delta_top_reason_shifts: mapDeltaChanges(delta?.delta_shap?.top_changes),
    applied_overrides: mapOverrides(scenario.applied_overrides),
    baseline_plain_language: text((baseline?.shap?.plain_language || []).join(' | ')),
    scenario_plain_language: text((scenario?.shap?.plain_language || []).join(' | ')),
    delta_plain_language: text((delta?.delta_shap?.plain_language || []).join(' | ')),
  }
}

export function buildForecastCsv(result) {
  const row = buildForecastExportRecord(result)
  const headers = Object.keys(row)
  const line = headers.map((h) => escapeCsvCell(row[h])).join(',')
  return `${headers.join(',')}\n${line}\n`
}

export function downloadForecastCsv(result, filenamePrefix = 'skynet_forecast_export') {
  const csv = buildForecastCsv(result)
  const stamp = new Date().toISOString().replaceAll(':', '').replace(/\..+$/, 'Z')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${filenamePrefix}_${stamp}.csv`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function escapeHtml(v) {
  return text(v)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function formatNum(value, digits = 2) {
  const n = num(value, null)
  if (n === null) return 'N/A'
  return n.toFixed(digits)
}

function metricCard(label, value, sub = '') {
  return `
    <div class="metric-card">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
      ${sub ? `<div class="metric-sub">${escapeHtml(sub)}</div>` : ''}
    </div>
  `
}

function buildForecastReportHtml(result) {
  const row = buildForecastExportRecord(result)
  const runLabel = row.run_id || 'not-saved'
  const scenarioDetail = row.scenario_mode === 'manual_custom'
    ? `${row.scenario_label || row.scenario_id} (manual custom overrides)`
    : row.scenario_mode === 'guided_intervention'
      ? `${row.scenario_label || row.scenario_id} (guided intervention, fixed intensity ${row.scenario_intensity})`
      : row.scenario_mode === 'baseline'
        ? `${row.scenario_label || row.scenario_id} (baseline forecast)`
        : `${row.scenario_label || row.scenario_id} (intensity ${row.scenario_intensity})`
  const detailRows = [
    ['Run ID', runLabel],
    ['Generated at', row.generated_at],
    ['Exported at', row.export_timestamp_utc],
    ['Mode', row.forecast_mode],
    ['Baseline source', row.baseline_source_label || row.baseline_source],
    ['Baseline timestamp', row.baseline_timestamp || 'N/A'],
    ['Live data used', String(row.live_data_used)],
    ['Baseline context note', row.baseline_context_note || 'N/A'],
    ['Location', `${row.location_name || 'N/A'} (${row.location_lat || 'N/A'}, ${row.location_lon || 'N/A'})`],
    ['Scenario', scenarioDetail],
    ['Scenario intensity note', row.scenario_intensity_note || 'N/A'],
    ['Run quality', `${row.quality_label} (${formatNum(row.quality_score, 3)})`],
    ['Reliability guidance', `${row.reliability_label || row.quality_label || 'N/A'} (${formatNum(row.reliability_score, 3)})`],
    [
      'Uncertainty guidance',
      row.uncertainty_available
        ? `Empirical residual bands (90%): baseline [${formatNum(row.baseline_band_90_low, 1)}, ${formatNum(row.baseline_band_90_high, 1)}], scenario [${formatNum(row.scenario_band_90_low, 1)}, ${formatNum(row.scenario_band_90_high, 1)}]`
        : (row.uncertainty_note || 'Unavailable'),
    ],
    ['OOD', `${row.ood_flag ? 'Flagged' : 'Clear'} | score=${formatNum(row.ood_score, 3)} | soft=${row.ood_soft_count} hard=${row.ood_hard_count}`],
  ]
  const detailTable = detailRows
    .map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`)
    .join('')

  return `
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SKYNET Forecast Export (${escapeHtml(runLabel)})</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; color: #0f172a; }
      .header { margin-bottom: 18px; }
      h1 { margin: 0 0 4px 0; font-size: 22px; color: #0b3552; }
      p { margin: 0; color: #334155; font-size: 13px; }
      .section-title { font-size: 14px; font-weight: 700; color: #0b3552; margin: 18px 0 8px; text-transform: uppercase; letter-spacing: 0.03em; }
      .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
      .metric-card { border: 1px solid #dbe3ea; border-radius: 10px; padding: 10px; background: #f8fbff; }
      .metric-label { font-size: 11px; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.03em; }
      .metric-value { font-size: 22px; font-weight: 800; color: #0b3552; margin-top: 4px; }
      .metric-sub { font-size: 11px; color: #475569; margin-top: 3px; }
      table { width: 100%; border-collapse: collapse; font-size: 12px; }
      th, td { border: 1px solid #e2e8f0; padding: 7px 8px; vertical-align: top; }
      th { background: #f8fafc; text-align: left; width: 32%; font-weight: 700; color: #1e293b; }
      .box { border: 1px solid #dbe3ea; border-radius: 10px; padding: 10px; background: #fff; font-size: 12px; color: #1e293b; line-height: 1.45; white-space: pre-wrap; }
      .notice { border: 1px solid #fed7aa; border-radius: 10px; padding: 10px; background: #fff7ed; color: #9a3412; font-size: 12px; line-height: 1.45; margin-top: 12px; }
      @media print {
        body { margin: 11mm; }
        .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      }
    </style>
  </head>
  <body>
    <div class="header">
      <h1>SKYNET Forecast Result Export</h1>
      <p>Structured snapshot for reporting and review.</p>
      ${row.live_data_used ? '' : `<div class="notice">${escapeHtml(row.baseline_context_note)}</div>`}
    </div>

    <div class="section-title">Key Forecast Metrics</div>
    <div class="grid">
      ${metricCard('Current PM2.5', `${formatNum(row.baseline_pm25_now, 2)} µg/m³`, row.live_data_used ? 'Current observed value' : 'Fallback/reference baseline value')}
      ${metricCard('Baseline Next-Hour PM2.5', `${formatNum(row.baseline_pred_pm25_t_plus_1, 2)} µg/m³`, 'Reference prediction')}
      ${metricCard('Scenario Next-Hour PM2.5', `${formatNum(row.scenario_pred_pm25_t_plus_1, 2)} µg/m³`, 'Scenario/custom prediction')}
    </div>
    <div class="grid" style="margin-top:10px;">
      ${metricCard('Change vs Baseline', `${formatNum(row.pm25_change_vs_baseline, 3)} µg/m³`, 'Scenario - baseline')}
      ${metricCard('Quality Score', formatNum(row.quality_score, 3), row.quality_label || '')}
      ${metricCard('OOD Status', row.ood_flag ? 'Flagged' : 'Clear', `score ${formatNum(row.ood_score, 3)}`)}
    </div>

    <div class="section-title">Run Details</div>
    <table><tbody>${detailTable}</tbody></table>

    <div class="section-title">Explainability Summary</div>
    <div class="box"><strong>Baseline:</strong> ${escapeHtml(row.baseline_summary_text || 'N/A')}</div>
    <div class="box" style="margin-top:8px;"><strong>Scenario:</strong> ${escapeHtml(row.scenario_summary_text || 'N/A')}</div>
    <div class="box" style="margin-top:8px;"><strong>Delta Reasoning:</strong> ${escapeHtml(row.delta_summary_text || 'N/A')}</div>

    <div class="section-title">Top Drivers</div>
    <div class="box"><strong>Baseline top drivers:</strong> ${escapeHtml(row.baseline_top_drivers || 'N/A')}</div>
    <div class="box" style="margin-top:8px;"><strong>Scenario top drivers:</strong> ${escapeHtml(row.scenario_top_drivers || 'N/A')}</div>
    <div class="box" style="margin-top:8px;"><strong>Top reason shifts:</strong> ${escapeHtml(row.delta_top_reason_shifts || 'N/A')}</div>

    <div class="section-title">Applied Overrides</div>
    <div class="box">${escapeHtml(row.applied_overrides || 'None')}</div>
  </body>
</html>`
}

export function openForecastPrintReport(result) {
  const html = buildForecastReportHtml(result)
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)

  const iframe = document.createElement('iframe')
  iframe.style.position = 'fixed'
  iframe.style.right = '0'
  iframe.style.bottom = '0'
  iframe.style.width = '0'
  iframe.style.height = '0'
  iframe.style.border = '0'
  iframe.src = url
  document.body.appendChild(iframe)

  const cleanup = () => {
    try {
      document.body.removeChild(iframe)
    } catch (_ignore) {
      // no-op
    }
    URL.revokeObjectURL(url)
  }

  const timeout = window.setTimeout(() => {
    cleanup()
    // Best-effort print flow; avoid async throw that bypasses caller error handlers.
  }, 12000)

  iframe.onload = () => {
    window.clearTimeout(timeout)
    const w = iframe.contentWindow
    if (!w) {
      cleanup()
      return
    }
    w.focus()
    window.setTimeout(() => {
      w.print()
      window.setTimeout(cleanup, 1200)
    }, 120)
  }
}
