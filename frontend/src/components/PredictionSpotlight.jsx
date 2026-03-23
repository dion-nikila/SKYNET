import React from 'react'
import { Typography, Grid, Chip, Box, Button, Stack } from '@mui/material'
import { sk } from '../theme/tokens'
import { labelScenario } from '../utils/labels'

function num(v, fallback = 0) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function pm25LevelInfo(v) {
  if (v <= 12) {
    return {
      band: 'Good',
      audienceLabel: 'Low pollution',
      tone: '#1f7752',
      note: 'Air is generally safe for outdoor activity for most people.',
      rangeNote: '0-12 µg/m³',
    }
  }
  if (v <= 35) {
    return {
      band: 'Moderate',
      audienceLabel: 'Mild pollution',
      tone: '#9a6a1f',
      note: 'Most people are okay; sensitive people may feel mild irritation.',
      rangeNote: '12.1-35 µg/m³',
    }
  }
  if (v <= 55) {
    return {
      band: 'Unhealthy for sensitive groups',
      audienceLabel: 'Sensitive groups at risk',
      tone: '#a55a1b',
      note: 'People with asthma, elderly adults, and children should reduce long exposure.',
      rangeNote: '35.1-55 µg/m³',
    }
  }
  if (v <= 150) {
    return {
      band: 'Unhealthy',
      audienceLabel: 'High pollution',
      tone: '#9d3131',
      note: 'Health effects are possible for many people; limit heavy outdoor activity.',
      rangeNote: '55.1-150 µg/m³',
    }
  }
  return {
    band: 'Very unhealthy',
    audienceLabel: 'Very high pollution',
    tone: '#7a214f',
    note: 'Air quality is poor for everyone; avoid prolonged outdoor exposure where possible.',
    rangeNote: '>150 µg/m³',
  }
}

function levelChipKind(levelInfo) {
  if (!levelInfo) return 'info'
  if (levelInfo.band === 'Good') return 'ok'
  if (levelInfo.band === 'Moderate') return 'warn'
  return 'danger'
}

function formatWhen(ts) {
  if (!ts) return 'Unknown'
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

function prettyScenario(id) {
  if (!id) return 'Scenario'
  const labeled = labelScenario(String(id))
  if (labeled && labeled !== String(id)) return labeled
  return String(id).replaceAll('_', ' ').replace(/\b\w/g, (m) => m.toUpperCase())
}

function deltaTone(delta) {
  if (delta > 0) return '#9d3131'
  if (delta < 0) return '#1f7752'
  return '#3d5c71'
}

function softChipSx(kind) {
  if (kind === 'info') {
    return { backgroundColor: 'rgba(219,242,255,0.16)', color: '#dcf4ff', border: '1px solid rgba(219,242,255,0.28)' }
  }
  if (kind === 'ok') {
    return { backgroundColor: 'rgba(84,214,149,0.2)', color: '#dcffec', border: '1px solid rgba(146,237,189,0.34)' }
  }
  if (kind === 'warn') {
    return { backgroundColor: 'rgba(255,176,109,0.18)', color: '#fff0d5', border: '1px solid rgba(255,210,160,0.3)' }
  }
  if (kind === 'danger') {
    return { backgroundColor: 'rgba(255,122,122,0.2)', color: '#ffe8e8', border: '1px solid rgba(255,182,182,0.34)' }
  }
  return { backgroundColor: 'rgba(219,242,255,0.16)', color: '#dcf4ff', border: '1px solid rgba(219,242,255,0.28)' }
}

function levelPillSx(levelInfo) {
  const band = String(levelInfo?.band || '')
  if (band === 'Good') {
    return { color: '#1e6346', border: '1px solid rgba(40,132,90,0.28)', backgroundColor: 'rgba(209,245,226,0.72)' }
  }
  if (band === 'Moderate') {
    return { color: '#8a5f19', border: '1px solid rgba(186,137,58,0.3)', backgroundColor: 'rgba(255,237,207,0.78)' }
  }
  if (band === 'Unhealthy for sensitive groups') {
    return { color: '#965718', border: '1px solid rgba(196,130,66,0.3)', backgroundColor: 'rgba(255,231,207,0.82)' }
  }
  if (band === 'Unhealthy') {
    return { color: '#8f2f2f', border: '1px solid rgba(184,90,90,0.3)', backgroundColor: 'rgba(255,224,224,0.82)' }
  }
  return { color: '#6d2350', border: '1px solid rgba(152,82,123,0.33)', backgroundColor: 'rgba(244,223,236,0.88)' }
}

function MetricCell({ label, value, subtitle, deltaText = '', delta = 0, divider = true, statusLabel = '', statusTone = '', statusNote = '' }) {
  return (
    <Box
      sx={{
        py: { xs: 0.95, sm: 1.05 },
        px: { xs: 0.95, sm: 1.15 },
        borderRight: divider ? `1px solid ${sk.divider}` : 'none',
      }}
    >
      <Typography variant="caption" sx={{ color: '#557489', fontWeight: 700, letterSpacing: '0.03em' }}>
        {label}
      </Typography>

      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.45, mt: 0.45 }}>
        <Typography
          variant="h3"
          sx={{
            color: '#0f3550',
            fontWeight: 900,
            lineHeight: 1.02,
            letterSpacing: '-0.02em',
            fontSize: { xs: '1.54rem', sm: '2.06rem' },
          }}
        >
          {value.toFixed(1)}
        </Typography>
        <Typography variant="caption" sx={{ color: '#49697e', fontWeight: 600 }}>
          µg/m³
        </Typography>
      </Box>

      <Typography variant="body2" sx={{ mt: 0.4, color: '#48687d' }}>
        {subtitle}
      </Typography>

      {statusLabel ? (
        <Typography variant="caption" sx={{ mt: 0.46, display: 'block', fontWeight: 700, color: statusTone || '#3d5c71' }}>
          Air-quality level: {statusLabel}
        </Typography>
      ) : null}
      {statusNote ? (
        <Typography variant="caption" sx={{ mt: 0.22, display: 'block', color: '#516f82', lineHeight: 1.35 }}>
          {statusNote}
        </Typography>
      ) : null}

      {deltaText ? (
        <Typography variant="caption" sx={{ mt: 0.6, display: 'block', fontWeight: 700, color: deltaTone(delta) }}>
          {deltaText}
        </Typography>
      ) : null}
    </Box>
  )
}

export default function PredictionSpotlight({
  data,
  hasScenario = false,
  onRefreshBaseline,
  loading = false,
  modelInfo = null,
}) {
  if (!data) {
    return (
      <Box sx={{ py: { xs: 1.25, sm: 1.4 }, borderBottom: `1px solid ${sk.divider}` }}>
          <Typography variant="overline" sx={{ color: sk.ink.muted }}>Forecast</Typography>
          <Typography variant="h5" sx={{ fontWeight: 900, mb: 0.45, color: sk.ink.title }}>Current vs Next-Hour PM2.5</Typography>
          <Typography variant="body2" sx={{ color: sk.ink.muted }}>
            Run a baseline forecast first, then apply scenarios to compare how the next-hour estimate shifts.
          </Typography>
      </Box>
    )
  }

  const baseline = num(data?.baseline?.prediction?.pm25_t_plus_1)
  const currentPm25 = num(data?.baseline?.inputs_snapshot?.pm25_current, baseline)
  const scenario = hasScenario
    ? num(data?.scenario?.prediction?.pm25_t_plus_1, baseline)
    : baseline

  const baselineDeltaFromNow = baseline - currentPm25
  const scenarioDeltaFromReference = scenario - baseline

  const currentLevel = pm25LevelInfo(currentPm25)
  const baselineLevel = pm25LevelInfo(baseline)
  const scenarioLevel = pm25LevelInfo(scenario)
  const referenceTime = formatWhen(data?.meta?.generated_at)
  const scenarioTitle = prettyScenario(data?.scenario?.scenario_id)
  const scenarioIntensity = num(data?.scenario?.intensity)
  const forecastMode = String(data?.meta?.forecast_mode || '')
  const scenarioId = String(data?.scenario?.scenario_id || '')
  const explicitMode = String(data?.scenario?.scenario_mode || '').toLowerCase()
  const isCustomScenario = explicitMode === 'manual_custom' || forecastMode === 'custom' || scenarioId === 'custom_what_if'
  const isGuidedIntervention = explicitMode === 'guided_intervention' || scenarioId === 'guided_intervention' || (forecastMode === 'live' && scenarioId === 'custom' && scenarioIntensity > 0)
  const scenarioContext = isCustomScenario ? 'Custom What-If' : scenarioTitle
  const metrics = modelInfo?.historical_test_metrics || null

  const hasMetrics =
    Number.isFinite(Number(metrics?.mae)) ||
    Number.isFinite(Number(metrics?.rmse)) ||
    Number.isFinite(Number(metrics?.r2))

  const metricText = [
    Number.isFinite(Number(metrics?.mae)) ? `MAE ${Number(metrics.mae).toFixed(3)}` : null,
    Number.isFinite(Number(metrics?.rmse)) ? `RMSE ${Number(metrics.rmse).toFixed(3)}` : null,
    Number.isFinite(Number(metrics?.r2)) ? `R² ${Number(metrics.r2).toFixed(3)}` : null,
  ].filter(Boolean).join(' · ')

  return (
    <Box
      className="prediction-spotlight-core"
      sx={{
        borderRadius: { xs: 0, sm: 2 },
        overflow: 'hidden',
        bgcolor: 'rgba(255,255,255,0.42)',
        borderBottom: `1px solid ${sk.divider}`,
      }}
    >
      <Box
        sx={{
          px: { xs: 1.15, sm: 1.45, md: 1.6 },
          py: { xs: 1.08, sm: 1.2 },
          borderBottom: `1px solid rgba(255,255,255,0.14)`,
          background: sk.spotlightBar,
        }}
      >
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={1.1}
          alignItems={{ xs: 'flex-start', sm: 'center' }}
          justifyContent="space-between"
        >
          <Box>
            <Typography variant="overline" sx={{ color: '#cbe8fb', fontWeight: 700, letterSpacing: '0.08em' }}>
              Forecast
            </Typography>
            <Typography
              variant="h5"
              sx={{
                color: '#eef8ff',
                fontWeight: 900,
                lineHeight: 1.08,
                letterSpacing: '-0.02em',
                fontSize: { xs: '1.22rem', sm: '1.74rem' },
              }}
            >
              Current vs Next-Hour PM2.5
            </Typography>
            <Typography variant="body2" sx={{ color: '#d8edf9', mt: 0.38, maxWidth: 760 }}>
              Baseline and scenario-adjusted estimates are shown together for fast directional interpretation.
            </Typography>
          </Box>

          <Stack direction="row" spacing={0.65} useFlexGap flexWrap="wrap" alignItems="center" justifyContent={{ xs: 'flex-start', sm: 'flex-end' }}>
            {hasScenario ? (
              <Chip
                size="small"
                variant="outlined"
                label={`Scenario level: ${scenarioLevel.band}`}
                sx={{ ...softChipSx(levelChipKind(scenarioLevel)), fontWeight: 700 }}
              />
            ) : null}
            <Button
              size="small"
              variant="outlined"
              onClick={onRefreshBaseline}
              disabled={loading || !onRefreshBaseline}
              sx={{
                flexShrink: 0,
                borderColor: 'rgba(255,255,255,0.45)',
                color: '#eef8ff',
                fontWeight: 700,
                '&:hover': { borderColor: 'rgba(255,255,255,0.65)', bgcolor: 'rgba(255,255,255,0.08)' },
              }}
            >
              Refresh baseline
            </Button>
          </Stack>
        </Stack>
      </Box>

      <Box sx={{ p: { xs: 1.2, sm: 1.75, md: 1.95 } }}>
        {hasScenario ? (
          <Box
            sx={{
              mb: 1.45,
              p: { xs: 0.85, sm: 1 },
              borderRadius: 1,
              borderLeft: '3px solid',
              borderLeftColor: scenarioDeltaFromReference > 0 ? 'warning.main' : scenarioDeltaFromReference < 0 ? 'success.main' : 'info.main',
              borderTop: 'none',
              borderRight: 'none',
              borderBottom: 'none',
              background:
                scenarioDeltaFromReference > 0
                  ? 'linear-gradient(180deg, rgba(255,232,232,0.9) 0%, rgba(255,255,255,0.98) 100%)'
                  : scenarioDeltaFromReference < 0
                    ? 'linear-gradient(180deg, rgba(226,255,239,0.85) 0%, rgba(255,255,255,0.98) 100%)'
                    : 'linear-gradient(180deg, rgba(237,247,255,0.92) 0%, rgba(255,255,255,0.98) 100%)',
            }}
          >
            <Stack direction="row" spacing={0.8} alignItems="center" useFlexGap flexWrap="wrap">
              <Typography variant="caption" sx={{ fontWeight: 800, color: '#35546c', letterSpacing: '0.04em' }}>
                Scenario effect
              </Typography>
              <Chip
                size="small"
                label={`${scenarioDeltaFromReference >= 0 ? '+' : ''}${scenarioDeltaFromReference.toFixed(1)} µg/m³ vs baseline`}
                color={scenarioDeltaFromReference > 0 ? 'warning' : scenarioDeltaFromReference < 0 ? 'success' : 'default'}
                sx={{ fontWeight: 700 }}
              />
              {!isCustomScenario ? (
                <Typography variant="caption" sx={{ color: '#4a627a', fontWeight: 700 }}>
                  {isGuidedIntervention
                    ? `Guided intervention · fixed intensity ${scenarioIntensity.toFixed(0)}/100`
                    : `Scenario intensity ${scenarioIntensity.toFixed(0)}/100`}
                </Typography>
              ) : null}
            </Stack>
            <Typography variant="body2" sx={{ mt: 0.45, color: '#3f6076' }}>
              {scenarioContext} run is currently {scenarioDeltaFromReference >= 0 ? 'above' : 'below'} baseline for the next-hour forecast.
            </Typography>
          </Box>
        ) : (
          <Box
            sx={{
              mb: 1.35,
              pl: 1.15,
              py: 0.75,
              borderLeft: '3px solid',
              borderLeftColor: 'info.main',
              bgcolor: 'rgba(13, 106, 148, 0.06)',
              borderRadius: '0 8px 8px 0',
            }}
          >
            <Typography variant="body2" sx={{ color: sk.ink.body, fontWeight: 600 }}>
              Baseline forecast is active. Apply a scenario to compare model-estimated changes.
            </Typography>
          </Box>
        )}

        <Grid container spacing={0}>
          <Grid item xs={12}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: hasScenario ? 'repeat(3, minmax(0, 1fr))' : 'repeat(2, minmax(0, 1fr))' },
                borderTop: `1px solid ${sk.divider}`,
                mt: 0.5,
                background: 'rgba(255,255,255,0.35)',
                borderRadius: 1.1,
              }}
            >
              <MetricCell
                label="Current PM2.5"
                value={currentPm25}
                subtitle="Observed concentration now"
                statusLabel={`${currentLevel.audienceLabel} (${currentLevel.band})`}
                statusTone={currentLevel.tone}
                statusNote={currentLevel.note}
                divider
              />
              <MetricCell
                label="Baseline forecast"
                value={baseline}
                subtitle="One-hour estimate without intervention"
                statusLabel={`${baselineLevel.audienceLabel} (${baselineLevel.band})`}
                statusTone={baselineLevel.tone}
                statusNote={baselineLevel.note}
                delta={baselineDeltaFromNow}
                deltaText={`Vs now: ${baselineDeltaFromNow >= 0 ? '+' : ''}${baselineDeltaFromNow.toFixed(1)} µg/m³`}
                divider={hasScenario}
              />
              {hasScenario ? (
                <MetricCell
                  label="Scenario-adjusted forecast"
                  value={scenario}
                  subtitle={isCustomScenario
                    ? 'Manual custom what-if result'
                    : isGuidedIntervention
                      ? `Guided intervention · fixed intensity ${scenarioIntensity.toFixed(0)}/100`
                      : `${scenarioTitle} · intensity ${scenarioIntensity.toFixed(0)}/100`}
                  statusLabel={`${scenarioLevel.audienceLabel} (${scenarioLevel.band})`}
                  statusTone={scenarioLevel.tone}
                  statusNote={scenarioLevel.note}
                  delta={scenarioDeltaFromReference}
                  deltaText={`Vs baseline: ${scenarioDeltaFromReference >= 0 ? '+' : ''}${scenarioDeltaFromReference.toFixed(1)} µg/m³`}
                  divider={false}
                />
              ) : null}
            </Box>
          </Grid>
        </Grid>

        <Box
          sx={{
            mt: 1.05,
            p: { xs: 0.95, sm: 1.05 },
            borderRadius: 1,
            borderLeft: '3px solid rgba(43, 112, 147, 0.42)',
            background: 'linear-gradient(180deg, rgba(247,252,255,0.95) 0%, rgba(239,248,253,0.83) 100%)',
          }}
        >
          <Typography variant="caption" sx={{ fontWeight: 800, color: '#244f68', letterSpacing: '0.02em' }}>
            Plain-language interpretation
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', mt: 0.42, color: '#2f556f', lineHeight: 1.44 }}>
            Current air quality is <strong>{currentLevel.audienceLabel.toLowerCase()}</strong> ({currentLevel.rangeNote}). The next-hour baseline forecast indicates <strong>{baselineLevel.audienceLabel.toLowerCase()}</strong> ({baselineLevel.rangeNote}).
            {hasScenario
              ? ` Under the selected scenario, the next-hour level is ${scenarioLevel.audienceLabel.toLowerCase()} (${scenarioLevel.rangeNote}).`
              : ''}
          </Typography>
          <Stack
            direction="row"
            spacing={0.6}
            useFlexGap
            flexWrap="wrap"
            sx={{ mt: 0.62 }}
          >
            <Chip
              size="small"
              label={`Now: ${currentLevel.band}`}
              sx={{ ...levelPillSx(currentLevel), fontWeight: 800 }}
            />
            <Chip
              size="small"
              label={`Baseline: ${baselineLevel.band}`}
              sx={{ ...levelPillSx(baselineLevel), fontWeight: 800 }}
            />
            {hasScenario ? (
              <Chip
                size="small"
                label={`Scenario: ${scenarioLevel.band}`}
                sx={{ ...levelPillSx(scenarioLevel), fontWeight: 800 }}
              />
            ) : null}
          </Stack>
          <Typography variant="caption" sx={{ mt: 0.52, display: 'block', color: '#3d5f74', lineHeight: 1.42 }}>
            PM2.5 guide (µg/m³): Good 0-12 · Moderate 12.1-35 · Sensitive groups at risk 35.1-55 · Unhealthy 55.1-150 · Very unhealthy &gt;150.
          </Typography>
        </Box>

        <Stack spacing={0.65} sx={{ mt: 1.35 }}>
          <Typography variant="caption" sx={{ color: sk.ink.muted, lineHeight: 1.45 }}>
            Reference updated: {referenceTime}
          </Typography>
          {hasMetrics ? (
            <Typography variant="caption" sx={{ color: sk.ink.muted, lineHeight: 1.45 }}>
              Historical model performance: {metricText}. These values summarize past test behavior and are not run-specific error guarantees.
            </Typography>
          ) : null}
          {Array.isArray(data?.health?.uncertainty?.baseline_bands) && data.health.uncertainty.baseline_bands.length ? (
            <Typography variant="caption" sx={{ color: sk.ink.muted, lineHeight: 1.45 }}>
              Empirical uncertainty guidance: baseline {data.health.uncertainty.baseline_bands[0].coverage_pct}% residual band {' '}
              [{Number(data.health.uncertainty.baseline_bands[0].lower).toFixed(1)}, {Number(data.health.uncertainty.baseline_bands[0].upper).toFixed(1)}] µg/m³.
            </Typography>
          ) : null}
          <Typography variant="caption" sx={{ color: sk.ink.muted, lineHeight: 1.45, pt: 0.25 }}>
            Refresh baseline from the forecast header when conditions change.
          </Typography>
        </Stack>
      </Box>
    </Box>
  )
}
