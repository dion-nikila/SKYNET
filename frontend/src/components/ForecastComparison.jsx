import React from 'react'
import {
  Typography,
  Alert,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Stack,
  Box,
} from '@mui/material'
import { labelFeature } from '../utils/labels'
import { sk } from '../theme/tokens'

function num(v, fallback = 0) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function displayOverrideValue(feature, value) {
  const n = num(value, 0)
  if (String(feature) === 'pressure') return n * 10
  return n
}

function overrideFrom(item) {
  if (item == null || typeof item !== 'object') return 0
  return displayOverrideValue(item.feature, item.from ?? item.from_value)
}

function overrideTo(item) {
  if (item == null || typeof item !== 'object') return 0
  return displayOverrideValue(item.feature, item.to ?? item.to_value)
}

function asDeltaRows(overrides) {
  return overrides
    .map((r) => {
      const from = overrideFrom(r)
      const to = overrideTo(r)
      const diff = to - from
      const pct = from === 0 ? null : (diff / from) * 100
      return { ...r, from, to, diff, pct }
    })
    .sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff))
}

function toneForDelta(delta) {
  if (delta > 0) return { chip: 'warning', color: '#9a2f2f', label: 'Higher risk' }
  if (delta < 0) return { chip: 'success', color: '#1f7a58', label: 'Lower risk' }
  return { chip: 'default', color: '#18435f', label: 'No clear change' }
}

const SCENARIO_INTENT = {
  traffic_gridlock: 'up',
  strong_dispersion: 'down',
  heatwave: 'up',
  dust_resuspension: 'up',
  trapped_pollution: 'up',
  industrial_plume: 'up',
}

function plainImpactSentence(delta, pct) {
  if (delta > 0) return `Model suggests air quality may worsen next hour (${delta.toFixed(1)} µg/m³ above baseline, ${Math.abs(pct).toFixed(1)}%).`
  if (delta < 0) return `Model suggests air quality may improve next hour (${Math.abs(delta).toFixed(1)} µg/m³ below baseline, ${Math.abs(pct).toFixed(1)}%).`
  return 'Model suggests little to no change versus baseline in this run.'
}

function driverRows(rows) {
  const top = rows.slice(0, 3)
  if (!top.length) return []
  return top.map((r) => {
    const trendWord = r.diff >= 0 ? 'increased' : 'decreased'
    const byText = r.pct === null
      ? `${Math.abs(r.diff).toFixed(1)} units`
      : `${Math.abs(r.pct).toFixed(1)}%`
    return {
      key: `${r.feature}-${r.diff}`,
      text: `${labelFeature(r.feature)} ${trendWord} by ${byText}.`,
      clamped: Boolean(r.clamped),
      directionLimited: Boolean(r.direction_limited),
    }
  })
}

export default function ForecastComparison({ data, hasScenario = false }) {
  if (!data) {
    return (
      <Box sx={{ py: { xs: 1.1, sm: 1.25 } }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title }}>
          What Changed
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.6, color: sk.ink.muted }}>
          Simple comparison of baseline and scenario impact.
        </Typography>
        <Typography variant="body2" sx={{ color: sk.ink.muted }}>No run yet.</Typography>
      </Box>
    )
  }

  const baseline = num(data?.baseline?.prediction?.pm25_t_plus_1)
  const scenarioPredRaw = num(data?.scenario?.prediction?.pm25_t_plus_1, baseline)
  const scenario = hasScenario ? scenarioPredRaw : baseline
  const deltaRaw = num(data?.delta?.pm25_change, scenario - baseline)
  const delta = hasScenario ? deltaRaw : 0
  const deltaPct = baseline === 0 ? 0 : (delta / baseline) * 100
  const deltaTone = toneForDelta(delta)
  const scenarioId = String(data?.scenario?.scenario_id || '')
  const forecastMode = String(data?.meta?.forecast_mode || '').toLowerCase()
  const scenarioIntensity = num(data?.scenario?.intensity, 0)
  const explicitMode = String(data?.scenario?.scenario_mode || '').toLowerCase()
  const isManualCustom = explicitMode === 'manual_custom' || forecastMode === 'custom' || scenarioId === 'custom_what_if'
  const isGuidedIntervention = explicitMode === 'guided_intervention' || scenarioId === 'guided_intervention' || (forecastMode === 'live' && scenarioId === 'custom' && scenarioIntensity > 0)
  const intentDirection = SCENARIO_INTENT[scenarioId] || null
  const resultDirection = delta > 0 ? 'up' : delta < 0 ? 'down' : 'flat'
  const intentMismatch = Boolean(
    hasScenario &&
    intentDirection &&
    resultDirection !== 'flat' &&
    intentDirection !== resultDirection
  )

  const overrides = Array.isArray(data?.scenario?.applied_overrides)
    ? asDeltaRows(data.scenario.applied_overrides)
    : []
  const drivers = driverRows(overrides)
  const directionLimitedCount = overrides.filter((r) => Boolean(r.direction_limited)).length

  const roundedLooksEqual = hasScenario && baseline.toFixed(1) === scenario.toFixed(1) && Math.abs(delta) >= 0.01

  return (
    <Box sx={{ py: { xs: 1.15, sm: 1.35 } }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title }}>
          What Changed
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.55, color: sk.ink.muted }}>
          {hasScenario ? 'Plain-language summary of scenario impact.' : 'Apply a scenario to compare against baseline.'}
        </Typography>

        {hasScenario ? (
          <Alert severity={delta > 0 ? 'warning' : delta < 0 ? 'success' : 'info'} sx={{ mb: 1 }}>
            {plainImpactSentence(delta, deltaPct)}
            <Typography variant="caption" sx={{ display: 'block', mt: 0.45, color: 'text.secondary' }}>
              {isManualCustom
                ? 'Manual custom overrides set direct current-condition targets. Final direction still comes from full model evaluation under current conditions.'
                : isGuidedIntervention
                  ? 'Guided interventions encode directional intent. Final direction always comes from full model evaluation under current conditions.'
                  : 'Scenario templates describe typical intent. Final direction always comes from full model evaluation under current conditions.'}
            </Typography>
            {intentMismatch ? (
              <Typography variant="caption" sx={{ display: 'block', mt: 0.25, color: '#7b4a00' }}>
                This run moved opposite to the scenario&apos;s typical intent, which can happen under different baseline states.
              </Typography>
            ) : null}
            {directionLimitedCount > 0 ? (
              <Typography variant="caption" sx={{ display: 'block', mt: 0.25, color: '#7b4a00' }}>
                {directionLimitedCount} intervention value(s) were direction-limited by plausibility bounds.
              </Typography>
            ) : null}
          </Alert>
        ) : (
          <Alert severity="info" sx={{ mb: 1.2 }}>
            Baseline is active. Apply a scenario to see what changes.
          </Alert>
        )}

        {hasScenario ? (
          <Stack direction="row" spacing={0.8} alignItems="center" sx={{ mb: 0.95, flexWrap: 'wrap' }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Net effect
            </Typography>
            <Chip
              size="small"
              color={deltaTone.chip}
              label={`${deltaTone.label}: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)} µg/m³`}
            />
            <Chip
              size="small"
              variant="outlined"
              label={isManualCustom ? 'Manual custom run' : isGuidedIntervention ? 'Guided intervention run' : 'Macro scenario run'}
            />
            {roundedLooksEqual ? (
              <Typography variant="caption" color="text.secondary">
                Rounded values can look equal while a small non-zero change still exists.
              </Typography>
            ) : null}
          </Stack>
        ) : null}

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.95 }}>
          Forecast values are shown in the top forecast section. This view focuses on what drove the change.
        </Typography>

        {hasScenario ? (
          <>
            <Box sx={{ mb: 0.95, py: 0.62, borderTop: `1px solid ${sk.divider}`, borderBottom: `1px solid ${sk.divider}` }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.45 }}>
                  Why this changed
                </Typography>
                {drivers.length > 0 ? (
                  <Stack spacing={0.45}>
                    {drivers.map((d) => (
                      <Stack key={d.key} direction="row" spacing={0.8} alignItems="center" sx={{ flexWrap: 'wrap' }}>
                        <Typography sx={{ color: '#14506f', fontWeight: 900, lineHeight: 1 }}>{d.text.includes('increased') ? '▲' : '▼'}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {d.text}
                        </Typography>
                        {d.clamped ? <Chip size="small" color="warning" label="clamped" /> : null}
                        {d.directionLimited ? <Chip size="small" color="warning" variant="outlined" label="direction-limited" /> : null}
                      </Stack>
                    ))}
                  </Stack>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No major variable override details are available for this run.
                  </Typography>
                )}
            </Box>

            {overrides.length > 0 ? (
              <Accordion
                disableGutters
                elevation={0}
                sx={{
                  border: sk.border,
                  borderRadius: 1.25,
                  backgroundColor: 'rgba(255,255,255,0.88)',
                  boxShadow: sk.surface.shadowSoft,
                  '&::before': { display: 'none' },
                }}
              >
                <AccordionSummary expandIcon={<Typography sx={{ fontSize: 14, opacity: 0.6 }}>▾</Typography>}>
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>
                    Technical details
                  </Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ pt: 0.2 }}>
                  <TableContainer sx={{ overflowX: 'auto' }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Variable</TableCell>
                          <TableCell>From</TableCell>
                          <TableCell>To</TableCell>
                          <TableCell>Δ</TableCell>
                          <TableCell>Δ%</TableCell>
                          <TableCell>Clamped</TableCell>
                          <TableCell>Direction-limited</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {overrides.map((r, idx) => (
                          <TableRow key={`d-${idx}`}>
                            <TableCell>{labelFeature(r.feature)}</TableCell>
                            <TableCell>{r.from.toFixed(2)}</TableCell>
                            <TableCell>{r.to.toFixed(2)}</TableCell>
                            <TableCell>{r.diff >= 0 ? '+' : ''}{r.diff.toFixed(2)}</TableCell>
                            <TableCell>{r.pct === null ? '-' : `${r.pct >= 0 ? '+' : ''}${r.pct.toFixed(1)}%`}</TableCell>
                            <TableCell>{r.clamped ? 'Yes' : 'No'}</TableCell>
                            <TableCell>{r.direction_limited ? 'Yes' : 'No'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.7 }}>
                    Pressure values in this table are displayed in hPa.
                  </Typography>
                </AccordionDetails>
              </Accordion>
            ) : null}
          </>
        ) : null}
    </Box>
  )
}
