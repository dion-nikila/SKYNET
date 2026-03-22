import React from 'react'
import {
  Alert,
  Box,
  Button,
  Grid,
  TextField,
  Typography,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Chip,
  ToggleButtonGroup,
  ToggleButton,
  Stack,
} from '@mui/material'
import { labelFeature } from '../utils/labels'
import { sk } from '../theme/tokens'

const FIELD_CONFIG = [
  { key: 'PM10', label: 'PM10', helper: '0 to 1000 µg/m³' },
  { key: 'NO2', label: 'NO2', helper: '0 to 500 µg/m³' },
  { key: 'CO', label: 'CO', helper: '0 to 50 mg/m³' },
  { key: 'temperature', label: 'Temperature', helper: '-50 to 60 °C' },
  { key: 'humidity', label: 'Humidity', helper: '0 to 100 %' },
  { key: 'wind_speed', label: 'Wind speed', helper: '0 to 80 m/s' },
  { key: 'pressure', label: 'Pressure', helper: '850 to 1100 hPa (converted internally)' },
  { key: 'O3', label: 'O3 (optional)', helper: '0 to 500 µg/m³' },
  { key: 'SO2', label: 'SO2 (optional)', helper: '0 to 500 µg/m³' },
]

function baselineLabel(source) {
  if (source === 'live_api') return 'Live Open-Meteo baseline'
  if (source === 'reference_profile') return 'Reference dataset baseline'
  if (source === 'demo_default') return 'Demo default baseline'
  return 'Baseline context'
}

function asNumber(v, fallback = 0) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function displayOverrideValue(feature, value) {
  const n = asNumber(value, 0)
  if (String(feature) === 'pressure') return n * 10
  return n
}

function previewTone(level) {
  if (level === 'high') return { severity: 'warning', chip: 'warning' }
  if (level === 'medium') return { severity: 'info', chip: 'info' }
  return { severity: 'success', chip: 'success' }
}

export default function CustomWhatIfPanel({
  values,
  touched,
  errors,
  impactMode = 'conservative',
  onImpactModeChange,
  groupPresets = {},
  onApplyGroupPreset,
  impactPreviewEstimate = null,
  runImpactPreview = null,
  onValueChange,
  onFieldBlur,
  onRun,
  onClear,
  loading,
  canRunForecast,
  allValid,
  hasAnyValue,
  baselineSource,
  liveUnavailable,
  appliedOverrides = [],
}) {
  const overrideRows = Array.isArray(appliedOverrides)
    ? appliedOverrides.map((row) => {
      const feature = row?.feature || ''
      const from = displayOverrideValue(feature, row?.from ?? row?.from_value)
      const to = displayOverrideValue(feature, row?.to ?? row?.to_value)
      const delta = to - from
      const pct = from === 0 ? null : (delta / from) * 100
      return {
        feature,
        from,
        to,
        delta,
        pct,
        clamped: Boolean(row?.clamped),
        reason: row?.reason || '',
      }
    }) : []

  return (
    <Box sx={{ borderTop: `1px solid ${sk.divider}` }}>
      <Box sx={{ py: { xs: 1.1, sm: 1.25 } }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 840, color: '#10354f' }}>
          Custom What-If Forecast
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.7 }}>
          Baseline-anchored forecasting with edited current conditions.
        </Typography>

        <Alert severity="info" sx={{ mb: 1.15 }}>
          Custom What-If Forecast uses a baseline historical context and your edited current conditions.
        </Alert>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.95 }}>
          This panel is for direct variable overrides. For category-level guided interventions in live mode, use Scenario Simulator.
        </Typography>

        {liveUnavailable ? (
          <Alert severity="warning" sx={{ mb: 1 }}>
            Live data is unavailable right now. You can still continue using Custom What-If Forecast.
          </Alert>
        ) : null}

        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.2, fontWeight: 500 }}>
          Baseline source for your latest run: <strong>{baselineLabel(baselineSource)}</strong>
        </Typography>

        <Box sx={{ mb: 1.25 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.55 }}>
            Impact mode
          </Typography>
          <ToggleButtonGroup
            exclusive
            size="small"
            value={impactMode}
            onChange={(_, v) => {
              if (!v) return
              onImpactModeChange?.(v)
            }}
            sx={{ mb: 0.65, '& .MuiToggleButton-root': { textTransform: 'none', minHeight: 40, px: 1.3 } }}
          >
            <ToggleButton value="conservative">Conservative</ToggleButton>
            <ToggleButton value="stronger_realistic">Stronger realistic</ToggleButton>
          </ToggleButtonGroup>
          <Typography variant="caption" color="text.secondary">
            Stronger realistic mode pushes selected conditions closer to outer training ranges while staying baseline-anchored and plausibility-bounded.
          </Typography>
        </Box>

        <Box sx={{ mb: 1.25 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.55 }}>
            Intervention groups (starting templates)
          </Typography>
          <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap" sx={{ mb: 0.65 }}>
            {Object.entries(groupPresets || {}).map(([groupId, cfg]) => (
              <Button
                key={groupId}
                size="small"
                variant="outlined"
                onClick={() => onApplyGroupPreset?.(groupId)}
              >
                {cfg.title}
              </Button>
            ))}
          </Stack>
          <Box sx={{ display: 'grid', gap: 0.25, mb: 0.5 }}>
            {Object.entries(groupPresets || {}).map(([groupId, cfg]) => (
              <Typography key={`desc-${groupId}`} variant="caption" color="text.secondary">
                <strong>{cfg.title}:</strong> {cfg.description}
              </Typography>
            ))}
          </Box>
          <Typography variant="caption" color="text.secondary">
            Templates fill editable current-condition fields only. You can adjust values manually before running.
          </Typography>
        </Box>

        {impactPreviewEstimate ? (
          <Alert severity={previewTone(impactPreviewEstimate.level).severity} sx={{ mb: 1.05 }}>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>
              Impact preview (estimate): {String(impactPreviewEstimate.level || 'low').toUpperCase()}
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', mt: 0.35 }}>
              {impactPreviewEstimate.note || 'Preview uses selected controls and typical model sensitivity.'}
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', mt: 0.35, color: 'text.secondary' }}>
              Estimated preview only — final forecast may differ after full model evaluation.
            </Typography>
            {Array.isArray(impactPreviewEstimate.factors) && impactPreviewEstimate.factors.length ? (
              <Stack direction="row" spacing={0.6} useFlexGap flexWrap="wrap" sx={{ mt: 0.6 }}>
                {impactPreviewEstimate.factors.slice(0, 3).map((f) => (
                  <Chip key={`est-${f}`} size="small" variant="outlined" label={labelFeature(f)} />
                ))}
              </Stack>
            ) : null}
          </Alert>
        ) : null}

        <Grid container spacing={1.2}>
          {FIELD_CONFIG.map((f) => {
            const val = values?.[f.key] ?? ''
            const fieldError = touched?.[f.key] ? errors?.[f.key] : ''
            return (
              <Grid item xs={12} sm={6} md={4} key={f.key}>
                <TextField
                  fullWidth
                  size="small"
                  label={f.label}
                  value={val}
                  onChange={(e) => onValueChange(f.key, e.target.value)}
                  onBlur={() => onFieldBlur(f.key)}
                  error={Boolean(fieldError)}
                  helperText={fieldError || f.helper}
                />
              </Grid>
            )
          })}
        </Grid>

        <Box sx={{ mt: 1.1 }}>
          {!hasAnyValue ? (
            <Typography variant="caption" color="text.secondary">
              Enter at least one value to run a custom forecast.
            </Typography>
          ) : null}
          {hasAnyValue && !allValid ? (
            <Typography variant="caption" color="error">
              Fix invalid values before submitting.
            </Typography>
          ) : null}
          {!canRunForecast ? (
            <Typography variant="caption" color="error" sx={{ display: 'block', mt: 0.5 }}>
              Enter valid coordinates first (latitude: -90 to 90, longitude: -180 to 180).
            </Typography>
          ) : null}
        </Box>

        <Box sx={{ display: 'flex', gap: 0.85, mt: 1.2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            onClick={onRun}
            disabled={loading || !canRunForecast || !allValid || !hasAnyValue}
            sx={{ fontWeight: 800 }}
          >
            Run Custom What-If Forecast
          </Button>
          <Button variant="outlined" onClick={onClear} disabled={loading}>
            Clear Inputs
          </Button>
        </Box>

        {overrideRows.length > 0 ? (
          <Box sx={{ mt: 1.45 }}>
            {runImpactPreview ? (
              <Alert severity={previewTone(runImpactPreview.level).severity} sx={{ mb: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  Last run calibrated preview: {String(runImpactPreview.level || 'low').toUpperCase()}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 0.3 }}>
                  {runImpactPreview.note || 'Estimated from actual applied override magnitudes and model leverage.'}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 0.3, color: 'text.secondary' }}>
                  Estimated preview only — final forecast may differ after full model evaluation.
                </Typography>
                {Array.isArray(runImpactPreview.factors) && runImpactPreview.factors.length ? (
                  <Stack direction="row" spacing={0.6} useFlexGap flexWrap="wrap" sx={{ mt: 0.55 }}>
                    {runImpactPreview.factors.slice(0, 3).map((f) => (
                      <Chip key={`run-${f}`} size="small" variant="outlined" label={labelFeature(f)} />
                    ))}
                  </Stack>
                ) : null}
              </Alert>
            ) : null}
            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.7 }}>
              Applied Overrides
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.8 }}>
              Values below reflect what was actually applied by the backend after bounds/clamping checks.
              Pressure is displayed in hPa for readability.
            </Typography>
            <TableContainer sx={{ overflowX: 'auto', border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Feature</TableCell>
                    <TableCell>Baseline</TableCell>
                    <TableCell>Final</TableCell>
                    <TableCell>Δ</TableCell>
                    <TableCell>Δ%</TableCell>
                    <TableCell>Clamped</TableCell>
                    <TableCell>Notes</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {overrideRows.map((r, idx) => (
                    <TableRow key={`${r.feature}-${idx}`}>
                      <TableCell>{labelFeature(r.feature)}</TableCell>
                      <TableCell>{r.from.toFixed(3)}</TableCell>
                      <TableCell>{r.to.toFixed(3)}</TableCell>
                      <TableCell>{r.delta >= 0 ? '+' : ''}{r.delta.toFixed(3)}</TableCell>
                      <TableCell>{r.pct === null ? '-' : `${r.pct >= 0 ? '+' : ''}${r.pct.toFixed(1)}%`}</TableCell>
                      <TableCell>
                        <Chip size="small" color={r.clamped ? 'warning' : 'success'} label={r.clamped ? 'Yes' : 'No'} />
                      </TableCell>
                      <TableCell>{r.reason || '-'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        ) : null}
      </Box>
    </Box>
  )
}
