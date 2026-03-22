import React, { useMemo } from 'react'
import {
  Typography,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Divider,
  Alert,
  Box,
  Chip,
  Stack,
} from '@mui/material'
import { labelFeature } from '../utils/labels'
import { sk } from '../theme/tokens'

function prettyMethod(method) {
  if (method === 'xgboost_pred_contribs') return 'XGBoost additive pred_contribs'
  if (method === 'tree_shap') return 'TreeSHAP fallback'
  if (method === 'unavailable') return 'Unavailable for this run'
  if (!method) return 'Not reported'
  return String(method)
}

function uniqueLines(lines) {
  if (!Array.isArray(lines)) return []
  const seen = new Set()
  const cleaned = []
  for (const raw of lines) {
    const text = String(raw || '').trim()
    if (!text) continue
    const key = text.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    cleaned.push(text)
  }
  return cleaned
}

function humanizeExplanationText(raw) {
  const text = String(raw || '').trim()
  if (!text) return ''

  return text
    .replace(/\bFor this run,\s*/gi, '')
    .replace(/\bIn this run,\s*/gi, '')
    .replace(/\bthis run's\b/gi, 'the')
    .replace(/\bforecast signal\b/gi, 'forecast tendency')
    .replace(/\bmodel signal\b/gi, 'forecast tendency')
    .replace(/\battribution\b/gi, 'explanation')
    .replace(/\bcontributes more positively than baseline\b/gi, 'shows a stronger upward influence than baseline')
    .replace(/\bcontributes more negatively than baseline\b/gi, 'shows a stronger downward influence than baseline')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function ImpactPill({ direction }) {
  if (direction === 'up') {
    return <Chip size="small" color="warning" label="Upward influence" sx={{ fontWeight: 700 }} />
  }
  if (direction === 'down') {
    return <Chip size="small" color="success" label="Downward influence" sx={{ fontWeight: 700 }} />
  }
  return <Chip size="small" variant="outlined" label="Mixed influence" sx={{ fontWeight: 700 }} />
}

function EvidenceNarrative({ title, lines, subtitle = '' }) {
  return (
    <Box
      sx={{
        height: '100%',
        p: { xs: 0.25, sm: 0.4 },
      }}
    >
      <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>{title}</Typography>
      {subtitle ? (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.2, mb: 0.7 }}>
          {subtitle}
        </Typography>
      ) : (
        <Box sx={{ mb: 0.45 }} />
      )}

      {!lines?.length ? (
        <Typography variant="body2" color="text.secondary">No explanation lines available for this run.</Typography>
      ) : (
        <Stack spacing={0.7}>
          {lines.map((line, idx) => (
            <Box
              key={`${title}-${idx}`}
              sx={{
                borderLeft: '3px solid rgba(11, 94, 131, 0.35)',
                p: 0.52,
              }}
            >
              <Typography variant="body2" sx={{ color: '#1d425a', lineHeight: 1.45 }}>
                {line}
              </Typography>
            </Box>
          ))}
        </Stack>
      )}
    </Box>
  )
}

function InfluenceLane({ rows, title, direction }) {
  const filtered = (rows || []).filter((row) => {
    const d = String(row?.direction || '')
    return direction === 'up' ? d === 'up' : d === 'down'
  })
  const maxAbs = Math.max(
    ...filtered.map((r) => Math.abs(Number(r?.shap || 0))),
    0.0001
  )

  return (
    <Box sx={{ height: '100%', p: 0.5 }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.8 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>{title}</Typography>
          <ImpactPill direction={direction} />
        </Stack>

        {!filtered.length ? (
          <Typography variant="body2" color="text.secondary">
            No dominant {direction === 'up' ? 'upward' : 'downward'} signal in this run.
          </Typography>
        ) : (
          <Stack spacing={0.8}>
            {filtered.slice(0, 4).map((r, idx) => {
              const abs = Math.abs(Number(r?.shap || 0))
              const width = Math.max(8, (abs / maxAbs) * 100)
              return (
                <Box key={`${direction}-${r.feature || idx}`}>
                  <Stack direction="row" spacing={0.8} alignItems="center" justifyContent="space-between">
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      {r.feature_label || labelFeature(r.feature)}
                    </Typography>
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      {Number(r.shap).toFixed(3)}
                    </Typography>
                  </Stack>
                  <Box className="influence-bar-track" sx={{ mt: 0.34 }}>
                    <Box
                      className={direction === 'up' ? 'influence-bar-fill-up' : 'influence-bar-fill-down'}
                      sx={{ width: `${width}%` }}
                    />
                  </Box>
                </Box>
              )
            })}
          </Stack>
        )}
    </Box>
  )
}

function DriverTable({ title, rows }) {
  return (
    <Box
      sx={{
        height: '100%',
        p: { xs: 0.32, sm: 0.45 },
      }}
    >
        <Typography variant="subtitle1" sx={{ fontWeight: 700 }} gutterBottom>{title}</Typography>
        <TableContainer sx={{ overflowX: 'auto' }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Factor</TableCell>
                <TableCell>Value</TableCell>
                <TableCell>Impact score</TableCell>
                <TableCell>Direction</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(rows || []).map((r) => (
                <TableRow key={`${title}-${r.feature}`}>
                  <TableCell>{r.feature_label || labelFeature(r.feature)}</TableCell>
                  <TableCell>{Number(r.value).toFixed(2)}</TableCell>
                  <TableCell>{Number(r.shap).toFixed(3)}</TableCell>
                  <TableCell>{r.direction === 'up' ? 'higher' : 'lower'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
    </Box>
  )
}

function ImportanceBars({ rows }) {
  if (!rows?.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        Feature-importance data is not available in current model metadata.
      </Typography>
    )
  }

  const maxPct = Math.max(...rows.map((r) => Number(r.pct || 0)), 1)
  return (
    <Box>
      {rows.slice(0, 10).map((r) => {
        const pct = Number(r.pct || 0)
        const width = Math.max(4, (pct / maxPct) * 100)
        return (
          <Box key={`imp-${r.feature}`} sx={{ mb: 0.95 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
              <Typography variant="caption">{r.feature_label || labelFeature(r.feature)}</Typography>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>{pct.toFixed(1)}%</Typography>
            </Box>
            <Box className="influence-bar-track" sx={{ mt: 0.32 }}>
              <Box className="influence-bar-fill-up" sx={{ width: `${width}%`, background: 'linear-gradient(90deg, #1976d2 0%, #2fa2f0 100%)' }} />
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}

export default function ExplainabilityPanel({ data, modelInfo, hasScenario = false }) {
  if (!data) {
    return (
      <Box className="explainability-star-shell" sx={{ py: { xs: 1.1, sm: 1.25 } }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title }}>
          Explainability
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.55, color: sk.ink.muted }}>
          Model influence summary for the current run.
        </Typography>
        <Typography variant="body2" sx={{ color: sk.ink.muted }}>Run a baseline or scenario first.</Typography>
      </Box>
    )
  }

  const baselineShap = data?.baseline?.shap || {}
  const scenarioShap = data?.scenario?.shap || {}
  const deltaShap = data?.delta?.delta_shap || {}
  const hasGlobalShap = Boolean(modelInfo?.has_global_shap)
  const targetSpaceNote = String(baselineShap?.target_space_note || '').trim()
  const method = String(baselineShap?.method || '').trim()
  const methodLabel = prettyMethod(method)
  const baseValue = Number(baselineShap?.base_value)
  const additivityError = Number(baselineShap?.additivity_error)
  const additivityTolerance = Number(baselineShap?.additivity_tolerance)
  const additivityOk = baselineShap?.additivity_ok
  const alignmentError = Number(baselineShap?.prediction_alignment_error)
  const alignmentOk = baselineShap?.prediction_alignment_ok
  const baselineLines = uniqueLines(baselineShap?.plain_language || []).map(humanizeExplanationText).filter(Boolean)
  const scenarioLines = uniqueLines(scenarioShap?.plain_language || []).map(humanizeExplanationText).filter(Boolean)
  const deltaLines = uniqueLines(deltaShap?.plain_language || []).map(humanizeExplanationText).filter(Boolean)
  const friendlyTargetNote = targetSpaceNote || 'This shows which inputs most influenced this run before final PM2.5 reconstruction.'
  const overrideCount = Array.isArray(data?.scenario?.applied_overrides) ? data.scenario.applied_overrides.length : 0
  const baselineDrivers = Array.isArray(baselineShap?.top_drivers) ? baselineShap.top_drivers : []
  const scenarioDrivers = Array.isArray(scenarioShap?.top_drivers) ? scenarioShap.top_drivers : []
  const deltaDrivers = Array.isArray(deltaShap?.top_changes) ? deltaShap.top_changes : []

  let methodStatus = 'Method metadata is limited; contribution lines are still shown for transparency.'
  if (method === 'unavailable') {
    methodStatus = 'Detailed explanation is unavailable for this run, but forecast outputs remain valid.'
  } else if (method === 'xgboost_pred_contribs') {
    if (additivityOk === true && alignmentOk !== false) {
      methodStatus = 'Contribution integrity checks passed for this run.'
    } else {
      methodStatus = 'Integrity checks are incomplete for this run; interpret influence direction as guidance.'
    }
  } else if (method === 'tree_shap') {
    methodStatus = 'TreeSHAP fallback was used; explanation remains useful but strict additivity can be approximate.'
  }

  const insightHeadline = useMemo(() => {
    if (baselineDrivers.length === 0) return 'No dominant driver emerged in this run.'
    const top = baselineDrivers[0]
    const label = top.feature_label || labelFeature(top.feature)
    const direction = String(top.direction || '') === 'up' ? 'upward' : 'downward'
    return `Strongest baseline signal: ${label} (${direction} influence).`
  }, [baselineDrivers])

  return (
    <Box className="explainability-star-shell" sx={{ py: { xs: 1.1, sm: 1.25 } }}>
        <Typography variant="overline" sx={{ color: '#2b5a76', letterSpacing: '0.08em', fontWeight: 800 }}>
          Primary Analysis View
        </Typography>
        <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title }}>
          Explainability
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.65, color: sk.ink.muted }}>
          How model influence shifted for this specific run.
        </Typography>

        <Alert severity="info" sx={{ mb: 0.95 }}>
          {hasScenario
            ? 'Review baseline drivers first, then scenario drivers, then contribution shifts. These are model-influence diagnostics, not causal claims.'
            : 'Review baseline influence evidence for this run. Add a scenario to compare contribution shifts.'}
        </Alert>

        <Box sx={{ py: { xs: 0.72, sm: 0.85 }, mb: 1.05, borderTop: `1px solid ${sk.divider}`, borderBottom: `1px solid ${sk.divider}`, borderRadius: 1, px: { xs: 0.5, sm: 0.75 }, bgcolor: 'rgba(255,255,255,0.45)' }}>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={0.7}
            alignItems={{ xs: 'flex-start', sm: 'center' }}
            justifyContent="space-between"
          >
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, color: sk.ink.title }}>
                {insightHeadline}
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.24 }}>
                {humanizeExplanationText(friendlyTargetNote)}
              </Typography>
            </Box>
            <Stack direction="row" spacing={0.6} useFlexGap flexWrap="wrap">
              <Chip size="small" label={methodLabel} variant="outlined" />
              {hasScenario ? <Chip size="small" label={`Mapped scenario edits: ${overrideCount}`} variant="outlined" /> : null}
            </Stack>
          </Stack>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.55 }}>
            {methodStatus}
          </Typography>
        </Box>

        <Grid container spacing={1.05}>
          <Grid item xs={12} md={hasScenario ? 6 : 12}>
            <EvidenceNarrative
              title="Baseline interpretation"
              subtitle="Top local explanation lines for the baseline run."
              lines={baselineLines}
            />
          </Grid>
          {hasScenario ? (
            <Grid item xs={12} md={6}>
              <EvidenceNarrative
                title="Scenario interpretation"
                subtitle="Top local explanation lines for the intervention run."
                lines={scenarioLines}
              />
            </Grid>
          ) : null}
        </Grid>

        <Grid container spacing={1.05} sx={{ mt: 0.15 }}>
          <Grid item xs={12} md={6}>
            <InfluenceLane rows={baselineDrivers} title="Baseline upward signals" direction="up" />
          </Grid>
          <Grid item xs={12} md={6}>
            <InfluenceLane rows={baselineDrivers} title="Baseline downward signals" direction="down" />
          </Grid>
        </Grid>

        {hasScenario ? (
          <>
            <Divider sx={{ my: 1.15, borderColor: sk.divider }} />
            <Grid container spacing={1.05}>
              <Grid item xs={12} md={6}>
                <InfluenceLane rows={scenarioDrivers} title="Scenario upward signals" direction="up" />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfluenceLane rows={scenarioDrivers} title="Scenario downward signals" direction="down" />
              </Grid>
            </Grid>
            <Box sx={{ mt: 1 }}>
              <EvidenceNarrative
                title="Mapped influence shifts"
                subtitle="How scenario controls changed contribution direction and strength versus baseline."
                lines={deltaLines}
              />
            </Box>
          </>
        ) : null}

        <Divider sx={{ my: 1.2, borderColor: sk.divider }} />
        <Box sx={{ p: { xs: 0.2, sm: 0.3 } }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }} gutterBottom>
              Global feature context
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {hasGlobalShap
                ? 'Average absolute SHAP contribution across an evaluation sample (global pattern context, not run-specific).'
                : 'Global ranking uses XGBoost gain fallback because global SHAP metadata is unavailable.'}
            </Typography>
            <ImportanceBars rows={modelInfo?.feature_importance || []} />
        </Box>

        <Divider sx={{ my: 1.35, borderColor: sk.divider }} />
        <Box component="details">
          <Box component="summary" sx={{ cursor: 'pointer', fontWeight: 700, mb: 1 }}>
            Technical evidence and audit table
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
            These rows are for technical verification and viva discussion.
          </Typography>

          <Grid container spacing={1.2}>
            <Grid item xs={12} md={hasScenario ? 6 : 12}>
              <DriverTable title="Baseline top drivers" rows={baselineDrivers} />
            </Grid>
            {hasScenario ? (
              <Grid item xs={12} md={6}>
                <DriverTable title="Scenario top drivers" rows={scenarioDrivers} />
              </Grid>
            ) : null}
          </Grid>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.8 }}>
            Method: {methodLabel || 'not reported'}
            {Number.isFinite(baseValue) ? ` · Base value: ${baseValue.toFixed(3)}` : ''}
            {Number.isFinite(additivityError) ? ` · Additivity check error: ${additivityError.toExponential(2)}` : ''}
            {Number.isFinite(additivityTolerance) ? ` · Tolerance: ${additivityTolerance.toExponential(1)}` : ''}
            {typeof additivityOk === 'boolean' ? ` · Additivity: ${additivityOk ? 'pass' : 'check needed'}` : ''}
            {Number.isFinite(alignmentError) ? ` · Signal alignment error: ${alignmentError.toExponential(2)}` : ''}
            {typeof alignmentOk === 'boolean' ? ` · Alignment: ${alignmentOk ? 'pass' : 'check needed'}` : ''}
          </Typography>

          {hasScenario ? (
            <>
              <Divider sx={{ my: 1.5 }} />
              <Typography variant="subtitle1" sx={{ fontWeight: 700 }} gutterBottom>
                Top contribution shifts (baseline vs scenario)
              </Typography>
              <TableContainer sx={{ overflowX: 'auto' }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Factor</TableCell>
                      <TableCell>Baseline</TableCell>
                      <TableCell>Scenario</TableCell>
                      <TableCell>Change</TableCell>
                      <TableCell>Sign flip</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {deltaDrivers.map((r) => (
                      <TableRow key={`delta-${r.feature}`}>
                        <TableCell>{r.feature_label || labelFeature(r.feature)}</TableCell>
                        <TableCell>{Number(r.baseline_shap).toFixed(3)}</TableCell>
                        <TableCell>{Number(r.scenario_shap).toFixed(3)}</TableCell>
                        <TableCell>{Number(r.delta_shap).toFixed(3)}</TableCell>
                        <TableCell>{r.sign_flip ? 'yes' : 'no'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Typography variant="body2" sx={{ mt: 0.9 }}>
                {deltaShap?.summary_text || 'No scenario contribution shift summary is available for this run.'}
              </Typography>
            </>
          ) : null}
        </Box>
    </Box>
  )
}
