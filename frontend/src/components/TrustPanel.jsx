import React from 'react'
import {
  Typography,
  Alert,
  Box,
  Divider,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
} from '@mui/material'
import { labelFeature } from '../utils/labels'
import { sk } from '../theme/tokens'

function safePct(v) {
  return Math.min(100, Math.max(0, Number(v || 0)))
}

function safeNumber(v, fallback = 0) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function pct(v, digits = 0) {
  return `${safePct(v).toFixed(digits)}%`
}

function qualityBand(score) {
  const s = Number(score || 0)
  if (s >= 0.8) return { label: 'High', color: 'success' }
  if (s >= 0.65) return { label: 'Good', color: 'success' }
  if (s >= 0.5) return { label: 'Moderate', color: 'warning' }
  return { label: 'Low', color: 'error' }
}

function scoreTone(scorePct) {
  if (scorePct >= 80) return 'success'
  if (scorePct >= 60) return 'info'
  if (scorePct >= 45) return 'warning'
  return 'error'
}

function guidance({ coveragePct, imputationPct, oodFlag, gaps }) {
  if (oodFlag) return 'Some values are outside the model’s usual training range. Treat this run as directional guidance.'
  if (imputationPct > 20) return 'Many features used fallback defaults in this run, so reliability guidance is lower.'
  if (coveragePct < 50) return 'Recent history is limited. Re-check when more hours are available.'
  if (gaps > 2) return 'Multiple data gaps were detected; small differences may be unstable.'
  return 'Input quality is stable for normal interpretation.'
}

function componentMeta(name) {
  const key = String(name || '').trim().toLowerCase()
  if (key === 'data_completeness') {
    return {
      label: 'Data completeness',
      short: 'History/features available',
      tone: 'success',
      meaning: 'Higher means more required history/features were available before fallback.',
    }
  }
  if (key === 'domain_plausibility') {
    return {
      label: 'Domain plausibility',
      short: 'In-range plausibility',
      tone: 'info',
      meaning: 'Higher means inputs stayed closer to training quantile ranges.',
    }
  }
  if (key === 'imputation_burden') {
    return {
      label: 'Imputation burden control',
      short: 'Defaulted-feature burden',
      tone: 'warning',
      meaning: 'Higher means fewer model features were defaulted/imputed.',
    }
  }
  if (key === 'fallback_severity') {
    return {
      label: 'Fallback path quality',
      short: 'Fallback severity control',
      tone: 'warning',
      meaning: 'Higher means fallback path was lighter (less severe).',
    }
  }
  if (key === 'scenario_validity') {
    return {
      label: 'Scenario validity',
      short: 'Clamp/constraint checks',
      tone: 'info',
      meaning: 'Higher means fewer clamp or direction-limit constraints were triggered.',
    }
  }
  if (key === 'explainability_integrity') {
    return {
      label: 'Explainability integrity',
      short: 'Explanation diagnostics',
      tone: 'success',
      meaning: 'Higher means local explanation diagnostics were available and stable.',
    }
  }
  return {
    label: String(name || 'Unknown factor').replaceAll('_', ' '),
    short: 'Component signal',
    tone: 'default',
    meaning: 'Higher means better reliability contribution for this component.',
  }
}

function plainLanguageSummary({ scorePct, hasCautionSignals, cautionCount }) {
  if (hasCautionSignals) {
    return `Overall run score is ${scorePct.toFixed(0)}%, with ${cautionCount} active caution signal${cautionCount === 1 ? '' : 's'} to review before acting on magnitude.`
  }
  if (scorePct >= 80) return 'Overall reliability guidance is strong for normal interpretation of this run.'
  if (scorePct >= 60) return 'Overall reliability guidance is usable with normal caution.'
  return 'Overall reliability guidance is limited; treat this forecast as directional only.'
}

function reliabilityRationale({ health, meta, scenario }) {
  const coveragePct = safePct((Number(health?.history?.coverage_ratio || 0)) * 100)
  const oodFlag = Boolean(health?.ood?.flag)
  const hardCount = Number(health?.ood?.hard_count || 0)
  const softCount = Number(health?.ood?.soft_count || 0)
  const extremeCount = Number(health?.extreme_inputs?.count || 0)
  const fallbackLabel = String(health?.fallback?.label || '').toLowerCase()
  const baselineSource = String(meta?.baseline_source || '')
  const liveUsed = Boolean(meta?.live_data_used)
  const clampedCount = Array.isArray(scenario?.applied_overrides)
    ? scenario.applied_overrides.filter((x) => Boolean(x?.clamped)).length
    : 0

  if (extremeCount >= 3) {
    return `Primary caution: ${extremeCount} current inputs are at extreme tails versus training data.`
  }
  if (extremeCount >= 1) {
    return `Primary caution: ${extremeCount} current input${extremeCount === 1 ? ' is' : 's are'} at extreme tails versus training data.`
  }
  if (oodFlag) {
    const sev = hardCount > 0 ? `${hardCount} hard` : `${softCount} soft`
    return `Primary caution: this run exceeded normal training ranges (${sev} out-of-range event${hardCount + softCount === 1 ? '' : 's'}).`
  }
  if (!liveUsed) {
    return `Primary caution: live data was unavailable and ${baselineSource || 'fallback'} baseline context was used.`
  }
  if (clampedCount >= 2) {
    return `Primary caution: ${clampedCount} override values were clamped to training bounds.`
  }
  if (fallbackLabel.includes('high') || fallbackLabel.includes('level 2') || fallbackLabel.includes('level 3')) {
    return 'Primary caution: multiple features relied on fallback/default values.'
  }
  if (coveragePct >= 70) {
    return 'No major caution signal detected; history coverage and in-range checks are stable.'
  }
  return 'Primary caution: recent history coverage is limited for this run.'
}

function summarizeSignals({ history, gaps, imputation, ood, extremeInputs, fallback }) {
  const coveragePct = safePct((Number(history?.coverage_ratio || 0)) * 100)
  const imputationPct = safePct((Number(imputation?.ratio || 0)) * 100)
  const oodPct = safePct((Number(ood?.score || 0)) * 100)
  const extremeCount = Number(extremeInputs?.count || 0)

  return [
    {
      key: 'coverage',
      label: 'History coverage',
      value: `${coveragePct.toFixed(1)}%`,
      status: coveragePct >= 85 ? 'Strong' : coveragePct >= 70 ? 'Acceptable' : 'Limited',
      tone: coveragePct >= 70 ? 'success' : 'warning',
      note: `${history?.used_hours || 0}h of ${history?.target_hours || 0}h`,
    },
    {
      key: 'defaults',
      label: 'Fallback/default usage',
      value: `${imputationPct.toFixed(1)}%`,
      status: imputationPct <= 10 ? 'Low' : imputationPct <= 20 ? 'Moderate' : 'High',
      tone: imputationPct <= 10 ? 'success' : 'warning',
      note: `${imputation?.imputed_features || 0}/${imputation?.total_features || 0} features defaulted`,
    },
    {
      key: 'range',
      label: 'Out-of-range risk',
      value: `${oodPct.toFixed(0)}%`,
      status: ood?.flag ? 'Review' : 'OK',
      tone: ood?.flag ? 'warning' : 'success',
      note: `${ood?.soft_count || 0} soft, ${ood?.hard_count || 0} hard exceedances`,
    },
    {
      key: 'extreme',
      label: 'Extreme-tail inputs',
      value: String(extremeCount),
      status: extremeCount > 0 ? 'Present' : 'None',
      tone: extremeCount > 0 ? 'warning' : 'success',
      note: `Gaps: ${gaps?.gap_count || 0} · Fallback mode: ${fallback?.label || 'full-data path'}`,
    },
  ]
}

function describeExtremeEvent(event) {
  const feature = labelFeature(event?.feature || 'unknown')
  const value = safeNumber(event?.value)
  const q01 = safeNumber(event?.q01)
  const q99 = safeNumber(event?.q99)
  const side = String(event?.side || '')
  const sideText = side === 'above_q99' ? 'above q99' : side === 'below_q01' ? 'below q01' : 'outside q01-q99'
  return `${feature}: ${value.toFixed(2)} (${sideText}; q01-q99 ${q01.toFixed(2)} to ${q99.toFixed(2)})`
}

function describeOodEvent(event) {
  const feature = labelFeature(event?.feature || 'unknown')
  const value = safeNumber(event?.value)
  const q01 = safeNumber(event?.q01)
  const q99 = safeNumber(event?.q99)
  const bound = String(event?.bound || '').toLowerCase().includes('above')
    ? 'above upper bound'
    : String(event?.bound || '').toLowerCase().includes('below')
      ? 'below lower bound'
      : 'outside bound'
  return `${feature}: ${value.toFixed(2)} (${bound}; q01-q99 ${q01.toFixed(2)} to ${q99.toFixed(2)})`
}

const chipDenseSx = { height: 22, fontSize: '0.7rem', fontWeight: 700, maxWidth: '100%' }

function SignalRow({ row }) {
  return (
    <Box sx={{ py: 0.15, minWidth: 0 }}>
      <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={0.75} sx={{ minWidth: 0 }}>
        <Typography
          variant="caption"
          sx={{ fontWeight: 700, color: '#153a52', minWidth: 0, flex: '1 1 auto', pr: 0.5 }}
        >
          {row.label}
        </Typography>
        <Stack direction="row" spacing={0.4} alignItems="center" sx={{ flexShrink: 0 }}>
          <Typography variant="caption" sx={{ fontWeight: 800, color: '#0f3550' }}>
            {row.value}
          </Typography>
          <Chip size="small" color={row.tone} variant="outlined" label={row.status} sx={chipDenseSx} />
        </Stack>
      </Stack>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.15, color: '#2d4a5e', lineHeight: 1.38 }}>
        {row.note}
      </Typography>
    </Box>
  )
}

const panelRootSx = {
  minWidth: 0,
  width: '100%',
  maxWidth: '100%',
  overflow: 'hidden',
  boxSizing: 'border-box',
  px: { xs: 0.15, sm: 0.25 },
  py: 0,
}

const subtleAlertSx = {
  py: 0.45,
  alignItems: 'flex-start',
  border: 'none',
  bgcolor: 'rgba(13, 106, 148, 0.06)',
  '& .MuiAlert-message': { py: 0.15, width: '100%', minWidth: 0 },
}

export default function TrustPanel({ health, meta = null, scenario = null }) {
  const [detailsExpanded, setDetailsExpanded] = React.useState(false)

  if (!health) {
    return (
      <Box sx={panelRootSx}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, color: '#12384f', letterSpacing: '0.01em' }}>
          Reliability guidance
        </Typography>
        <Typography variant="caption" sx={{ mt: 0.35, display: 'block', color: '#2d4a5e' }}>
          Run baseline or scenario to view reliability guidance and empirical uncertainty context.
        </Typography>
      </Box>
    )
  }

  const { history, gaps, fallback, ood, quality_score } = health
  const imputation = health.imputation || { imputed_features: 0, total_features: 0, ratio: 0, features: [] }
  const extremeInputs = health.extreme_inputs || { count: 0, notes: [], events: [] }
  const quality = qualityBand(quality_score)
  const scorePct = safePct(Number(quality_score || 0) * 100)
  const hasExtremeCaution = Number(extremeInputs?.count || 0) >= 2
  const reliabilityBlock = health?.reliability || null
  const uncertaintyBlock = health?.uncertainty || null
  const signalRows = summarizeSignals({ history, gaps, imputation, ood, extremeInputs, fallback })
  const cautionCount = signalRows.filter((r) => r.tone !== 'success').length
  const hasCautionSignals = cautionCount > 0
  const headlineLabel = hasCautionSignals ? `${quality.label} with caution` : quality.label
  const plainSummary = plainLanguageSummary({ scorePct, hasCautionSignals, cautionCount })
  const rationale = reliabilityRationale({ health, meta, scenario })

  const componentRows = (Array.isArray(reliabilityBlock?.components) ? reliabilityBlock.components : [])
    .map((c) => {
      const score = safePct(Number(c?.score || 0) * 100)
      const weight = safePct(Number(c?.weight || 0) * 100)
      const points = safePct((Number(c?.score || 0) * Number(c?.weight || 0)) * 100)
      const metaInfo = componentMeta(c?.name)
      return {
        key: String(c?.name || 'unknown'),
        score,
        weight,
        points,
        rationale: String(c?.rationale || '').trim(),
        label: metaInfo.label,
        short: metaInfo.short,
        color: metaInfo.tone,
        meaning: metaInfo.meaning,
      }
    })
    .sort((a, b) => b.points - a.points)

  const componentSummary = componentRows.slice(0, 2)
  const topContributionText = componentSummary.length
    ? componentSummary
      .map((row) => `${row.label} ${pct(row.score)} (${row.short.toLowerCase()})`)
      .join(' · ')
    : ''

  const extremeEvents = Array.isArray(extremeInputs?.events) ? extremeInputs.events : []
  const oodEvents = Array.isArray(ood?.features_exceeded) ? ood.features_exceeded : []
  const liveUsed = Boolean(meta?.live_data_used)

  return (
    <Box sx={panelRootSx}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={0.8} sx={{ minWidth: 0 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, color: '#0f3550', letterSpacing: '0.01em' }}>
          Reliability guidance
        </Typography>
        <Chip
          size="small"
          variant="outlined"
          color={scoreTone(scorePct)}
          label={`${headlineLabel} ${scorePct.toFixed(0)}%`}
          sx={chipDenseSx}
        />
      </Stack>

      <Typography variant="caption" sx={{ display: 'block', mt: 0.45, color: sk.ink.muted, lineHeight: 1.42 }}>
        Heuristic run-quality guidance for this forecast. Not a calibrated probability of correctness.
      </Typography>

      <Alert
        severity={(ood.flag || hasExtremeCaution || hasCautionSignals) ? 'warning' : 'success'}
        sx={{ mt: 0.65, ...subtleAlertSx }}
      >
        <Typography variant="body2" sx={{ color: sk.ink.body, fontWeight: 600, lineHeight: 1.5 }}>
          {rationale}
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mt: 0.4, color: '#2d4a5e' }}>
          {plainSummary}
        </Typography>
        {topContributionText ? (
          <Typography variant="caption" sx={{ display: 'block', mt: 0.35, color: '#2d4a5e', lineHeight: 1.42 }}>
            Main factors used for this score: {topContributionText}.
          </Typography>
        ) : null}
        {scorePct >= 80 && hasCautionSignals ? (
          <Typography variant="caption" sx={{ display: 'block', mt: 0.35, color: '#2d4a5e', lineHeight: 1.42 }}>
            Why high score can still include caution: the score is a weighted mix; localized trigger checks can still flag caution even when most weighted components remain strong.
          </Typography>
        ) : null}
      </Alert>

      <Accordion
        disableGutters
        elevation={0}
        expanded={detailsExpanded}
        onChange={(_, expanded) => setDetailsExpanded(expanded)}
        sx={{
          mt: 0.95,
          border: 'none',
          borderRadius: 1,
          backgroundColor: 'rgba(13, 58, 82, 0.045)',
          '&::before': { display: 'none' },
          overflow: 'hidden',
        }}
      >
        <AccordionSummary
          expandIcon={<Typography sx={{ fontSize: 13, opacity: 0.55, lineHeight: 1 }}>▾</Typography>}
          sx={{ minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.6 } }}
        >
          <Stack
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            spacing={0.7}
            sx={{ width: '100%', pr: 0.35 }}
          >
            <Typography variant="caption" sx={{ fontWeight: 800, color: '#12384f' }}>
              {detailsExpanded ? 'Hide detailed reliability breakdown' : 'Show detailed reliability breakdown'}
            </Typography>
            <Chip
              size="small"
              variant="outlined"
              label={`${signalRows.length} signal checks`}
              sx={{ ...chipDenseSx, flexShrink: 0 }}
            />
          </Stack>
        </AccordionSummary>

        <AccordionDetails
          sx={{
            pt: 0.1,
            pb: 0.85,
            px: 0.95,
            maxHeight: { xs: 320, sm: 380, lg: 430 },
            overflowY: 'auto',
            overflowX: 'hidden',
          }}
        >
          <Typography variant="caption" sx={{ fontWeight: 800, color: sk.ink.title, display: 'block', mb: 0.62 }}>
            Run signal checks
          </Typography>
          <Stack spacing={0.85}>
            {signalRows.map((row) => (
              <SignalRow key={row.key} row={row} />
            ))}
          </Stack>

          {componentRows.length ? (
            <>
              <Typography variant="caption" sx={{ fontWeight: 800, color: sk.ink.title, display: 'block', mt: 1.15, mb: 0.5 }}>
                Component score breakdown
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', color: '#2d4a5e', lineHeight: 1.42, mb: 0.5 }}>
                Component percentages are shown on a good-direction scale: higher is better for reliability support.
              </Typography>

              <Stack spacing={0.65}>
                {componentRows.map((row) => (
                  <Box key={row.key} sx={{ minWidth: 0 }}>
                    <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={0.6}>
                      <Typography variant="caption" sx={{ color: '#153a52', fontWeight: 700, pr: 0.6 }}>
                        {row.label}
                      </Typography>
                      <Stack direction="row" spacing={0.35} alignItems="center" sx={{ flexShrink: 0 }}>
                        <Chip size="small" color={row.color} variant="outlined" label={pct(row.score)} sx={chipDenseSx} />
                        <Typography variant="caption" sx={{ color: '#355269', fontWeight: 700 }}>
                          w {pct(row.weight)}
                        </Typography>
                      </Stack>
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={row.score}
                      sx={{ mt: 0.26, mb: 0.17, height: 5, borderRadius: 999 }}
                      color={scoreTone(row.score)}
                    />
                    <Typography variant="caption" sx={{ display: 'block', color: '#2d4a5e', lineHeight: 1.35 }}>
                      {row.short}: {pct(row.points, 1)} weighted points ({pct(row.score)} × {pct(row.weight)}).
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', color: '#3a5668', lineHeight: 1.35 }}>
                      {row.meaning}
                    </Typography>
                    {row.rationale ? (
                      <Typography variant="caption" sx={{ display: 'block', color: '#3a5668', lineHeight: 1.35 }}>
                        Diagnostic basis: {row.rationale}
                      </Typography>
                    ) : null}
                  </Box>
                ))}
              </Stack>
              <Divider sx={{ my: 0.75, borderColor: sk.divider }} />
            </>
          ) : null}

          <Typography variant="caption" sx={{ fontWeight: 800, color: sk.ink.title, display: 'block', mb: 0.38 }}>
            Detailed diagnostics
          </Typography>
          <Typography variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
            History used: {history.used_hours}h / {history.target_hours}h ({safePct(history.coverage_ratio * 100).toFixed(1)}%)
          </Typography>
          <Typography variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
            History available: {history.available_hours}h | Gaps: {gaps.gap_count} (largest {gaps.largest_gap_hours}h)
          </Typography>
          <Typography variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
            Fallback mode: {fallback.label}
          </Typography>
          <Typography variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
            Out-of-range events: {ood.soft_count || 0} soft, {ood.hard_count || 0} hard
          </Typography>
          <Typography variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
            Extreme current inputs: {Number(extremeInputs?.count || 0)}
          </Typography>

          {Array.isArray(imputation?.features) && imputation.features.length ? (
            <Box sx={{ mt: 0.45 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, color: '#153a52' }}>
                Features defaulted in this run:
              </Typography>
              {imputation.features.slice(0, 8).map((f, idx) => (
                <Typography key={`imp-${idx}`} variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
                  {labelFeature(f)}
                </Typography>
              ))}
            </Box>
          ) : null}

          <Typography variant="caption" sx={{ fontWeight: 700, color: '#153a52', display: 'block', mb: 0.32 }}>
            Trigger evidence for this run
          </Typography>

          {!liveUsed ? (
            <Typography variant="caption" sx={{ display: 'block', mb: 0.35, color: '#2d4a5e', wordBreak: 'break-word' }}>
              Live data unavailable: baseline source was {String(meta?.baseline_source || 'fallback context')}.
            </Typography>
          ) : null}

          {extremeEvents.length ? (
            <Box sx={{ mb: 0.35, minWidth: 0 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, color: '#153a52' }}>
                Extreme-tail inputs:
              </Typography>
              {extremeEvents.slice(0, 5).map((e, idx) => (
                <Typography
                  key={`extreme-${idx}`}
                  variant="caption"
                  sx={{ display: 'block', mt: 0.1, color: '#2d4a5e', wordBreak: 'break-word', lineHeight: 1.38 }}
                >
                  {describeExtremeEvent(e)}
                </Typography>
              ))}
            </Box>
          ) : null}

          {oodEvents.length ? (
            <Box sx={{ mb: 0.2, minWidth: 0 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, color: '#153a52' }}>
                Quantile exceedances:
              </Typography>
              {oodEvents.slice(0, 5).map((e, idx) => (
                <Typography
                  key={`ood-${idx}`}
                  variant="caption"
                  sx={{ display: 'block', mt: 0.1, color: '#2d4a5e', wordBreak: 'break-word', lineHeight: 1.38 }}
                >
                  {describeOodEvent(e)}
                </Typography>
              ))}
            </Box>
          ) : null}

          {!extremeEvents.length && !oodEvents.length && liveUsed ? (
            <Typography variant="caption" sx={{ color: '#2d4a5e' }}>
              No feature-level caution trigger detected for this run.
            </Typography>
          ) : null}

          {Array.isArray(uncertaintyBlock?.baseline_bands) && uncertaintyBlock.baseline_bands.length ? (
            <Box sx={{ mt: 0.55 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, color: '#153a52' }}>
                Empirical uncertainty bands:
              </Typography>
              {uncertaintyBlock.baseline_bands.slice(0, 2).map((b, idx) => (
                <Typography key={`unc-baseline-${idx}`} variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
                  Baseline {Number(b.coverage_pct).toFixed(0)}%: [{Number(b.lower).toFixed(1)}, {Number(b.upper).toFixed(1)}] µg/m³
                </Typography>
              ))}
              {Array.isArray(uncertaintyBlock?.scenario_bands) && uncertaintyBlock.scenario_bands.slice(0, 2).map((b, idx) => (
                <Typography key={`unc-scenario-${idx}`} variant="caption" display="block" sx={{ color: '#2d4a5e' }}>
                  Scenario {Number(b.coverage_pct).toFixed(0)}%: [{Number(b.lower).toFixed(1)}, {Number(b.upper).toFixed(1)}] µg/m³
                </Typography>
              ))}
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.2 }}>
                {String(uncertaintyBlock.note || '')}
              </Typography>
            </Box>
          ) : null}

          <Alert severity="info" sx={{ ...subtleAlertSx, mt: 0.75 }}>
            {guidance({
              coveragePct: safePct(history?.coverage_ratio * 100),
              imputationPct: safePct(imputation?.ratio * 100),
              oodFlag: ood?.flag,
              gaps: gaps?.gap_count || 0,
            })}
          </Alert>
        </AccordionDetails>
      </Accordion>
    </Box>
  )
}
