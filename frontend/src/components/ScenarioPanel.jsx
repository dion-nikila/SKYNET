import React, { useEffect, useMemo, useState } from 'react'
import {
  Typography,
  Box,
  Tabs,
  Tab,
  Grid,
  Button,
  Slider,
  MenuItem,
  TextField,
  Stack,
  Chip,
  Alert,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableRow
} from '@mui/material'
import { CATEGORY_LABELS, labelFeature } from '../utils/labels'
import { sk } from '../theme/tokens'

const categoryOptions = ['wind', 'humidity', 'temperature', 'emission_proxy']
const dirOptions = ['increase', 'decrease']
const magOptions = ['small', 'medium', 'large']

const QUICK_TESTS = [
  { label: 'Cleaner Air Quick Test', scenario_id: 'strong_dispersion', intensity: 70 },
  { label: 'Worse Air Quick Test', scenario_id: 'trapped_pollution', intensity: 78 },
  { label: 'Dust Resuspension Test', scenario_id: 'dust_resuspension', intensity: 70 },
  { label: 'Traffic Spike', scenario_id: 'traffic_gridlock', intensity: 75 },
  { label: 'Industrial Spike', scenario_id: 'industrial_plume', intensity: 70 }
]

const GUIDED_BASE_INTENSITY = 70

const SCENARIO_EXPECTED = {
  traffic_gridlock: {
    effect: 'Typical intent: often increases PM2.5 accumulation risk',
    note: 'Boosts traffic-related pollution proxies while reducing ventilation.',
    mood: 'up',
  },
  strong_dispersion: {
    effect: 'Typical intent: often reduces PM2.5 accumulation risk',
    note: 'Core scientific signal is stronger ventilation; humidity and pressure reductions are secondary contextual adjustments.',
    mood: 'down',
  },
  heatwave: {
    effect: 'Typical intent: often increases PM2.5 accumulation risk',
    note: 'Primary scientific claim is ozone stress (O3 up); PM10 increase is treated as plausible but context-dependent.',
    mood: 'up',
  },
  dust_resuspension: {
    effect: 'Typical intent: often increases PM2.5 accumulation risk',
    note: 'Wind lifts dust and road particles, raising PM10 under drier conditions.',
    mood: 'up',
  },
  trapped_pollution: {
    effect: 'Typical intent: often increases PM2.5 accumulation risk',
    note: 'Low wind and poor dispersion can trap pollutants near the ground.',
    mood: 'up',
  },
  industrial_plume: {
    effect: 'Typical intent: often increases PM2.5 accumulation risk',
    note: 'SO2, NO2, CO, and PM10 rise together to represent industrial-combustion loading under weak dispersion.',
    mood: 'up',
  }
}

function intensityLabel(v) {
  if (v < 20) return 'Very Mild'
  if (v < 45) return 'Mild'
  if (v < 70) return 'Moderate'
  if (v < 90) return 'Strong'
  return 'Extreme'
}

function sliderNumber(v) {
  const raw = Array.isArray(v) ? v[0] : v
  const n = Number(raw)
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(100, Math.round(n)))
}

function moodStyles(mood) {
  if (mood === 'down') {
    return {
      border: 'rgba(33, 141, 100, 0.36)',
      bg: 'linear-gradient(180deg, rgba(229, 250, 239, 0.95) 0%, rgba(255,255,255,0.98) 100%)',
      chip: 'success',
    }
  }
  return {
    border: 'rgba(186, 127, 38, 0.34)',
    bg: 'linear-gradient(180deg, rgba(255, 242, 226, 0.94) 0%, rgba(255,255,255,0.98) 100%)',
    chip: 'warning',
  }
}

function MacroCard({ scenario, selected, onSelect }) {
  const expectation = SCENARIO_EXPECTED[scenario.scenario_id]
  const mood = moodStyles(expectation?.mood)
  return (
    <Box
      onClick={onSelect}
      sx={{
        cursor: 'pointer',
        border: `1px solid ${selected ? '#0b5e83' : mood.border}`,
        borderLeftWidth: 4,
        background: selected ? 'rgba(19,93,119,0.08)' : 'rgba(255,255,255,0.65)',
        borderRadius: 1,
        height: '100%',
        transform: selected ? 'translateY(-2px)' : 'none',
        transition: 'transform 180ms ease, border-color 180ms ease',
        '&:hover': {
          borderColor: selected ? 'primary.main' : 'rgba(19,93,119,0.4)',
        }
      }}
    >
      <Box sx={{ p: 1.35 }}>
        <Stack direction="row" spacing={0.7} alignItems="center" justifyContent="space-between" sx={{ mb: 0.52 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 900, color: '#12354d' }}>
            {scenario.title}
          </Typography>
          {selected ? <Chip size="small" color="primary" label="Active" sx={{ fontWeight: 700 }} /> : null}
        </Stack>

        <Typography variant="body2" color="text.secondary" sx={{ minHeight: 42, mb: 1.05, lineHeight: 1.45 }}>
          {scenario.description}
        </Typography>

        <Stack direction="row" spacing={0.55} useFlexGap flexWrap="wrap">
          {expectation?.effect ? (
            <Chip
              size="small"
              color={mood.chip}
              variant="outlined"
              label={expectation.effect}
              sx={{
                fontWeight: 700,
                height: 'auto',
                alignItems: 'flex-start',
                maxWidth: '100%',
                '& .MuiChip-label': {
                  display: 'block',
                  whiteSpace: 'normal',
                  overflow: 'visible',
                  textOverflow: 'clip',
                  lineHeight: 1.28,
                  px: 0.9,
                  py: 0.35,
                },
              }}
            />
          ) : null}
          <span className="scenario-type-pill">Macro template</span>
        </Stack>
      </Box>
    </Box>
  )
}

function guidedRowPreview(item) {
  return `${CATEGORY_LABELS[item.category]} set to ${item.direction} with ${item.magnitude} strength.`
}

export default function ScenarioPanel({ scenarios, loading, onApply, canRunForecast = true }) {
  const [tab, setTab] = useState(0)
  const [macroId, setMacroId] = useState('')
  const [macroIntensity, setMacroIntensity] = useState(70)

  const [items, setItems] = useState([
    { category: 'wind', direction: 'decrease', magnitude: 'large' },
    { category: 'humidity', direction: 'increase', magnitude: 'medium' }
  ])

  useEffect(() => {
    if (!macroId && scenarios?.length) {
      setMacroId(scenarios[0].scenario_id)
      setMacroIntensity(scenarios[0].default_intensity || 70)
    }
  }, [scenarios, macroId])

  const selectedMacro = useMemo(
    () => (scenarios || []).find((s) => s.scenario_id === macroId),
    [scenarios, macroId]
  )

  const usedCategories = useMemo(
    () => new Set(items.map((it) => it.category)),
    [items]
  )
  const hasDuplicateCategories = usedCategories.size !== items.length

  const addItem = () =>
    setItems((s) => {
      const used = new Set(s.map((it) => it.category))
      const nextCategory = categoryOptions.find((c) => !used.has(c))
      if (!nextCategory) return s
      return [...s, { category: nextCategory, direction: 'increase', magnitude: 'small' }]
    })

  const updateItem = (idx, key, value) =>
    setItems((s) => s.map((it, i) => (i === idx ? { ...it, [key]: value } : it)))

  const removeItem = (idx) => setItems((s) => s.filter((_, i) => i !== idx))

  return (
    <Box className="scenario-studio-shell">
      <Box sx={{ p: { xs: 1.2, sm: 1.35 } }}>
        <Typography variant="subtitle1" sx={{ color: sk.ink.title, fontWeight: 800 }}>
          Scenario controls
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.95, color: sk.ink.muted }}>
          Live-mode interventions anchored to the current baseline forecast.
        </Typography>

        {!canRunForecast ? (
          <Alert severity="warning" sx={{ mb: 1.2 }}>
            Enter valid coordinates first (latitude: -90 to 90, longitude: -180 to 180).
          </Alert>
        ) : null}

        <Box component="details" sx={{ mt: 0.9, mb: 1.2 }}>
          <Box component="summary" sx={{ cursor: 'pointer', fontWeight: 700, color: 'text.secondary' }}>
            Quick action presets
          </Box>
          <Box
            sx={{
              display: 'flex',
              gap: 0.85,
              flexWrap: { xs: 'nowrap', md: 'wrap' },
              overflowX: { xs: 'auto', md: 'visible' },
              pb: { xs: 0.45, md: 0 },
              mt: 0.85,
            }}
          >
            {QUICK_TESTS.map((q) => (
              <Button
                key={q.label}
                size="small"
                variant="outlined"
                sx={{ flexShrink: 0, minHeight: 30 }}
                onClick={() => {
                  setTab(0)
                  setMacroId(q.scenario_id)
                  setMacroIntensity(q.intensity)
                  onApply({ type: 'macro', scenario_id: q.scenario_id, intensity: q.intensity })
                }}
                disabled={loading || !canRunForecast}
              >
                {q.label}
              </Button>
            ))}
          </Box>
        </Box>

        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={{
            mb: 1.15,
            borderTop: `1px solid ${sk.divider}`,
            borderBottom: `1px solid ${sk.divider}`,
            borderRadius: 1,
            bgcolor: 'rgba(255,255,255,0.4)',
            py: 0.15,
            '& .MuiTab-root': { minHeight: 38, fontWeight: 700 },
            '& .Mui-selected': { backgroundColor: 'rgba(255,255,255,0.88)', color: 'primary.dark' },
          }}
        >
          <Tab label="Macro templates" />
          <Tab label="Guided intervention" />
        </Tabs>

        {tab === 0 ? (
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1.1 }}>
              Macro templates provide fast directional checks. Adjust intensity before applying.
            </Typography>
            <Grid container spacing={1.2} sx={{ mb: 1.4 }}>
              {(scenarios || []).map((s) => (
                <Grid key={s.scenario_id} item xs={12} md={6} lg={4}>
                  <MacroCard
                    scenario={s}
                    selected={s.scenario_id === macroId}
                    onSelect={() => {
                      setMacroId(s.scenario_id)
                      setMacroIntensity(s.default_intensity || 70)
                    }}
                  />
                </Grid>
              ))}
            </Grid>

            {selectedMacro ? (
              <Alert severity="info" sx={{ mb: 1.35 }}>
                <Typography variant="body2">
                  <strong>{selectedMacro.title}</strong>: {SCENARIO_EXPECTED[selectedMacro.scenario_id]?.effect || 'Scenario intent only; actual effect depends on conditions.'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  {SCENARIO_EXPECTED[selectedMacro.scenario_id]?.note || 'The model estimates impact from current live data.'}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.35 }}>
                  Directional template intent can invert under different baseline states. Results are model-based estimates, not guaranteed monotonic responses.
                </Typography>
                <Table size="small" sx={{ mt: 0.75 }}>
                  <TableBody>
                    <TableRow>
                      <TableCell sx={{ py: 0.4, px: 0.5, borderBottom: 'none', width: 140 }}>
                        Main variables
                      </TableCell>
                      <TableCell sx={{ py: 0.4, px: 0.5, borderBottom: 'none' }}>
                        {(selectedMacro.knobs || []).map((k) => labelFeature(k)).join(', ')}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Alert>
            ) : null}

            <Box
              sx={{
                mb: 1.35,
                py: 0.7,
                borderTop: '1px dashed rgba(12,61,86,0.2)',
                borderBottom: '1px dashed rgba(12,61,86,0.16)',
              }}
            >
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={0.85} alignItems={{ xs: 'flex-start', sm: 'center' }} justifyContent="space-between">
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
                    Intensity: {macroIntensity} ({intensityLabel(macroIntensity)})
                  </Typography>
                  {selectedMacro ? (
                    <Typography variant="caption" color="text.secondary">
                      Template default: {selectedMacro.default_intensity}
                    </Typography>
                  ) : null}
                </Stack>
                <Slider
                  value={macroIntensity}
                  min={0}
                  max={100}
                  valueLabelDisplay="auto"
                  onChange={(_, v) => setMacroIntensity(sliderNumber(v))}
                  sx={{ mt: 0.8 }}
                />
            </Box>

            <Button
              variant="contained"
              disabled={loading || !macroId || !canRunForecast}
              onClick={() =>
                onApply({
                  type: 'macro',
                  scenario_id: macroId,
                  intensity: macroIntensity
                })
              }
            >
              Apply Selected Scenario
            </Button>
          </Box>
        ) : (
          <Box>
            <Alert severity="info" sx={{ mb: 1.25 }}>
              Guided interventions apply category-level edits in live mode. Out-of-range edits are constrained for plausibility.
            </Alert>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.9 }}>
              Choose one row per category for cleaner interpretability. Strength controls are the main tuning signal.
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.2 }}>
              Guided mode uses a fixed baseline intensity ({GUIDED_BASE_INTENSITY}/100) so row-level choices remain comparable across runs.
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.2 }}>
              For direct variable-by-variable edits with baseline anchoring, use <strong>Forecast mode → Custom What-If</strong> in the setup panel.
            </Typography>

            <Divider sx={{ my: 1.25 }} />

            {hasDuplicateCategories ? (
              <Alert severity="warning" sx={{ mb: 1.2 }}>
                Duplicate categories detected. Keep one row per category before applying.
              </Alert>
            ) : null}

            <Stack spacing={1} sx={{ mb: 1.35 }}>
              {items.map((it, idx) => (
                <Box key={idx} sx={{ p: 0.8, borderBottom: '1px solid rgba(12,60,85,0.1)' }}>
                    <Grid container spacing={1} alignItems="center">
                      <Grid item xs={12} sm={4}>
                        <TextField
                          select
                          fullWidth
                          label="Category"
                          size="small"
                          value={it.category}
                          onChange={(e) => updateItem(idx, 'category', e.target.value)}
                        >
                          {categoryOptions.map((x) => (
                            <MenuItem
                              key={x}
                              value={x}
                              disabled={items.some((row, rowIdx) => rowIdx !== idx && row.category === x)}
                            >
                              {CATEGORY_LABELS[x]}
                            </MenuItem>
                          ))}
                        </TextField>
                      </Grid>
                      <Grid item xs={12} sm={3}>
                        <TextField
                          select
                          fullWidth
                          label="Direction"
                          size="small"
                          value={it.direction}
                          onChange={(e) => updateItem(idx, 'direction', e.target.value)}
                        >
                          {dirOptions.map((x) => (
                            <MenuItem key={x} value={x}>
                              {x.charAt(0).toUpperCase() + x.slice(1)}
                            </MenuItem>
                          ))}
                        </TextField>
                      </Grid>
                      <Grid item xs={12} sm={3}>
                        <TextField
                          select
                          fullWidth
                          label="Strength"
                          size="small"
                          value={it.magnitude}
                          onChange={(e) => updateItem(idx, 'magnitude', e.target.value)}
                        >
                          {magOptions.map((x) => (
                            <MenuItem key={x} value={x}>
                              {x.charAt(0).toUpperCase() + x.slice(1)}
                            </MenuItem>
                          ))}
                        </TextField>
                      </Grid>
                      <Grid item xs={12} sm={2}>
                        <Button color="error" onClick={() => removeItem(idx)} fullWidth>
                          Remove
                        </Button>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="caption" color="text.secondary">
                          Preview intent: {guidedRowPreview(it)}
                        </Typography>
                      </Grid>
                    </Grid>
                </Box>
              ))}
            </Stack>

            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                onClick={addItem}
                disabled={items.length >= Math.min(3, categoryOptions.length)}
              >
                Add change
              </Button>
              <Button
                variant="contained"
                disabled={loading || !canRunForecast || hasDuplicateCategories}
                onClick={() =>
                  onApply({
                    type: 'custom',
                    intensity: GUIDED_BASE_INTENSITY,
                    items
                  })
                }
              >
                Apply Guided Intervention
              </Button>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  )
}
