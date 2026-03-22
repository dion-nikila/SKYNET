import React from 'react'
import {
  Typography,
  TableContainer,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Button,
  Chip,
  Alert,
  Box,
  Stack,
} from '@mui/material'
import { labelScenario } from '../utils/labels'
import { sk } from '../theme/tokens'

function shortRunId(id) {
  const s = String(id || '')
  if (s.length <= 22) return s
  return `${s.slice(0, 20)}...`
}

function formatWhen(ts) {
  if (!ts) return '-'
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return String(ts)
  return d.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function signed(v, digits = 2) {
  const n = Number(v || 0)
  return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}`
}

function scenarioModeFromRow(row) {
  const explicit = String(row?.scenario_mode || '').toLowerCase()
  if (['macro', 'guided_intervention', 'manual_custom', 'baseline'].includes(explicit)) {
    return explicit
  }
  const mode = String(row?.forecast_mode || '').toLowerCase()
  const sid = String(row?.scenario_id || '')
  const intensity = Number(row?.intensity || 0)
  if (mode === 'custom' || sid === 'custom_what_if') return 'manual_custom'
  if (sid === 'guided_intervention') return 'guided_intervention'
  if (sid === 'baseline') return 'baseline'
  if (mode === 'live' && sid === 'custom' && intensity > 0) return 'guided_intervention'
  if (mode === 'live' && sid === 'custom' && intensity <= 0) return 'baseline'
  if (!sid) return 'baseline'
  return 'macro'
}

function scenarioLabelForRow(row) {
  const mode = scenarioModeFromRow(row)
  if (mode === 'guided_intervention') return labelScenario('guided_intervention')
  if (mode === 'manual_custom') return labelScenario('custom_what_if')
  if (mode === 'baseline') return labelScenario('baseline')
  return labelScenario(row?.scenario_id || 'baseline')
}

export default function RunHistoryPanel({
  runs,
  onLoadRun,
  onRerun,
  onDelete,
  onClearAll,
  loading,
  loadingRunId = null,
  activeRunId = null,
}) {
  const rows = Array.isArray(runs) ? runs : []

  return (
    <Box sx={{ py: { xs: 1.1, sm: 1.25 } }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title, mb: 0.1 }}>
          Recent Local Runs
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.62 }}>
          Stored on this browser/device only; backend/server run logs are separate and not shown here.
        </Typography>

        {rows.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            No local runs yet. Run a baseline or scenario and it will appear here.
          </Typography>
        ) : (
          <>
            <Alert severity="info" sx={{ mb: 1 }}>
              Use <strong>Open</strong> to view a past result snapshot or <strong>Re-run</strong> to execute that saved request again.
            </Alert>
            <Stack direction="row" spacing={0.85} sx={{ mb: 0.95, flexWrap: 'wrap' }}>
              <Button size="small" variant="outlined" color="inherit" onClick={onClearAll} disabled={loading}>
                Clear all local runs
              </Button>
              <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
                {rows.length} stored run{rows.length === 1 ? '' : 's'}
              </Typography>
            </Stack>
          </>
        )}

        <TableContainer sx={{ overflowX: 'auto', border: '1px solid rgba(16,42,67,0.1)', borderRadius: 1 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ display: { xs: 'none', md: 'table-cell' } }}>Run ID</TableCell>
                <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>Time</TableCell>
                <TableCell sx={{ display: { xs: 'none', md: 'table-cell' } }}>Mode</TableCell>
                <TableCell>Scenario</TableCell>
                <TableCell sx={{ display: { xs: 'none', md: 'table-cell' } }}>Location</TableCell>
                <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>Intensity</TableCell>
                <TableCell>Next-hour PM2.5</TableCell>
                <TableCell>Change vs baseline</TableCell>
                <TableCell>Range check</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((r) => (
                <TableRow
                  key={r.run_id}
                  sx={
                    activeRunId && r.run_id === activeRunId
                      ? { backgroundColor: 'rgba(19, 93, 119, 0.07)' }
                      : undefined
                  }
                >
                  <TableCell sx={{ display: { xs: 'none', md: 'table-cell' } }} title={r.run_id}>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {shortRunId(r.run_id)}
                    </Typography>
                  </TableCell>
                  <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>
                    <Typography variant="caption" color="text.secondary">
                      {formatWhen(r.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                    <Chip size="small" variant="outlined" label={String(r.forecast_mode || 'live')} />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" noWrap>
                      {scenarioLabelForRow(r)}
                    </Typography>
                  </TableCell>
                  <TableCell sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                    <Typography variant="caption" color="text.secondary" noWrap>
                      {String(r.location_name || '-')} {Number.isFinite(Number(r.location_lat)) && Number.isFinite(Number(r.location_lon)) ? `(${Number(r.location_lat).toFixed(2)}, ${Number(r.location_lon).toFixed(2)})` : ''}
                    </Typography>
                  </TableCell>
                  <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>
                    {scenarioModeFromRow(r) === 'baseline'
                      ? '-'
                      : scenarioModeFromRow(r) === 'manual_custom'
                        ? 'Manual'
                        : scenarioModeFromRow(r) === 'guided_intervention'
                          ? 'Guided'
                        : Number(r.intensity || 0)}
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 700 }}>{Number(r.pm25_t_plus_1 || 0).toFixed(2)}</Typography>
                  </TableCell>
                  <TableCell>
                    <Box
                      component="span"
                      sx={{
                        fontWeight: 700,
                        color: Number(r.pm25_change || 0) >= 0 ? '#9a2f2f' : '#1f7a58',
                      }}
                    >
                      {signed(r.pm25_change)}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip size="small" color={r.ood_flag ? 'warning' : 'success'} label={r.ood_flag ? 'check' : 'in range'} />
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={0.8} justifyContent="flex-end" sx={{ flexWrap: 'wrap' }}>
                      <Button
                        size="small"
                        variant={activeRunId && r.run_id === activeRunId ? 'contained' : 'outlined'}
                        onClick={() => onLoadRun(r.run_id)}
                        disabled={loading}
                      >
                        {loadingRunId === r.run_id ? 'Loading...' : activeRunId === r.run_id ? 'Opened' : 'Open'}
                      </Button>
                      <Button size="small" variant="outlined" onClick={() => onRerun(r.run_id)} disabled={loading}>
                        Re-run
                      </Button>
                      <Button size="small" color="inherit" onClick={() => onDelete(r.run_id)} disabled={loading}>
                        Delete
                      </Button>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
    </Box>
  )
}
