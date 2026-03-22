import React, { useState } from 'react'
import {
  Typography,
  Box,
  TextField,
  Button,
  MenuItem,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
  Alert,
  Chip,
} from '@mui/material'
import { sk } from '../theme/tokens'

export default function ForecastControlsPanel({
  location,
  forecastMode,
  onForecastModeChange,
  locations = [],
  selectedLocationId = '',
  onLocationSelect,
  useManualLocation = false,
  onUseManualLocationChange,
  onLocationNameChange,
  locationDraft,
  setLocationDraft,
  coordinateErrors,
  coordinatesValid,
  onRunBaseline,
  loading,
}) {
  const [touched, setTouched] = useState({ lat: false, lon: false })
  const latError = Boolean(touched.lat && coordinateErrors?.lat)
  const lonError = Boolean(touched.lon && coordinateErrors?.lon)

  return (
    <Box
      className="controls-strip"
      sx={{
        px: { xs: 1.05, sm: 1.4, md: 1.65 },
        py: { xs: 1.15, sm: 1.35, md: 1.5 },
      }}
    >
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={{ xs: 1, md: 0.9 }}
        alignItems={{ xs: 'flex-start', md: 'center' }}
        justifyContent="space-between"
        sx={{ mb: 1.15 }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 800, color: sk.ink.title, fontSize: { xs: '1.04rem', sm: '1.12rem' } }}>
            Forecast Setup
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', mt: 0.25, color: sk.ink.muted, fontSize: '0.77rem' }}>
            Choose mode, set location, and run the next-hour PM2.5 forecast.
          </Typography>
        </Box>
        <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
          <Chip
            size="small"
            label={forecastMode === 'custom' ? 'Manual custom what-if mode' : 'Live forecast mode'}
            variant="outlined"
            sx={{ fontWeight: 700, backgroundColor: 'rgba(255,255,255,0.7)' }}
          />
          <Chip
            size="small"
            label={coordinatesValid ? 'Coordinates valid' : 'Coordinates required'}
            color={coordinatesValid ? 'success' : 'warning'}
            variant="outlined"
            sx={{ fontWeight: 700, backgroundColor: 'rgba(255,255,255,0.7)' }}
          />
        </Stack>
      </Stack>

      <Box
        className="controls-grid-shell"
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))', md: 'repeat(12, minmax(0, 1fr))' },
          columnGap: { xs: 0.95, sm: 1.15, md: 1.2 },
          rowGap: { xs: 0.95, sm: 1.05 },
          alignItems: 'start',
          py: { xs: 0.85, sm: 0.95, md: 1.05 },
          px: { xs: 0.85, sm: 1, md: 1.15 },
        }}
      >
          <Box sx={{ gridColumn: { xs: '1 / -1', md: 'span 3' } }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.42, fontWeight: 700 }}>
              Forecast mode
            </Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={forecastMode}
              onChange={(_, v) => {
                if (!v) return
                onForecastModeChange(v)
              }}
              sx={{ '& .MuiToggleButton-root': { minHeight: 42, px: 1.1, fontWeight: 700 } }}
            >
              <ToggleButton value="live" sx={{ px: 1.15, textTransform: 'none' }}>
                Live
              </ToggleButton>
              <ToggleButton value="custom" sx={{ px: 1.15, textTransform: 'none' }}>
                Custom What-If
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {!useManualLocation ? (
            <TextField
              size="small"
              select
              label="Saved location"
              value={selectedLocationId}
              onChange={(e) => onLocationSelect(e.target.value)}
              sx={{ gridColumn: { xs: '1 / -1', md: 'span 5' }, '& .MuiInputBase-root': { minHeight: 46 } }}
            >
              {locations.map((loc) => (
                <MenuItem key={loc.location_id} value={loc.location_id}>
                  {loc.name}, {loc.country}
                </MenuItem>
              ))}
            </TextField>
          ) : (
            <TextField
              size="small"
              label="Location name"
              value={location.name}
              onChange={(e) => onLocationNameChange(e.target.value)}
              sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, '& .MuiInputBase-root': { minHeight: 46 } }}
            />
          )}

          {useManualLocation ? (
            <>
              <TextField
                size="small"
                label="Latitude"
                value={locationDraft.lat}
                onChange={(e) => setLocationDraft((s) => ({ ...s, lat: e.target.value }))}
                onBlur={() => setTouched((s) => ({ ...s, lat: true }))}
                error={latError}
                helperText={latError ? coordinateErrors.lat : ''}
                sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, '& .MuiInputBase-root': { minHeight: 46 } }}
              />

              <TextField
                size="small"
                label="Longitude"
                value={locationDraft.lon}
                onChange={(e) => setLocationDraft((s) => ({ ...s, lon: e.target.value }))}
                onBlur={() => setTouched((s) => ({ ...s, lon: true }))}
                error={lonError}
                helperText={lonError ? coordinateErrors.lon : ''}
                sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, '& .MuiInputBase-root': { minHeight: 46 } }}
              />
            </>
          ) : null}

          <Stack
            direction="row"
            spacing={0.9}
            sx={{
              gridColumn: { xs: '1 / -1', md: useManualLocation ? 'span 2' : 'span 4' },
              justifyContent: { xs: 'space-between', sm: 'flex-start', md: 'flex-end' },
              alignItems: 'center',
              pt: { xs: 0.3, md: 0.25 },
              flexWrap: 'wrap',
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.35 }}>
              <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1, fontWeight: 700 }}>
                Location input type
              </Typography>
              <ToggleButtonGroup
                exclusive
                size="small"
                value={useManualLocation ? 'custom' : 'preset'}
                onChange={(_, v) => {
                  if (!v) return
                  onUseManualLocationChange(v === 'custom')
                }}
                sx={{ '& .MuiToggleButton-root': { minHeight: 42, px: 1.05, fontWeight: 700 } }}
              >
                <ToggleButton value="preset" sx={{ px: 1.15, textTransform: 'none' }}>
                  Preset
                </ToggleButton>
                <ToggleButton value="custom" sx={{ px: 1.15, textTransform: 'none' }}>
                  Enter coordinates
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>

            <Button
              variant="contained"
              onClick={onRunBaseline}
              disabled={loading || !coordinatesValid}
              sx={{
                minWidth: 156,
                fontWeight: 800,
                px: 2,
                minHeight: 42,
                borderRadius: 1.2,
              }}
            >
              Run Forecast
            </Button>
          </Stack>
      </Box>

      {forecastMode === 'custom' ? (
        <Alert severity="info" sx={{ mt: 0.8, mb: 0.2 }}>
          Custom What-If keeps historical context and applies your edited current conditions on top.
        </Alert>
      ) : null}

      <Typography
        variant="caption"
        color={coordinatesValid ? 'text.secondary' : 'error'}
        sx={{ display: 'block', mt: 0.85, fontWeight: coordinatesValid ? 500 : 600, fontSize: '0.78rem' }}
      >
        {coordinatesValid
          ? forecastMode === 'custom'
            ? 'Custom mode = baseline history context + your edited current conditions.'
            : 'Live mode = current Open-Meteo inputs + recent history context.'
          : 'Enter valid coordinates to run forecasts (latitude: -90 to 90, longitude: -180 to 180).'}
      </Typography>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.65, color: sk.ink.muted }}>
        Methodology caveats are available in the right sidebar under <strong>Methodology caveats</strong>.
      </Typography>
    </Box>
  )
}
