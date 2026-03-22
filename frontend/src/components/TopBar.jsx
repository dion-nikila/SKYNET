import React from 'react'
import { Typography, Box, Chip, Stack } from '@mui/material'
import { sk } from '../theme/tokens'
import skynetLogo from '../assets/logo.png'

export default function TopBar({ forecastMode, modelInfo, locationName = '' }) {
  return (
    <Box
      sx={{
        position: 'sticky',
        top: 0,
        zIndex: 1200,
        borderBottom: '1px solid rgba(255,255,255,0.16)',
        background: sk.topBar,
        backdropFilter: 'blur(10px) saturate(1.12)',
        boxShadow: '0 14px 34px rgba(5, 28, 45, 0.24)',
      }}
    >
      <Box
        sx={{
          px: { xs: 0.9, sm: 1.8, md: 2.6, xl: 3.2 },
          py: { xs: 1.15, sm: 1.95, md: 2.15 },
          width: '100%',
          maxWidth: { xs: '100%', sm: `min(${sk.maxWidth}px, calc(100vw - 20px))` },
          mx: 'auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: { xs: 1.05, sm: 1.8 },
          flexWrap: 'wrap',
        }}
      >
        <Stack direction="row" spacing={{ xs: 0.9, sm: 1.2 }} alignItems="center" sx={{ minWidth: 0 }}>
          <Box
            sx={{
              width: { xs: 40, sm: 54 },
              height: { xs: 40, sm: 54 },
              borderRadius: 1.6,
              background: 'linear-gradient(145deg, rgba(230,247,255,0.92) 0%, rgba(193,232,252,0.82) 100%)',
              display: 'grid',
              placeItems: 'center',
              p: 0.18,
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.8), 0 4px 10px rgba(5,27,45,0.2)',
              overflow: 'hidden',
            }}
          >
            <Box
              component="img"
              src={skynetLogo}
              alt="SKYNET logo"
              sx={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                objectPosition: 'center',
                borderRadius: 1.3,
              }}
            />
          </Box>

          <Box sx={{ minWidth: 0 }}>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 900,
                color: '#eef8ff',
                lineHeight: 1.06,
                letterSpacing: '0.01em',
                fontSize: { xs: '1.03rem', sm: '1.58rem', md: '1.72rem' },
              }}
            >
              SKYNET, PM2.5 Forecasting System
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: 'rgba(219,239,252,0.9)',
                display: 'block',
                mt: 0.34,
                maxWidth: 820,
                fontSize: { xs: '0.72rem', sm: '0.87rem' },
                lineHeight: 1.38,
              }}
            >
              One-hour-ahead PM2.5 decision-support forecasting with explainability, intervention simulation, and reliability diagnostics.
            </Typography>
          </Box>
        </Stack>

        <Stack
          direction="row"
          spacing={0.8}
          useFlexGap
          flexWrap="wrap"
          alignItems="center"
          sx={{ width: { xs: '100%', sm: 'auto' }, justifyContent: { xs: 'flex-start', sm: 'flex-end' } }}
        >
          <Chip
            size="small"
            label="Horizon: +1 hour"
            variant="outlined"
            sx={{
              color: '#e8f6ff',
              border: '1px solid rgba(214,239,255,0.36)',
              backgroundColor: 'rgba(219,242,255,0.08)',
              fontWeight: 700,
              height: { xs: 24, sm: 28 },
              '& .MuiChip-label': { px: { xs: 0.85, sm: 1.15 } },
            }}
          />
          <Chip
            size="small"
            label="Research Prototype"
            variant="outlined"
            sx={{
              color: '#e8f6ff',
              border: '1px solid rgba(214,239,255,0.36)',
              backgroundColor: 'rgba(219,242,255,0.08)',
              fontWeight: 700,
              display: { xs: 'none', sm: 'inline-flex' },
              height: { xs: 24, sm: 28 },
              '& .MuiChip-label': { px: { xs: 0.85, sm: 1.15 } },
            }}
          />
          <Chip
            size="small"
            label={forecastMode === 'custom' ? 'Mode: Custom What-If' : 'Mode: Live Forecast'}
            variant="outlined"
            sx={{
              color: '#e8f6ff',
              border: '1px solid rgba(214,239,255,0.36)',
              backgroundColor: forecastMode === 'custom' ? 'rgba(176,122,255,0.15)' : 'rgba(40,184,238,0.12)',
              fontWeight: 700,
              height: { xs: 24, sm: 28 },
              '& .MuiChip-label': { px: { xs: 0.85, sm: 1.15 } },
            }}
          />
          {locationName ? (
            <Chip
              size="small"
              label={locationName}
              variant="outlined"
              sx={{
                color: '#def2ff',
                border: '1px solid rgba(214,239,255,0.36)',
                backgroundColor: 'rgba(219,242,255,0.06)',
                maxWidth: { xs: '100%', sm: 240 },
                height: { xs: 24, sm: 28 },
                '& .MuiChip-label': { px: { xs: 0.8, sm: 1.1 } },
              }}
            />
          ) : null}
          {modelInfo?.model_version ? (
            <Chip
              size="small"
              label={modelInfo.model_version}
              variant="outlined"
              sx={{
                color: '#def2ff',
                border: '1px solid rgba(214,239,255,0.36)',
                backgroundColor: 'rgba(219,242,255,0.06)',
                fontWeight: 700,
                display: { xs: 'none', md: 'inline-flex' },
                height: { xs: 24, sm: 28 },
                '& .MuiChip-label': { px: { xs: 0.8, sm: 1.1 } },
              }}
            />
          ) : null}
        </Stack>
      </Box>
    </Box>
  )
}
