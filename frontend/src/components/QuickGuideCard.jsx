import React from 'react'
import { Typography, Chip, Box, Stack, Accordion, AccordionSummary, AccordionDetails } from '@mui/material'
import { sk } from '../theme/tokens'

const LIVE_STEPS = [
  { title: 'Run baseline', text: 'Capture the next-hour baseline forecast for the selected location.' },
  { title: 'Apply intervention', text: 'Use macro templates or guided controls to test directional changes.' },
  { title: 'Review evidence', text: 'Read comparison, explanation, reliability guidance, and uncertainty context together.' },
]

const CUSTOM_STEPS = [
  { title: 'Set custom values', text: 'Edit one or more current-condition variables in Custom What-If mode.' },
  { title: 'Run custom forecast', text: 'Generate baseline-anchored output using your edited conditions.' },
  { title: 'Compare and verify', text: 'Use change view, explanation, and diagnostics to assess plausibility.' },
]

const accordionSx = {
  border: 'none',
  borderRadius: 1,
  background: 'rgba(13, 58, 82, 0.04)',
  boxShadow: 'none',
  overflow: 'hidden',
  '&::before': { display: 'none' },
}

function StepList({ steps }) {
  return (
    <Stack spacing={1.15}>
      {steps.map((s, idx) => (
        <Stack key={s.title} direction="row" spacing={1.15} alignItems="flex-start" sx={{ minWidth: 0 }}>
          <Box
            aria-hidden
            sx={{
              flexShrink: 0,
              width: 28,
              height: 28,
              borderRadius: '50%',
              display: 'grid',
              placeItems: 'center',
              fontSize: '0.8125rem',
              fontWeight: 800,
              bgcolor: idx === 2 ? 'rgba(37, 120, 80, 0.14)' : 'rgba(14, 93, 130, 0.08)',
              color: idx === 2 ? 'success.dark' : 'primary.dark',
              border: '1px solid rgba(13, 58, 82, 0.1)',
            }}
          >
            {idx + 1}
          </Box>
          <Box sx={{ minWidth: 0, pt: 0.1 }}>
            <Typography variant="body2" sx={{ fontWeight: 700, color: sk.ink.strong, lineHeight: 1.35 }}>
              {s.title}
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', mt: 0.35, color: sk.ink.muted, lineHeight: 1.45 }}>
              {s.text}
            </Typography>
          </Box>
        </Stack>
      ))}
    </Stack>
  )
}

export default function QuickGuideCard({ forecastMode = 'live', embedded = false }) {
  const isCustom = forecastMode === 'custom'
  const steps = isCustom ? CUSTOM_STEPS : LIVE_STEPS

  const summary = (
    <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1} sx={{ width: '100%', pr: 0.5 }}>
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, color: sk.ink.title, letterSpacing: '-0.01em' }}>
          Quick Start Guide
        </Typography>
        <Typography variant="caption" sx={{ color: sk.ink.muted, display: 'block', mt: 0.2 }}>
          {embedded ? 'Expand to show or hide the 3-step walkthrough.' : '3-step walkthrough.'}
        </Typography>
      </Box>
      <Chip
        size="small"
        color={isCustom ? 'warning' : 'primary'}
        variant="outlined"
        label={isCustom ? 'Custom mode' : 'Live mode'}
        sx={{ flexShrink: 0, fontWeight: 700 }}
      />
    </Stack>
  )

  return (
    <Accordion
      defaultExpanded={!embedded}
      disableGutters
      elevation={0}
      sx={{
        ...accordionSx,
        ...(embedded
          ? {}
          : {
              borderRadius: 2,
              p: 0,
            }),
      }}
    >
      <AccordionSummary
        expandIcon={<Typography sx={{ fontSize: 15, opacity: 0.55, lineHeight: 1, color: sk.ink.muted }}>▾</Typography>}
        sx={{
          px: embedded ? 1.1 : 1.35,
          py: 0.65,
          minHeight: 52,
          '& .MuiAccordionSummary-content': { my: 0.75 },
        }}
      >
        {summary}
      </AccordionSummary>
      <AccordionDetails sx={{ px: embedded ? 1.1 : 1.35, pt: 0.35, pb: 1.5 }}>
        <StepList steps={steps} />
      </AccordionDetails>
    </Accordion>
  )
}
