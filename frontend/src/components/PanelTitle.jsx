import React from 'react'
import { Box, Typography } from '@mui/material'
import { sk } from '../theme/tokens'

export default function PanelTitle({ title, subtitle = '', sx = {} }) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 1,
        mb: 1.25,
        ...sx,
      }}
    >
      <Box>
        <Typography
          variant="h6"
          sx={{
            lineHeight: 1.2,
            fontWeight: 800,
            letterSpacing: '-0.015em',
            color: sk.ink.title,
            fontSize: { xs: '1.04rem', sm: '1.14rem' },
          }}
        >
          {title}
        </Typography>
        {subtitle ? (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ display: 'block', mt: 0.3, fontSize: { xs: '0.8rem', sm: '0.84rem' } }}
          >
            {subtitle}
          </Typography>
        ) : null}
      </Box>
    </Box>
  )
}
