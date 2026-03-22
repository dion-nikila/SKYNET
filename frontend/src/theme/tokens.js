/**
 * Shared visual tokens for the SKYNET dashboard (presentation only).
 */
export const sk = {
  maxWidth: 1720,
  radius: { xs: 1.25, sm: 1.5, md: 2, lg: 2.5 },
  border: '1px solid rgba(13, 58, 82, 0.11)',
  /** Use for outer page sections — barely visible edge */
  borderMuted: '1px solid rgba(13, 58, 82, 0.055)',
  divider: 'rgba(13, 58, 82, 0.065)',
  ink: {
    title: '#0c3349',
    strong: '#14384d',
    body: '#2a4a5f',
    muted: '#4d6679',
  },
  surface: {
    card: 'linear-gradient(180deg, rgba(255,255,255,0.985) 0%, rgba(248,251,254,0.96) 100%)',
    rail: 'linear-gradient(180deg, rgba(255,255,255,0.93) 0%, rgba(241,248,252,0.9) 100%)',
    wash: 'linear-gradient(180deg, rgba(255,255,255,0.58) 0%, rgba(250,252,254,0.42) 100%)',
    shadow: '0 1px 0 rgba(255,255,255,0.88) inset, 0 10px 36px rgba(7, 42, 64, 0.05)',
    shadowSoft: '0 8px 28px rgba(7, 42, 64, 0.045)',
    /** Single workspace row — one gentle slab, not stacked cards */
    workspaceMerge: 'rgba(255, 255, 255, 0.52)',
  },
  topBar: 'linear-gradient(115deg, rgba(8,46,70,0.98) 0%, rgba(12,72,102,0.96) 48%, rgba(16,95,122,0.93) 100%)',
  spotlightBar: 'linear-gradient(115deg, rgba(9,50,76,0.96) 0%, rgba(13,78,108,0.94) 50%, rgba(16,98,128,0.9) 100%)',
}

/** One horizontal band: left analysis + right rail share top alignment and padding rhythm */
export const workspaceRowSx = {
  display: 'flex',
  flexDirection: { xs: 'column', lg: 'row' },
  alignItems: 'stretch',
  minWidth: 0,
  borderRadius: { xs: 2, lg: 2.5 },
  bgcolor: sk.surface.workspaceMerge,
  border: sk.borderMuted,
  overflow: 'hidden',
}

export const workspaceLeftColSx = {
  flex: 1,
  minWidth: 0,
  pt: { xs: 1.35, sm: 1.5 },
  pb: { xs: 1.35, sm: 1.55 },
  pl: { xs: 1.35, sm: 1.6, md: 1.75 },
  pr: { xs: 1.35, sm: 1.6, md: 1.85 },
  borderRight: { lg: `1px solid ${sk.divider}` },
}

export const workspaceRightColSx = {
  width: { xs: '100%', lg: '33%' },
  maxWidth: { lg: 430 },
  minWidth: 0,
  pt: { xs: 1.35, sm: 1.5 },
  pb: { xs: 1.35, sm: 1.55 },
  pl: { xs: 1.35, sm: 1.6, md: 1.65 },
  pr: { xs: 1.35, sm: 1.6, md: 1.75 },
  borderTop: { xs: `1px solid ${sk.divider}`, lg: 'none' },
}

export const workspaceInnerSx = {
  px: { xs: 0, sm: 0 },
  py: { xs: 0, sm: 0 },
}
