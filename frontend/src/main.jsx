import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CssBaseline, ThemeProvider } from '@mui/material'
import { createTheme, responsiveFontSizes } from '@mui/material/styles'
import App from './App'
import './styles.css'

const queryClient = new QueryClient()

let theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#0d6a94' },
    secondary: { main: '#4a6d98' },
    success: { main: '#1f7a52' },
    warning: { main: '#9d6a1a' },
    error: { main: '#a83232' },
    text: {
      primary: '#0f3550',
      secondary: '#3d5a70',
    },
    background: { default: '#e4eef7', paper: '#ffffff' },
    divider: 'rgba(13, 58, 82, 0.1)',
  },
  shape: {
    borderRadius: 10,
  },
  typography: {
    fontFamily: '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif',
    h3: { fontWeight: 900, letterSpacing: '-0.024em' },
    h4: { fontWeight: 900, letterSpacing: '-0.02em' },
    h5: { fontWeight: 900, letterSpacing: '-0.018em' },
    h6: { fontWeight: 800, letterSpacing: '-0.012em' },
    subtitle1: { fontWeight: 800, letterSpacing: '-0.012em', lineHeight: 1.28, color: '#0f3550' },
    subtitle2: { fontWeight: 700, color: '#14384d', lineHeight: 1.32 },
    body1: {
      lineHeight: 1.55,
      color: '#2a4a5f',
      fontSize: '0.97rem',
      '@media (max-width:600px)': { fontSize: '0.91rem' },
    },
    body2: {
      lineHeight: 1.55,
      color: '#35556b',
      fontSize: '0.93rem',
      '@media (max-width:600px)': { fontSize: '0.88rem' },
    },
    caption: {
      lineHeight: 1.42,
      color: '#4d6679',
      fontSize: '0.8rem',
      '@media (max-width:600px)': { fontSize: '0.74rem' },
    },
    overline: { fontWeight: 700, letterSpacing: '0.08em', fontSize: '0.68rem' },
    button: { fontWeight: 700, textTransform: 'none', letterSpacing: '0.01em' },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '::selection': {
          backgroundColor: 'rgba(11, 94, 131, 0.22)'
        }
      }
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          border: '1px solid rgba(13, 58, 82, 0.06)',
          borderRadius: 10,
          boxShadow: 'none',
          overflow: 'hidden',
          '&:before': { display: 'none' },
        },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: {
          minHeight: 48,
          '&.Mui-expanded': { minHeight: 48 },
        },
        content: { marginTop: 8, marginBottom: 8 },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid rgba(13, 58, 82, 0.1)',
          boxShadow: '0 10px 28px rgba(8, 35, 56, 0.08)',
          overflow: 'hidden',
          transition: 'box-shadow 180ms ease, border-color 180ms ease',
        },
      },
      variants: [
        {
          props: { variant: 'outlined' },
          style: {
            borderColor: 'rgba(8, 35, 56, 0.16)',
            boxShadow: '0 8px 18px rgba(8, 35, 56, 0.07)',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
          }
        }
      ]
    },
    MuiPaper: {
      styleOverrides: {
        rounded: { borderRadius: 12 },
        elevation1: { boxShadow: '0 8px 24px rgba(7, 42, 64, 0.06)' },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: 18,
          '&:last-child': {
            paddingBottom: 18
          },
          '@media (min-width:600px)': {
            padding: 22,
            '&:last-child': {
              paddingBottom: 22
            }
          }
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 9,
          letterSpacing: '0.01em',
          minHeight: 35,
          '@media (max-width:600px)': {
            minHeight: 33,
            fontSize: '0.83rem',
          },
        },
        sizeSmall: {
          minHeight: 31,
          paddingTop: 4,
          paddingBottom: 4,
        },
        contained: {
          boxShadow: '0 6px 14px rgba(7, 49, 71, 0.18)'
        }
      }
    },
    MuiToggleButtonGroup: {
      styleOverrides: {
        grouped: {
          borderRadius: 9,
          borderColor: 'rgba(8, 35, 56, 0.2)',
        }
      }
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          borderColor: 'rgba(8, 35, 56, 0.2)',
          color: '#36536a',
          '&.Mui-selected': {
            backgroundColor: 'rgba(11, 94, 131, 0.12)',
            color: '#0f4462',
            fontWeight: 700,
          },
        }
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
          letterSpacing: '0.005em',
          borderColor: 'rgba(11, 55, 82, 0.2)',
        },
        sizeSmall: {
          height: 23,
          '& .MuiChip-label': {
            paddingLeft: 7,
            paddingRight: 7,
          },
        }
      }
    },
    MuiTabs: {
      styleOverrides: {
        indicator: {
          height: 3,
          borderRadius: 3
        }
      }
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 700,
          minHeight: 42,
          paddingTop: 6,
          paddingBottom: 6,
          '@media (max-width:600px)': {
            minHeight: 36,
            fontSize: '0.79rem',
            paddingTop: 4,
            paddingBottom: 4,
          },
        }
      }
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 9,
          backgroundColor: 'rgba(255,255,255,0.96)',
          borderColor: 'rgba(12, 58, 83, 0.17)',
        }
      }
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: 'rgba(11, 55, 82, 0.14)',
        }
      }
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(8, 45, 69, 0.12)',
        }
      }
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(11, 94, 131, 0.07)'
        }
      }
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontWeight: 700
        }
      }
    }
  }
})
theme = responsiveFontSizes(theme, { factor: 2.2 })

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <App />
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>
)
