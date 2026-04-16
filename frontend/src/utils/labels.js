export const FEATURE_LABELS = {
  'PM2.5_current': 'Current PM2.5',
  lag1: 'PM2.5 one hour ago',
  lag3: 'PM2.5 three hours ago',
  lag6: 'PM2.5 six hours ago',
  lag12: 'PM2.5 twelve hours ago',
  lag24: 'PM2.5 one day ago',
  lag48: 'PM2.5 two days ago',
  lag72: 'PM2.5 three days ago',
  lag168: 'PM2.5 one week ago',
  roll3: '3-hour PM2.5 average',
  roll6: '6-hour PM2.5 average',
  roll24: '24-hour PM2.5 average',
  roll48: '48-hour PM2.5 average',
  roll168: '7-day PM2.5 average',
  std6: 'PM2.5 variation (6h)',
  std24: 'PM2.5 variation (24h)',
  min24: 'Lowest PM2.5 in last 24h',
  max24: 'Highest PM2.5 in last 24h',
  ewm6: 'Recent PM2.5 trend (6h)',
  ewm24: 'Recent PM2.5 trend (24h)',
  trend_24: 'Change since yesterday',
  trend_168: 'Change since last week',
  roll_diff_3_24: 'Short vs daily PM2.5 trend',
  roll_diff_24_168: 'Daily vs weekly PM2.5 trend',
  PM10: 'PM10 concentration',
  NO2: 'NO2 concentration',
  SO2: 'SO2 concentration',
  O3: 'Ozone concentration',
  CO: 'Carbon monoxide concentration',
  temperature: 'Air temperature',
  humidity: 'Humidity',
  pressure: 'Air pressure',
  wind_speed: 'Wind speed',
  sin_hour: 'Hour-of-day cycle',
  cos_hour: 'Hour-of-day cycle',
  sin_day: 'Day-of-week cycle',
  cos_day: 'Day-of-week cycle',
  is_weekend: 'Weekend pattern'
}

export const CATEGORY_LABELS = {
  wind: 'Wind',
  humidity: 'Humidity',
  temperature: 'Temperature',
  emission_proxy: 'Traffic/Emissions'
}

export const SCENARIO_LABELS = {
  baseline: 'Baseline Forecast',
  traffic_gridlock: 'Traffic Gridlock',
  strong_dispersion: 'Strong Dispersion',
  heavy_rainstorm: 'Strong Dispersion',
  heatwave: 'Heatwave',
  dust_resuspension: 'Dust Resuspension',
  windy_dispersion: 'Dust Resuspension',
  trapped_pollution: 'Trapped Pollution',
  stagnation: 'Trapped Pollution',
  industrial_plume: 'Industrial Source Loading',
  guided_intervention: 'Guided Intervention',
  custom_what_if: 'Custom What-If',
  custom: 'Custom Scenario'
}

export function labelFeature(name) {
  return FEATURE_LABELS[name] || name
}

export function labelScenario(id) {
  return SCENARIO_LABELS[id] || id
}
