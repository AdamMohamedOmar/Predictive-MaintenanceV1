// Design tokens — ported from src/dashboard/theme.py
// Single source of truth for the dark automotive-cockpit palette.

export const T = {
  // Backgrounds
  BG_BASE:    '#0d1117',
  BG_SURFACE: '#161b22',
  BG_RAISED:  '#21262d',

  // Borders
  BORDER:        '#30363d',
  BORDER_STRONG: '#58a6ff',

  // Text
  TEXT_PRIMARY:   '#e6edf3',
  TEXT_SECONDARY: '#8b949e',
  TEXT_MUTED:     '#484f58',

  // Accents
  ACCENT_OK:    '#3fb950',
  ACCENT_WARN:  '#d29922',
  ACCENT_ALERT: '#f85149',
  ACCENT_DATA:  '#58a6ff',
  ACCENT_INFO:  '#a371f7',

  // Fonts
  FONT_DISPLAY: '"JetBrains Mono", "Fira Code", monospace',
  FONT_BODY:    '"Inter", system-ui, sans-serif',
  FONT_MONO:    '"JetBrains Mono", "Fira Code", monospace',
} as const;

// Map fault labels to display colors
export const FAULT_COLOR: Record<string, string> = {
  healthy:                  T.ACCENT_OK,
  cold_start:               T.ACCENT_INFO,
  air_system:               T.ACCENT_ALERT,
  fuel_system:              T.ACCENT_ALERT,
  coolant_temp_sensor:      T.ACCENT_WARN,
  throttle_position_sensor: T.ACCENT_WARN,
  warming_up:               T.TEXT_SECONDARY,
};

export const FAULT_DISPLAY: Record<string, string> = {
  healthy:                  'Healthy',
  cold_start:               'Cold Start',
  air_system:               'Air System',
  fuel_system:              'Fuel System',
  coolant_temp_sensor:      'Coolant Sensor',
  throttle_position_sensor: 'TPS',
  warming_up:               'Warming up…',
};

export const INJECTABLE_FAULTS = [
  'air_system', 'fuel_system', 'coolant_temp_sensor', 'throttle_position_sensor',
];

export const DIAGNOSTIC_STEPS: Record<string, string[]> = {
  air_system: [
    'Inspect intake hose for cracks or loose clamps',
    'Check all vacuum lines (brake booster, PCV valve, MAP sensor hose)',
    'Inspect intake manifold gasket for leaks',
    'Smoke test the intake system to locate the leak',
    'Clear codes and re-test after repair',
  ],
  fuel_system: [
    'Check fuel pressure (target 300–350 kPa idle)',
    'Inspect injectors — use fuel-injector cleaner or flow test',
    'Replace fuel filter if >60k km since last change',
    'Check fuel pump delivery rate',
    'Inspect for exhaust leaks near the O₂ sensor',
  ],
  coolant_temp_sensor: [
    'Check coolant level and ECT sensor connector for corrosion',
    'Measure ECT resistance — should decrease as engine warms',
    'Compare sensor reading vs infrared thermometer',
    'Check wiring harness for chafing or shorts',
    'Replace ECT sensor if reading is stuck or implausible',
  ],
  throttle_position_sensor: [
    'Check TPS wiring and connector for looseness',
    'Measure TPS voltage: idle 0.5–0.8 V, WOT 4.5–4.8 V',
    'Clean throttle body and blade',
    'Perform throttle adaptation reset per workshop manual',
    'Replace TPS if voltage is erratic or out of range',
  ],
};
