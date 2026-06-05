import { T, FAULT_DISPLAY, INJECTABLE_FAULTS } from '../theme';

interface Props {
  severities: Record<string, number>;
  forecasts: Record<string, number>;
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ height: 6, background: T.BG_RAISED, borderRadius: 3, overflow: 'hidden' }}>
      <div style={{ width: `${Math.min(100, value * 100).toFixed(1)}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
    </div>
  );
}

function severityColor(v: number) {
  if (v > 0.7) return T.ACCENT_ALERT;
  if (v > 0.35) return T.ACCENT_WARN;
  return T.ACCENT_OK;
}

export default function SeverityGrid({ severities, forecasts }: Props) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12, marginBottom: 20 }}>
      {INJECTABLE_FAULTS.map(fault => {
        const cur = severities[fault] ?? 0;
        const fwd = forecasts[fault] ?? 0;
        return (
          <div key={fault} style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '14px 16px' }}>
            <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 10 }}>
              {FAULT_DISPLAY[fault]}
            </div>

            <div style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_SECONDARY, letterSpacing: '0.1em' }}>NOW</span>
                <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, fontWeight: 700, color: severityColor(cur) }}>
                  {(cur * 100).toFixed(0)}%
                </span>
              </div>
              <Bar value={cur} color={severityColor(cur)} />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED, letterSpacing: '0.1em' }}>60s FORECAST</span>
                <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.TEXT_SECONDARY }}>
                  {(fwd * 100).toFixed(0)}%
                </span>
              </div>
              <Bar value={fwd} color={`${severityColor(fwd)}88`} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
