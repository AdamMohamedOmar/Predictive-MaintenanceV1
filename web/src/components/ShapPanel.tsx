import { T } from '../theme';

interface Props { topShap: [string, number][] | undefined }

function humanize(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/__mean$/, ' mean').replace(/__std$/, ' std').replace(/__min$/, ' min')
    .replace(/__max$/, ' max').replace(/__delta$/, ' delta')
    .replace(/ENGINE RPM/gi, 'RPM')
    .replace(/COOLANT TEMPERATURE/gi, 'Coolant')
    .replace(/LONG TERM FUEL TRIM BANK 1/gi, 'LTFT')
    .replace(/SHORT TERM FUEL TRIM BANK 1/gi, 'STFT')
    .replace(/INTAKE MANIFOLD PRESSURE/gi, 'MAP')
    .replace(/INTAKE AIR TEMPERATURE/gi, 'IAT')
    .replace(/VEHICLE SPEED/gi, 'Speed')
    .replace(/ENGINE LOAD/gi, 'Load')
    .replace(/THROTTLE/gi, 'Throttle')
    .replace(/TIMING ADVANCE/gi, 'Timing')
    .replace(/CONTROL MODULE VOLTAGE/gi, 'Voltage')
    .toLowerCase()
    .replace(/^./, c => c.toUpperCase());
}

export default function ShapPanel({ topShap }: Props) {
  if (!topShap || topShap.length === 0) {
    return (
      <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '20px', marginBottom: 20, color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 12, textAlign: 'center' }}>
        SHAP — available in live sessions
      </div>
    );
  }

  const maxAbs = Math.max(...topShap.map(([, v]) => Math.abs(v)), 0.001);

  return (
    <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '16px 18px', marginBottom: 20 }}>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 12 }}>
        Top SHAP Features
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {topShap.slice(0, 6).map(([name, val], i) => {
          const pct = (Math.abs(val) / maxAbs) * 100;
          const pos = val >= 0;
          return (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 56px', alignItems: 'center', gap: 10 }}>
              <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.TEXT_SECONDARY, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {humanize(name)}
              </div>
              <div style={{ height: 6, background: T.BG_RAISED, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${pct.toFixed(1)}%`, height: '100%', background: pos ? T.ACCENT_ALERT : T.ACCENT_DATA, borderRadius: 3 }} />
              </div>
              <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: pos ? T.ACCENT_ALERT : T.ACCENT_DATA, textAlign: 'right' }}>
                {val > 0 ? '+' : ''}{val.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
