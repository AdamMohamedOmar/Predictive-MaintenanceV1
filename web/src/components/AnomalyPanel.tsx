import { T } from '../theme';

interface Props { score: number }

export default function AnomalyPanel({ score }: Props) {
  const color = score > 0.7 ? T.ACCENT_ALERT : score > 0.4 ? T.ACCENT_WARN : T.ACCENT_OK;
  const label = score > 0.7 ? 'HIGH' : score > 0.4 ? 'ELEVATED' : 'NORMAL';

  return (
    <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '16px 20px', marginBottom: 20 }}>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 14 }}>
        One-class anomaly detector · IsolationForest
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <div>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 28, fontWeight: 700, color }}>{score.toFixed(3)}</span>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 10, color, marginLeft: 8, padding: '2px 7px', border: `1px solid ${color}`, borderRadius: 3 }}>{label}</span>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ height: 8, background: T.BG_RAISED, borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
            {/* Threshold marker at 0.85 */}
            <div style={{ position: 'absolute', left: '85%', top: 0, bottom: 0, width: 1, background: T.ACCENT_ALERT, opacity: 0.5, zIndex: 1 }} />
            <div style={{ width: `${(score * 100).toFixed(1)}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.4s ease' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            <span style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED }}>0.0 normal</span>
            <span style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.ACCENT_ALERT }}>0.85 threshold</span>
            <span style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED }}>1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}
