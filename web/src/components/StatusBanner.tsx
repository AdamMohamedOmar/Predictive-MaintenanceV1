import { T, FAULT_COLOR, FAULT_DISPLAY } from '../theme';

interface Props {
  label: string;
  confidence: number;
  anomalyScore: number;
  faultFraction?: number;
}

export default function StatusBanner({ label, confidence, anomalyScore, faultFraction }: Props) {
  const color = FAULT_COLOR[label] ?? T.TEXT_SECONDARY;
  const isHealthy = label === 'healthy';
  const lowConf = confidence < 0.6;
  const displayLabel = FAULT_DISPLAY[label] ?? label;

  return (
    <div style={{
      borderLeft: `4px solid ${color}`,
      background: `${color}12`,
      border: `1px solid ${color}40`,
      borderRadius: 4,
      padding: '16px 20px',
      marginBottom: 20,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        {/* Status dot */}
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, boxShadow: `0 0 8px ${color}` }} />
        <div>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 16, fontWeight: 700, color, letterSpacing: '0.05em' }}>
            {displayLabel.toUpperCase()}
          </span>
          {lowConf && (
            <span style={{ marginLeft: 10, fontFamily: T.FONT_MONO, fontSize: 10, color: T.ACCENT_WARN, border: `1px solid ${T.ACCENT_WARN}`, borderRadius: 3, padding: '2px 6px' }}>
              LOW CONFIDENCE
            </span>
          )}
          {!isHealthy && (
            <span style={{ marginLeft: 10, fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED }}>
              FAULT DETECTED
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 24 }}>
        <Stat label="CONFIDENCE" value={`${(confidence * 100).toFixed(0)}%`} color={lowConf ? T.ACCENT_WARN : T.TEXT_PRIMARY} />
        <Stat label="ANOMALY" value={anomalyScore.toFixed(3)} color={anomalyScore > 0.7 ? T.ACCENT_ALERT : anomalyScore > 0.4 ? T.ACCENT_WARN : T.ACCENT_OK} />
        {faultFraction != null && (
          <Stat label="FAULT WINDOWS" value={`${(faultFraction * 100).toFixed(0)}%`} color={faultFraction > 0.3 ? T.ACCENT_ALERT : T.TEXT_PRIMARY} />
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 2 }}>{label}</div>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 16, fontWeight: 700, color }}>{value}</div>
    </div>
  );
}
