import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { T } from '../theme';

interface PidPoint { elapsed_s: number; [pid: string]: number }
interface Props { data: PidPoint[] }

const PANEL_PIDS = [
  { key: 'ENGINE_RPM', label: 'RPM', color: T.ACCENT_DATA, unit: 'rpm' },
  { key: 'COOLANT_TEMPERATURE', label: 'Coolant', color: T.ACCENT_WARN, unit: '°C' },
  { key: 'LONG_TERM_FUEL_TRIM_BANK_1', label: 'LTFT', color: T.ACCENT_OK, unit: '%' },
  { key: 'THROTTLE', label: 'Throttle', color: T.ACCENT_INFO, unit: '%' },
];

const tip = {
  contentStyle: { background: T.BG_RAISED, border: `1px solid ${T.BORDER}`, fontFamily: T.FONT_MONO, fontSize: 11 },
  labelStyle: { color: T.TEXT_MUTED },
};

function Chart({ pidA, pidB, data }: { pidA: (typeof PANEL_PIDS)[0]; pidB: (typeof PANEL_PIDS)[0]; data: PidPoint[] }) {
  return (
    <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '12px 16px', flex: 1 }}>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED, letterSpacing: '0.12em', marginBottom: 8 }}>
        {pidA.label} ({pidA.unit}) · {pidB.label} ({pidB.unit})
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
          <XAxis dataKey="elapsed_s" tick={{ fontSize: 9, fill: T.TEXT_MUTED, fontFamily: T.FONT_MONO }} tickLine={false} />
          <YAxis tick={{ fontSize: 9, fill: T.TEXT_MUTED, fontFamily: T.FONT_MONO }} tickLine={false} axisLine={false} />
          <Tooltip {...tip} />
          <Legend wrapperStyle={{ fontFamily: T.FONT_MONO, fontSize: 9 }} />
          <Line type="monotone" dataKey={pidA.key} name={pidA.label} stroke={pidA.color} dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey={pidB.key} name={pidB.label} stroke={pidB.color} dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function PidStrip({ data }: Props) {
  const hasData = data.length > 0;

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 10 }}>
        Live PID Readings
      </div>
      {!hasData ? (
        <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '24px', textAlign: 'center', color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 12 }}>
          PID timeline — available in live sessions
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 12 }}>
          <Chart pidA={PANEL_PIDS[0]} pidB={PANEL_PIDS[1]} data={data} />
          <Chart pidA={PANEL_PIDS[2]} pidB={PANEL_PIDS[3]} data={data} />
        </div>
      )}
    </div>
  );
}
