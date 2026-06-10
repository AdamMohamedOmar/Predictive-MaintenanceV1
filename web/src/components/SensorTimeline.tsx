import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Brush } from 'recharts';
import { T } from '../theme';
import { USEFUL_PIDS } from '../pids';

export interface TimelineAlert { elapsed_s: number; label: string }
interface Props {
  rows: Record<string, number>[];
  alerts: TimelineAlert[];
  withBrush?: boolean;
}

const DEFAULT_SEL = ['ENGINE_RPM', 'INTAKE_MANIFOLD_PRESSURE',
  'SHORT_TERM_FUEL_TRIM_BANK_1', 'LONG_TERM_FUEL_TRIM_BANK_1', 'COOLANT_TEMPERATURE'];
const COLORS = [T.ACCENT_DATA, T.ACCENT_WARN, T.ACCENT_OK, T.ACCENT_INFO,
  '#b88aff', '#ff9a6b', '#6bd6ff', '#ffd66b'];
const LS_KEY = 'pm_timeline_pids';

export default function SensorTimeline({ rows, alerts, withBrush = false }: Props) {
  const [selected, setSelected] = useState<string[]>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(LS_KEY) ?? 'null');
      if (Array.isArray(s) && s.length) return s;
    } catch { /* corrupted localStorage — fall through */ }
    return DEFAULT_SEL;
  });
  const [cursor, setCursor] = useState<number | null>(null);

  const toggle = (pid: string) => setSelected(prev => {
    const next = prev.includes(pid) ? prev.filter(p => p !== pid) : [...prev, pid];
    localStorage.setItem(LS_KEY, JSON.stringify(next));
    return next.length ? next : prev;
  });

  // Per-series min-max normalisation — RPM (0-3000) and trims (±25%) can't share a raw axis.
  // Exact raw values live in the click-readout panel below.
  const data = useMemo(() => {
    const ranges: Record<string, [number, number]> = {};
    for (const pid of selected) {
      let lo = Infinity, hi = -Infinity;
      for (const r of rows) {
        const v = r[pid];
        if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      }
      ranges[pid] = lo === Infinity ? [0, 1] : [lo, hi === lo ? lo + 1 : hi];
    }
    return rows.map(r => {
      const d: Record<string, number> = { elapsed_s: r.elapsed_s };
      for (const pid of selected) {
        const v = r[pid]; const [lo, hi] = ranges[pid];
        if (Number.isFinite(v)) d[pid] = (v - lo) / (hi - lo);
      }
      return d;
    });
  }, [rows, selected]);

  const readout = useMemo(() => {
    if (cursor == null || !rows.length) return null;
    let best = rows[0];
    for (const r of rows)
      if (Math.abs(r.elapsed_s - cursor) < Math.abs(best.elapsed_s - cursor)) best = r;
    const alert = alerts.find(a => Math.abs(a.elapsed_s - best.elapsed_s) <= 5);
    return { row: best, alert };
  }, [cursor, rows, alerts]);

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
        {USEFUL_PIDS.map(pid => (
          <button key={pid} onClick={() => toggle(pid)} style={{
            fontFamily: T.FONT_MONO, fontSize: 9, padding: '3px 7px', cursor: 'pointer',
            border: `1px solid ${selected.includes(pid) ? T.ACCENT_DATA : T.BORDER}`,
            background: selected.includes(pid) ? `${T.ACCENT_DATA}22` : 'transparent',
            color: selected.includes(pid) ? T.TEXT_PRIMARY : T.TEXT_MUTED,
          }}>{pid.replace(/_/g, ' ')}</button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} onClick={(e) => { if (e && e.activeLabel != null) setCursor(Number(e.activeLabel)); }}>
          <XAxis dataKey="elapsed_s" tickFormatter={fmt}
                 stroke={T.TEXT_MUTED} fontSize={10} fontFamily={T.FONT_MONO} />
          <YAxis domain={[0, 1]} hide />
          <Tooltip
            labelFormatter={(v) => `t = ${fmt(Number(v))}`}
            contentStyle={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, fontFamily: T.FONT_MONO, fontSize: 10 }}
          />
          {selected.map((pid, i) => (
            <Line key={pid} dataKey={pid} dot={false} strokeWidth={1.5}
                  stroke={COLORS[i % COLORS.length]} isAnimationActive={false} />
          ))}
          {alerts.map((a, i) => (
            <ReferenceLine key={i} x={a.elapsed_s} stroke={T.ACCENT_ALERT}
                           strokeDasharray="4 3"
                           label={{ value: a.label, fill: T.ACCENT_ALERT, fontSize: 9, position: 'top' }} />
          ))}
          {cursor != null && <ReferenceLine x={cursor} stroke={T.TEXT_PRIMARY} />}
          {withBrush && <Brush dataKey="elapsed_s" height={18} travellerWidth={8}
                               stroke={T.ACCENT_DATA} tickFormatter={fmt} />}
        </LineChart>
      </ResponsiveContainer>

      {readout && (
        <div style={{ marginTop: 8, padding: 10, border: `1px solid ${T.BORDER}`, background: T.BG_SURFACE }}>
          <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginBottom: 6 }}>
            SENSORS @ {fmt(readout.row.elapsed_s)}
            {readout.alert && <span style={{ color: T.ACCENT_ALERT }}> — ALERT: {readout.alert.label}</span>}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 4 }}>
            {USEFUL_PIDS.map(pid => (
              <div key={pid} style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_SECONDARY }}>
                {pid.replace(/_/g, ' ')}: <span style={{ color: T.TEXT_PRIMARY }}>
                  {Number.isFinite(readout.row[pid]) ? readout.row[pid].toFixed(1) : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
