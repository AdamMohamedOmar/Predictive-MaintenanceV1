import { useEffect, useRef, useState } from 'react';
import { listSerialPorts, openLiveSocket, type Car, type SerialPort, type TelemetryFrame } from '../api';
import { T } from '../theme';

type Phase = 'idle' | 'recording' | 'fitting' | 'done' | 'rejected';
interface Props { car: Car; onCalibrated: () => void }

export default function CalibrationCard({ car, onCalibrated }: Props) {
  const [ports, setPorts] = useState<SerialPort[]>([]);
  const [port, setPort] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [rows, setRows] = useState(0);
  const [message, setMessage] = useState('');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    listSerialPorts().then(ps => { setPorts(ps); if (ps.length) setPort(ps[0].device); }).catch(() => {});
  }, []);

  const start = () => {
    setPhase('recording'); setRows(0); setMessage('');
    const ws = openLiveSocket((f: TelemetryFrame) => {
      if (f.type === 'calibrate_progress') setRows(f.rows_collected ?? 0);
      else if (f.type === 'calibrate_result') {
        if (f.ok) { setPhase('done'); setMessage(`${f.n_windows} windows`); onCalibrated(); }
        else { setPhase('rejected'); setMessage(f.reason ?? 'rejected'); }
        wsRef.current?.close(); wsRef.current = null;
      } else if (f.type === 'error') {
        setPhase('rejected'); setMessage(f.message ?? 'connection error');
        wsRef.current?.close(); wsRef.current = null;
      }
    }, () => { if (phase === 'recording') setPhase('idle'); wsRef.current = null; });
    ws.onopen = () => ws.send(JSON.stringify({ action: 'connect', port, car_id: car.id, mode: 'calibrate' }));
    wsRef.current = ws;
  };

  const finish = () => { setPhase('fitting'); wsRef.current?.send(JSON.stringify({ action: 'finish_calibration' })); };

  const calibrated = !!car.baseline_normalizer_path;
  const mins = Math.floor(rows / 60), secs = rows % 60;

  return (
    <div style={{ border: `1px solid ${T.BORDER}`, padding: 16, marginBottom: 16, background: T.BG_SURFACE }}>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', marginBottom: 8 }}>
        BASELINE CALIBRATION
      </div>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 12, marginBottom: 10,
                    color: calibrated ? T.ACCENT_OK : T.ACCENT_WARN }}>
        {calibrated ? '✓ CALIBRATED — fault alerts armed' : '○ NOT CALIBRATED — fault alerts disarmed'}
      </div>

      {phase === 'idle' && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select value={port} onChange={e => setPort(e.target.value)}
                  style={{ background: T.BG_BASE, color: T.TEXT_PRIMARY, border: `1px solid ${T.BORDER}`,
                           fontFamily: T.FONT_MONO, fontSize: 11, padding: '6px 8px' }}>
            {ports.map(p => <option key={p.device} value={p.device}>{p.description}</option>)}
          </select>
          <button onClick={start} style={{ fontFamily: T.FONT_MONO, fontSize: 11, padding: '6px 14px',
                  background: T.ACCENT_DATA, color: T.BG_BASE, border: 'none', cursor: 'pointer' }}>
            {calibrated ? 'RE-CALIBRATE' : 'CALIBRATE'}
          </button>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED }}>
            engine warm · ~5 min of normal driving
          </span>
        </div>
      )}

      {phase === 'recording' && (
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_PRIMARY }}>
            ● RECORDING {mins}:{String(secs).padStart(2, '0')} ({rows} rows)
          </span>
          <button onClick={finish} disabled={rows < 240}
                  style={{ fontFamily: T.FONT_MONO, fontSize: 11, padding: '6px 14px', cursor: 'pointer',
                           background: rows < 240 ? T.BG_BASE : T.ACCENT_OK,
                           color: rows < 240 ? T.TEXT_MUTED : T.BG_BASE, border: `1px solid ${T.BORDER}` }}>
            {rows < 240 ? `FINISH (need ${240 - rows}s more)` : 'FINISH & FIT'}
          </button>
        </div>
      )}

      {phase === 'fitting' && <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_SECONDARY }}>FITTING…</span>}
      {phase === 'done' && <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.ACCENT_OK }}>✓ SAVED — {message}</span>}
      {phase === 'rejected' && (
        <div>
          <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_ALERT, marginBottom: 6 }}>REJECTED: {message}</div>
          <button onClick={() => setPhase('idle')} style={{ fontFamily: T.FONT_MONO, fontSize: 11, padding: '4px 10px', cursor: 'pointer' }}>TRY AGAIN</button>
        </div>
      )}
    </div>
  );
}
