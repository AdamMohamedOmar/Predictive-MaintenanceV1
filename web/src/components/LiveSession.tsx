import { useState, useEffect, useRef, useCallback } from 'react';
import { listSerialPorts, openLiveSocket, type TelemetryFrame, type SerialPort } from '../api';
import { T } from '../theme';
import StatusBanner from './StatusBanner';
import SeverityGrid from './SeverityGrid';
import AnomalyPanel from './AnomalyPanel';
import ShapPanel from './ShapPanel';
import PidStrip from './PidStrip';

interface Props { carId: number }

type ConnState = 'disconnected' | 'connecting' | 'waiting' | 'live' | 'error';

const HISTORY_LEN = 300;

export default function LiveSession({ carId }: Props) {
  const [ports, setPorts] = useState<SerialPort[]>([]);
  const [port, setPort] = useState('');
  const [connState, setConnState] = useState<ConnState>('disconnected');
  const [warnings, setWarnings] = useState<string[]>([]);
  const [frame, setFrame] = useState<TelemetryFrame | null>(null);
  const [pidHistory, setPidHistory] = useState<Record<string, number>[]>([]);
  const [leakStart, setLeakStart] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshPorts = useCallback(() => {
    listSerialPorts().then(ps => { setPorts(ps); if (ps.length && !port) setPort(ps[0].device); }).catch(() => {});
  }, [port]);

  useEffect(() => { refreshPorts(); }, [refreshPorts]);

  const connect = () => {
    if (wsRef.current) return;
    setConnState('connecting'); setWarnings([]);
    const ws = openLiveSocket(
      (f) => {
        if (f.type === 'telemetry') {
          setConnState('live');
          setFrame(f);
          setPidHistory(prev => {
            const entry = { elapsed_s: f.elapsed_s ?? 0, ...f.telemetry };
            const next = [...prev, entry];
            return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next;
          });
        } else if (f.type === 'warning' && f.message) {
          setWarnings(prev => [...prev, f.message!]);
        } else if (f.type === 'error') {
          setWarnings(prev => [...prev, f.message ?? 'Connection error']);
          setConnState('error');
        }
      },
      () => { setConnState('disconnected'); wsRef.current = null; }
    );
    wsRef.current = ws;
    ws.onopen = () => ws.send(JSON.stringify({ action: 'connect', port, car_id: carId }));
  };

  const disconnect = () => {
    wsRef.current?.send(JSON.stringify({ action: 'stop' }));
    wsRef.current?.close();
    wsRef.current = null;
    setConnState('disconnected');
    setFrame(null);
    setPidHistory([]);
    setLeakStart(null);
  };

  const markLeak = (state: 'start' | 'stop') => {
    wsRef.current?.send(JSON.stringify({ action: 'mark_leak', state }));
    if (state === 'start') setLeakStart(frame?.elapsed_s ?? null);
    else setLeakStart(null);
  };

  const stateColor: Record<ConnState, string> = {
    disconnected: T.TEXT_MUTED, connecting: T.ACCENT_WARN, waiting: T.ACCENT_WARN,
    live: T.ACCENT_OK, error: T.ACCENT_ALERT,
  };
  const stateLabel: Record<ConnState, string> = {
    disconnected: 'Disconnected', connecting: 'Connecting…', waiting: 'Waiting for ECU…',
    live: 'Live', error: 'Error',
  };

  const lastWindow = frame;
  const severities = lastWindow?.severities ?? {};
  const forecasts = lastWindow?.forecasts ?? {};

  return (
    <div>
      {/* Connection controls */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 20, flexWrap: 'wrap' }}>
        <select value={port} onChange={e => setPort(e.target.value)}
          style={{ background: T.BG_RAISED, border: `1px solid ${T.BORDER}`, color: T.TEXT_PRIMARY, fontFamily: T.FONT_MONO, fontSize: 12, padding: '8px 12px', borderRadius: 4, minWidth: 160 }}>
          {ports.length === 0 ? <option>No ports found</option> : ports.map(p => <option key={p.device} value={p.device}>{p.device} — {p.description}</option>)}
        </select>
        <button onClick={refreshPorts} style={{ background: 'none', border: `1px solid ${T.BORDER}`, color: T.TEXT_SECONDARY, fontFamily: T.FONT_MONO, fontSize: 11, padding: '7px 12px', borderRadius: 4, cursor: 'pointer' }}>↺</button>
        {connState === 'disconnected' || connState === 'error'
          ? <button onClick={connect} disabled={!port} style={{ background: T.ACCENT_DATA, border: 'none', color: '#fff', fontFamily: T.FONT_MONO, fontSize: 12, padding: '8px 18px', borderRadius: 4, cursor: 'pointer' }}>Connect</button>
          : <button onClick={disconnect} style={{ background: T.ACCENT_ALERT, border: 'none', color: '#fff', fontFamily: T.FONT_MONO, fontSize: 12, padding: '8px 18px', borderRadius: 4, cursor: 'pointer' }}>Disconnect</button>
        }
        {/* Status pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: stateColor[connState], boxShadow: connState === 'live' ? `0 0 6px ${T.ACCENT_OK}` : 'none' }} />
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: stateColor[connState] }}>{stateLabel[connState]}</span>
        </div>
      </div>

      {/* Warnings */}
      {warnings.map((w, i) => (
        <div key={i} style={{ padding: '8px 12px', background: `${T.ACCENT_WARN}18`, border: `1px solid ${T.ACCENT_WARN}40`, borderRadius: 4, color: T.ACCENT_WARN, fontFamily: T.FONT_MONO, fontSize: 11, marginBottom: 8 }}>⚠ {w}</div>
      ))}

      {/* Degraded PID warning */}
      {frame?.degraded_pid_count != null && frame.degraded_pid_count > 0 && (
        <div style={{ padding: '8px 12px', background: `${T.ACCENT_WARN}18`, border: `1px solid ${T.ACCENT_WARN}40`, borderRadius: 4, color: T.ACCENT_WARN, fontFamily: T.FONT_MONO, fontSize: 11, marginBottom: 12 }}>
          ⚠ {frame.degraded_pid_count} PID(s) unsupported by this ECU — classifier confidence degraded
        </div>
      )}

      {connState === 'live' && frame?.label && (
        <>
          <StatusBanner label={frame.label} confidence={frame.confidence ?? 0} anomalyScore={frame.anomaly_score ?? 0} />

          {/* Mark leak controls */}
          <div style={{ display: 'flex', gap: 10, marginBottom: 16, alignItems: 'center' }}>
            {leakStart == null
              ? <button onClick={() => markLeak('start')} style={{ background: T.ACCENT_WARN, border: 'none', color: '#000', fontFamily: T.FONT_MONO, fontSize: 11, padding: '7px 16px', borderRadius: 4, cursor: 'pointer', fontWeight: 700 }}>▶ Mark Leak Start</button>
              : <>
                  <button onClick={() => markLeak('stop')} style={{ background: T.ACCENT_ALERT, border: 'none', color: '#fff', fontFamily: T.FONT_MONO, fontSize: 11, padding: '7px 16px', borderRadius: 4, cursor: 'pointer', fontWeight: 700 }}>■ Mark Leak Stop</button>
                  <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_WARN }}>Fault interval: {leakStart}s → {frame?.elapsed_s ?? '…'}s</span>
                </>
            }
          </div>

          <SeverityGrid severities={severities} forecasts={forecasts} />
          <AnomalyPanel score={frame.anomaly_score ?? 0} />
          <PidStrip data={pidHistory as { elapsed_s: number; [k: string]: number }[]} />
          <ShapPanel topShap={frame.top_shap} />
        </>
      )}

      {(connState === 'disconnected' && !frame) && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 13 }}>
          <div style={{ fontSize: 28, marginBottom: 12 }}>⊙</div>
          Select a COM port and press Connect to begin live monitoring
        </div>
      )}
    </div>
  );
}
