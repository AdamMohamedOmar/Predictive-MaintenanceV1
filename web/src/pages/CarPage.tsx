import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCar, listRecordings, uploadRecording, type Car, type RecordingOut, type BaselineOut } from '../api';
import { T } from '../theme';
import LiveSession from '../components/LiveSession';
import CalibrationCard from '../components/CalibrationCard';

type Tab = 'overview' | 'upload' | 'live' | 'history';

// Shared sub-styles
const hdr: React.CSSProperties = { fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6 };

export default function CarPage() {
  const { carId } = useParams<{ carId: string }>();
  const nav = useNavigate();
  const id = Number(carId);
  const [car, setCar] = useState<Car | null>(null);
  const [tab, setTab] = useState<Tab>('overview');
  const [recordings, setRecordings] = useState<RecordingOut[]>([]);

  useEffect(() => {
    getCar(id).then(c => { setCar(c); }).catch(() => nav('/garage'));
    listRecordings(id).then(setRecordings).catch(() => {});
  }, [id, nav]);

  if (!car) return <div style={{ color: T.TEXT_MUTED, padding: 40, fontFamily: T.FONT_MONO }}>Loading…</div>;

  const tabs: { key: Tab; label: string }[] = [
    { key: 'overview', label: 'OVERVIEW' }, { key: 'upload', label: 'ADD RECORDING' },
    { key: 'live', label: 'LIVE' }, { key: 'history', label: 'HISTORY' },
  ];

  return (
    <div style={{ minHeight: '100vh', background: T.BG_BASE }}>
      {/* Header */}
      <div style={{ borderBottom: `1px solid ${T.BORDER}`, padding: '0 32px', display: 'flex', alignItems: 'center', gap: 16, height: 56 }}>
        <button onClick={() => nav('/garage')} style={{ background: 'none', border: 'none', color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 12, cursor: 'pointer', padding: 0 }}>← GARAGE</button>
        <span style={{ color: T.BORDER }}>|</span>
        <span style={{ fontFamily: T.FONT_MONO, fontSize: 13, color: T.TEXT_PRIMARY }}>{car.make} {car.model} {car.year}</span>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: `1px solid ${T.BORDER}`, display: 'flex', padding: '0 32px' }}>
        {tabs.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            padding: '12px 18px', background: 'none', border: 'none',
            borderBottom: `2px solid ${tab === t.key ? T.ACCENT_DATA : 'transparent'}`,
            color: tab === t.key ? T.TEXT_PRIMARY : T.TEXT_SECONDARY,
            fontFamily: T.FONT_MONO, fontSize: 11, letterSpacing: '0.1em', cursor: 'pointer',
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{ maxWidth: 1000, margin: '0 auto', padding: '28px 24px' }}>
        {tab === 'overview' && <Overview car={car} recordings={recordings} onOpen={r => nav(`/cars/${id}/recordings/${r.id}`)} onCalibrated={() => getCar(id).then(setCar)} />}
        {tab === 'upload' && <UploadTab car={car} carId={id} onBaseline={c => setCar(c)} />}
        {tab === 'live' && <LiveSession carId={id} />}
        {tab === 'history' && <History recordings={recordings} onOpen={r => nav(`/cars/${id}/recordings/${r.id}`)} />}
      </div>
    </div>
  );
}

// ── Overview ─────────────────────────────────────────────────────────────────
function Overview({ car, recordings, onOpen, onCalibrated }: { car: Car; recordings: RecordingOut[]; onOpen: (r: RecordingOut) => void; onCalibrated: () => void }) {
  return (
    <div>
      <CalibrationCard car={car} onCalibrated={onCalibrated} />
      <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '20px 24px', marginBottom: 20 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20 }}>
          {[['Make', car.make], ['Model', car.model], ['Year', String(car.year)], ['Engine metering', car.engine_metering]].map(([k, v]) => (
            <div key={k}><div style={hdr}>{k}</div><div style={{ fontFamily: T.FONT_MONO, fontSize: 14, color: T.TEXT_PRIMARY }}>{v}</div></div>
          ))}
        </div>
      </div>
      <div style={{ background: car.baseline_normalizer_path ? `${T.ACCENT_OK}18` : T.BG_SURFACE, border: `1px solid ${car.baseline_normalizer_path ? T.ACCENT_OK : T.BORDER}`, borderRadius: 4, padding: '14px 20px', marginBottom: 20 }}>
        <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: car.baseline_normalizer_path ? T.ACCENT_OK : T.TEXT_MUTED }}>
          {car.baseline_normalizer_path ? '✓ Baseline captured — scoring uses this vehicle\'s normalizer' : '⊙ No baseline — upload a healthy drive to improve accuracy'}
        </span>
      </div>
      {recordings.slice(0, 3).map(r => <RecordingRow key={r.id} r={r} onOpen={onOpen} />)}
    </div>
  );
}

// ── Upload tab ────────────────────────────────────────────────────────────────
function UploadTab({ carId, car, onBaseline }: { carId: number; car: Car; onBaseline: (c: Car) => void }) {
  const [isBaseline, setIsBaseline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [result, setResult] = useState<RecordingOut | BaselineOut | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const nav = useNavigate();

  const doUpload = async (file: File) => {
    setLoading(true); setErr(''); setResult(null);
    try {
      const res = await uploadRecording(carId, file, { isBaseline });
      setResult(res);
      if ('mode' in res && res.mode === 'baseline') {
        const updated = { ...car, baseline_normalizer_path: res.normalizer_path };
        onBaseline(updated as Car);
      } else if ('id' in res) {
        nav(`/cars/${carId}/recordings/${res.id}`);
      }
    } catch (ex: unknown) {
      const e = ex as { message?: string; status?: number };
      setErr(e.message ?? 'Upload failed');
    } finally { setLoading(false); }
  };

  const onDrop = (e: React.DragEvent) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) doUpload(f); };

  return (
    <div>
      {car.baseline_normalizer_path && !isBaseline && (
        <div style={{ padding: '8px 14px', background: `${T.ACCENT_OK}18`, border: `1px solid ${T.ACCENT_OK}30`, borderRadius: 4, fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_OK, marginBottom: 14 }}>
          Scoring will use this car's baseline normalizer ✓
        </div>
      )}
      <label style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
        <input type="checkbox" checked={isBaseline} onChange={e => setIsBaseline(e.target.checked)} />
        <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_SECONDARY }}>This is a healthy baseline drive</span>
      </label>
      <div onDrop={onDrop} onDragOver={e => e.preventDefault()} onClick={() => inputRef.current?.click()}
        style={{ border: `2px dashed ${T.BORDER}`, borderRadius: 6, padding: '48px 24px', textAlign: 'center', cursor: 'pointer', background: T.BG_SURFACE }}>
        <input ref={inputRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) doUpload(f); }} />
        <div style={{ fontFamily: T.FONT_MONO, fontSize: 13, color: loading ? T.ACCENT_DATA : T.TEXT_SECONDARY }}>
          {loading ? '⟳ Processing…' : '⊕ Drop CSV here or click to browse'}
        </div>
        <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.TEXT_MUTED, marginTop: 8 }}>
          Torque / Car-Scanner ELM327 export or adapted clean-column CSV
        </div>
      </div>
      {err && <div style={{ marginTop: 14, padding: '10px 14px', background: `${T.ACCENT_ALERT}18`, border: `1px solid ${T.ACCENT_ALERT}40`, borderRadius: 4, color: T.ACCENT_ALERT, fontFamily: T.FONT_MONO, fontSize: 12 }}>{err}</div>}
      {result && 'mode' in result && result.mode === 'baseline' && (
        <div style={{ marginTop: 14, padding: '12px 16px', background: `${T.ACCENT_OK}18`, border: `1px solid ${T.ACCENT_OK}40`, borderRadius: 4, color: T.ACCENT_OK, fontFamily: T.FONT_MONO, fontSize: 12 }}>
          ✓ {result.message}
        </div>
      )}
    </div>
  );
}

// ── History ───────────────────────────────────────────────────────────────────
function History({ recordings, onOpen }: { recordings: RecordingOut[]; onOpen: (r: RecordingOut) => void }) {
  if (!recordings.length) return <div style={{ color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 13, padding: '40px 0', textAlign: 'center' }}>No recordings yet</div>;
  return <div>{recordings.map(r => <RecordingRow key={r.id} r={r} onOpen={onOpen} />)}</div>;
}

function RecordingRow({ r, onOpen }: { r: RecordingOut; onOpen: (r: RecordingOut) => void }) {
  let labels: Record<string, number> = {};
  try { if (r.label_summary) labels = JSON.parse(r.label_summary); } catch { /* ignored */ }
  const dominant = Object.entries(labels).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'unknown';
  return (
    <div onClick={() => onOpen(r)} style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '14px 18px', marginBottom: 10, cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div>
        <div style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_PRIMARY }}>{r.original_filename ?? 'recording'}</div>
        <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginTop: 3 }}>{r.created_at?.slice(0, 10) ?? ''}</div>
      </div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        {r.anomaly_mean != null && <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.TEXT_SECONDARY }}>anomaly {r.anomaly_mean.toFixed(2)}</span>}
        {r.recall != null && <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_OK }}>recall {(r.recall * 100).toFixed(0)}%</span>}
        <span style={{ fontFamily: T.FONT_MONO, fontSize: 10, padding: '3px 8px', borderRadius: 12, background: T.BG_RAISED, color: T.TEXT_SECONDARY }}>{dominant}</span>
      </div>
    </div>
  );
}
