import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getRecording, getRecordingRows, type RecordingDetail, type OBDWindow, type RecordingRows } from '../api';
import { T, FAULT_DISPLAY, DIAGNOSTIC_STEPS } from '../theme';
import StatusBanner from '../components/StatusBanner';
import SeverityGrid from '../components/SeverityGrid';
import AnomalyPanel from '../components/AnomalyPanel';
import SensorTimeline from '../components/SensorTimeline';
import ShapPanel from '../components/ShapPanel';

export default function Results() {
  const { carId, rid } = useParams<{ carId: string; rid: string }>();
  const nav = useNavigate();
  const [detail, setDetail] = useState<RecordingDetail | null>(null);
  const [rowsData, setRowsData] = useState<RecordingRows | null>(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    const n = Number(rid);
    getRecording(n).then(setDetail).catch(e => setErr(e.message ?? 'Failed to load'));
    getRecordingRows(n).then(setRowsData).catch(() => {});
  }, [rid]);

  if (err) return <Msg color={T.ACCENT_ALERT}>{err}</Msg>;
  if (!detail) return <Msg color={T.TEXT_MUTED}>Loading…</Msg>;

  const { recording, result, inspect } = detail;
  if (!result) return <Msg color={T.TEXT_MUTED}>No result data available for this recording.</Msg>;

  const windows: OBDWindow[] = result.windows ?? [];
  const summary = result.summary;

  // Derive the "dominant" label for the banner (most non-healthy, or overall most common)
  const labelCounts = summary.label_counts;
  const dominant = Object.entries(labelCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'healthy';

  // Pick the last window for per-window panels
  const lastW = windows[windows.length - 1];
  const severities = lastW?.severities ?? {};
  const forecasts = lastW?.forecasts ?? {};
  const topShap = lastW?.top_shap;
  const anomalyMean = recording.anomaly_mean ?? (windows.length ? windows.reduce((s, w) => s + w.anomaly_score, 0) / windows.length : 0);
  const avgConf = windows.length ? windows.reduce((s, w) => s + w.confidence, 0) / windows.length : 0;

  // Alert onset markers: first window of each new fault label (transitions only)
  const alertMarkers = windows
    .filter((w, i, ws) =>
      w.label !== 'healthy' && w.label !== 'cold_start' &&
      (i === 0 || ws[i - 1].label !== w.label))
    .map(w => ({ elapsed_s: w.elapsed_s, label: w.label }));

  // Diagnostic steps for the dominant fault
  const steps = DIAGNOSTIC_STEPS[dominant] ?? [];

  const isFault = dominant !== 'healthy' && dominant !== 'cold_start';

  return (
    <div style={{ minHeight: '100vh', background: T.BG_BASE }}>
      {/* Header */}
      <div style={{ borderBottom: `1px solid ${T.BORDER}`, padding: '0 32px', display: 'flex', alignItems: 'center', gap: 16, height: 56 }}>
        <button onClick={() => nav(`/cars/${carId}`)} style={{ background: 'none', border: 'none', color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 12, cursor: 'pointer', padding: 0 }}>← {recording.original_filename ?? 'Recording'}</button>
      </div>

      <div style={{ maxWidth: 1000, margin: '0 auto', padding: '28px 24px' }}>
        {/* Honest caveats */}
        {inspect?.warnings?.some(w => w.includes('MAF')) && (
          <Caveat color={T.ACCENT_WARN}>
            ⚠ MAF engine detected — MAP carries no idle vacuum signal. A vacuum leak on this car will most likely appear as positive LTFT/STFT (fuel_system or air_system label). Do not expect a clean air_system label.
          </Caveat>
        )}
        {windows.some(w => w.confidence < 0.6) && (
          <Caveat color={T.TEXT_MUTED}>ⓘ Some windows have low classifier confidence (below 60%). Treat those predictions as uncertain.</Caveat>
        )}

        <StatusBanner label={dominant} confidence={avgConf} anomalyScore={anomalyMean} faultFraction={summary.fault_fraction} />

        {/* Summary counts */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
          {Object.entries(labelCounts).map(([lbl, count]) => (
            <div key={lbl} style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '8px 14px' }}>
              <div style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED, letterSpacing: '0.1em', marginBottom: 2 }}>{FAULT_DISPLAY[lbl] ?? lbl}</div>
              <div style={{ fontFamily: T.FONT_MONO, fontSize: 18, fontWeight: 700, color: T.TEXT_PRIMARY }}>
                {count}
                <span style={{ fontSize: 10, color: T.TEXT_MUTED, marginLeft: 4 }}>win</span>
              </div>
            </div>
          ))}
          {recording.recall != null && (
            <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.ACCENT_OK}`, borderRadius: 4, padding: '8px 14px' }}>
              <div style={{ fontFamily: T.FONT_MONO, fontSize: 9, color: T.TEXT_MUTED, letterSpacing: '0.1em', marginBottom: 2 }}>RECALL</div>
              <div style={{ fontFamily: T.FONT_MONO, fontSize: 18, fontWeight: 700, color: T.ACCENT_OK }}>
                {(recording.recall * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>

        <SeverityGrid severities={severities} forecasts={forecasts} />
        <AnomalyPanel score={anomalyMean} />
        {rowsData && (
          <SensorTimeline
            rows={rowsData.rows.map(r => ({ ...r, elapsed_s: (r.elapsed_s as number) })) as Record<string, number>[]}
            alerts={alertMarkers}
            withBrush
          />
        )}
        <ShapPanel topShap={topShap} />

        {/* Diagnostic steps */}
        {isFault && steps.length > 0 && (
          <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 4, padding: '16px 20px' }}>
            <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 12 }}>
              Diagnostic steps · {FAULT_DISPLAY[dominant]}
            </div>
            {steps.map((step, i) => (
              <div key={i} style={{ display: 'flex', gap: 12, padding: '8px 0', borderBottom: i < steps.length - 1 ? `1px solid ${T.BORDER}` : 'none' }}>
                <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_DATA, minWidth: 20 }}>{String(i + 1).padStart(2, '0')}</span>
                <span style={{ fontFamily: T.FONT_BODY, fontSize: 13, color: T.TEXT_PRIMARY, lineHeight: 1.5 }}>{step}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Msg({ children, color }: { children: React.ReactNode; color: string }) {
  return <div style={{ padding: 48, fontFamily: T.FONT_MONO, fontSize: 13, color, textAlign: 'center' }}>{children}</div>;
}

function Caveat({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <div style={{ padding: '8px 14px', background: `${color}18`, border: `1px solid ${color}40`, borderRadius: 4, color, fontFamily: T.FONT_MONO, fontSize: 11, marginBottom: 14, lineHeight: 1.5 }}>
      {children}
    </div>
  );
}
