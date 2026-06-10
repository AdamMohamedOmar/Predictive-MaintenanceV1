// Typed API client — mirrors the FastAPI backend contract exactly.

const BASE = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000') as string;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Car {
  id: number; user_id: number; make: string; model: string; year: number;
  engine_metering: string; baseline_normalizer_path: string | null; created_at: string | null;
}
export interface TokenOut { access_token: string; token_type: string; user: { id: number; username: string } }
export interface RecordingOut {
  id: number; car_id: number; kind: string; original_filename: string | null;
  adapted_csv_path: string | null; result_json_path: string | null;
  label_summary: string | null; anomaly_mean: number | null;
  recall: number | null; fault_from_s: number | null; fault_to_s: number | null;
  created_at: string | null;
}
export interface BaselineOut { mode: 'baseline'; normalizer_path: string; n_windows: number | null; message: string }
/** One classifier stride window from evaluate_real_fault or the live WS. */
export interface OBDWindow {
  elapsed_s: number; label: string; confidence: number; anomaly_score: number;
  all_probs: Record<string, number>; severities: Record<string, number>;
  forecasts: Record<string, number>; top_shap?: [string, number][];
}
export interface FullResult {
  n_windows: number; windows: OBDWindow[];
  summary: { fault_window_count: number; fault_fraction: number; label_counts: Record<string, number> };
}
export interface InspectReport {
  metering_type: string; warnings: string[]; pid_coverage?: Record<string, { present: boolean; fill_pct: number }>;
}
export interface RecordingDetail { recording: RecordingOut; result: FullResult | null; inspect: InspectReport | null }
export interface SerialPort { device: string; description: string }
export type AlertEvent =
  | { kind: 'stable'; fault_type: string; confidence: number; elapsed_s: number }
  | { kind: 'clear'; elapsed_s: number }
  | { kind: 'rule'; rule: string; elapsed_s: number };
export interface TelemetryFrame {
  type: 'telemetry' | 'warning' | 'error' | 'mark_ack';
  elapsed_s?: number; telemetry?: Record<string, number | null>; label?: string;
  confidence?: number; severities?: Record<string, number>; forecasts?: Record<string, number>;
  anomaly_score?: number; top_shap?: [string, number][]; degraded_pid_count?: number;
  missing_pids?: string[]; poll_hz?: number; t_poll?: number; message?: string;
  alert_events?: AlertEvent[]; armed?: boolean;
}
export interface CalibrateProgress { type: 'calibrate_progress'; rows_collected: number; elapsed_s: number }
export interface CalibrateResult { type: 'calibrate_result'; ok: boolean; n_windows?: number; path?: string; reason?: string }
export type WsConnectPayload = {
  action: 'connect'; port: string | null; car_id: number | null;
  mode?: 'monitor' | 'calibrate'; allow_idle?: boolean;
};

// ── Core fetch helper ────────────────────────────────────────────────────────

let _token: string | null = null;
export const setToken = (t: string | null) => { _token = t; };

async function fetchJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init.headers as Record<string, string> ?? {}) };
  if (_token) headers['Authorization'] = `Bearer ${_token}`;
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* ignored */ }
    throw Object.assign(new Error(detail), { status: res.status });
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export const signup = (username: string, password: string) =>
  fetchJson<TokenOut>('/api/auth/signup', { method: 'POST', body: JSON.stringify({ username, password }) });

export const login = (username: string, password: string) =>
  fetchJson<TokenOut>('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) });

// ── Cars ─────────────────────────────────────────────────────────────────────

export const listCars = () => fetchJson<Car[]>('/api/cars');
export const createCar = (b: { make: string; model: string; year: number; engine_metering: string }) =>
  fetchJson<Car>('/api/cars', { method: 'POST', body: JSON.stringify(b) });
export const getCar = (id: number) => fetchJson<Car>(`/api/cars/${id}`);
export const deleteCar = (id: number) => fetchJson<void>(`/api/cars/${id}`, { method: 'DELETE' });

// ── Recordings ────────────────────────────────────────────────────────────────

export async function uploadRecording(
  carId: number, file: File, opts: { isBaseline?: boolean; faultFromS?: number; faultToS?: number } = {}
): Promise<RecordingOut | BaselineOut> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('is_baseline', String(opts.isBaseline ?? false));
  if (opts.faultFromS != null) fd.append('fault_from_s', String(opts.faultFromS));
  if (opts.faultToS != null) fd.append('fault_to_s', String(opts.faultToS));
  const headers: Record<string, string> = {};
  if (_token) headers['Authorization'] = `Bearer ${_token}`;
  const res = await fetch(`${BASE}/api/cars/${carId}/recordings`, { method: 'POST', headers, body: fd });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* ignored */ }
    throw Object.assign(new Error(detail), { status: res.status });
  }
  return res.json();
}

export const getRecording = (id: number) => fetchJson<RecordingDetail>(`/api/recordings/${id}`);
export const listRecordings = (carId: number) => fetchJson<RecordingOut[]>(`/api/cars/${carId}/recordings`);

// ── Live ─────────────────────────────────────────────────────────────────────

export const listSerialPorts = () => fetchJson<SerialPort[]>('/api/serial/ports');

export function openLiveSocket(onFrame: (f: TelemetryFrame) => void, onClose: () => void): WebSocket {
  const wsBase = BASE.replace(/^http/, 'ws');
  const ws = new WebSocket(`${wsBase}/ws/live`);
  ws.onmessage = (e) => { try { onFrame(JSON.parse(e.data)); } catch { /* ignored */ } };
  ws.onclose = onClose;
  return ws;
}
