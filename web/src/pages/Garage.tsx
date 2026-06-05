import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth';
import { listCars, createCar, deleteCar, type Car } from '../api';
import { T } from '../theme';

const METERING_LABEL: Record<string, string> = { maf: 'MAF', speed_density: 'Speed-density', unknown: 'Unknown' };
const METERING_COLOR: Record<string, string> = { maf: T.ACCENT_WARN, speed_density: T.ACCENT_DATA, unknown: T.TEXT_MUTED };

export default function Garage() {
  const { username, logout } = useAuth();
  const nav = useNavigate();
  const [cars, setCars] = useState<Car[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({ make: '', model: '', year: new Date().getFullYear(), engine_metering: 'unknown' });
  const [adding, setAdding] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => { listCars().then(setCars).catch(() => {}); }, []);

  const addCar = async (e: React.FormEvent) => {
    e.preventDefault(); setAdding(true); setErr('');
    try {
      const car = await createCar(form);
      setCars(prev => [...prev, car]);
      setShowModal(false);
      setForm({ make: '', model: '', year: new Date().getFullYear(), engine_metering: 'unknown' });
    } catch (ex: unknown) { setErr(ex instanceof Error ? ex.message : 'Failed'); }
    finally { setAdding(false); }
  };

  const del = async (id: number) => {
    if (!confirm('Remove this car?')) return;
    await deleteCar(id).catch(() => {});
    setCars(prev => prev.filter(c => c.id !== id));
  };

  const inp: React.CSSProperties = {
    width: '100%', padding: '8px 12px', background: T.BG_RAISED, border: `1px solid ${T.BORDER}`,
    borderRadius: 4, color: T.TEXT_PRIMARY, fontFamily: T.FONT_MONO, fontSize: 13, boxSizing: 'border-box',
  };

  return (
    <div style={{ minHeight: '100vh', background: T.BG_BASE }}>
      {/* Header */}
      <div style={{ borderBottom: `1px solid ${T.BORDER}`, padding: '0 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 56 }}>
        <span style={{ fontFamily: T.FONT_MONO, fontSize: 13, color: T.ACCENT_DATA, letterSpacing: '0.12em' }}>◈ YOUR GARAGE</span>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.TEXT_MUTED }}>{username}</span>
          <button onClick={logout} style={{ background: 'none', border: `1px solid ${T.BORDER}`, color: T.TEXT_SECONDARY, fontFamily: T.FONT_MONO, fontSize: 11, padding: '5px 12px', borderRadius: 4, cursor: 'pointer' }}>Logout</button>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px' }}>
        {/* Add car button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 24 }}>
          <button onClick={() => setShowModal(true)} style={{ background: T.ACCENT_DATA, border: 'none', color: '#fff', fontFamily: T.FONT_MONO, fontSize: 12, padding: '9px 18px', borderRadius: 4, cursor: 'pointer', letterSpacing: '0.08em' }}>＋ ADD CAR</button>
        </div>

        {cars.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '80px 0', color: T.TEXT_MUTED, fontFamily: T.FONT_MONO, fontSize: 13 }}>
            <div style={{ fontSize: 32, marginBottom: 16 }}>◻</div>
            No cars yet — add your first vehicle
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
            {cars.map(car => (
              <div key={car.id} style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 6, padding: '20px 22px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
                  <div>
                    <div style={{ fontFamily: T.FONT_MONO, fontSize: 15, fontWeight: 700, color: T.TEXT_PRIMARY }}>{car.make} {car.model}</div>
                    <div style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_MUTED, marginTop: 2 }}>{car.year}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    {/* Baseline dot */}
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: car.baseline_normalizer_path ? T.ACCENT_OK : T.TEXT_MUTED }} title={car.baseline_normalizer_path ? 'Baseline set' : 'No baseline'} />
                    <button onClick={() => del(car.id)} style={{ background: 'none', border: 'none', color: T.TEXT_MUTED, cursor: 'pointer', fontSize: 16, padding: 0 }}>×</button>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                  <span style={{ fontFamily: T.FONT_MONO, fontSize: 10, padding: '3px 8px', borderRadius: 12, background: `${METERING_COLOR[car.engine_metering] ?? T.TEXT_MUTED}22`, color: METERING_COLOR[car.engine_metering] ?? T.TEXT_MUTED, border: `1px solid ${METERING_COLOR[car.engine_metering] ?? T.TEXT_MUTED}44` }}>
                    {METERING_LABEL[car.engine_metering] ?? car.engine_metering}
                  </span>
                  <span style={{ fontFamily: T.FONT_MONO, fontSize: 10, padding: '3px 8px', borderRadius: 12, background: `${car.baseline_normalizer_path ? T.ACCENT_OK : T.TEXT_MUTED}22`, color: car.baseline_normalizer_path ? T.ACCENT_OK : T.TEXT_MUTED }}>
                    {car.baseline_normalizer_path ? 'Baseline ✓' : 'No baseline'}
                  </span>
                </div>
                <button onClick={() => nav(`/cars/${car.id}`)} style={{ width: '100%', padding: '8px', background: T.BG_RAISED, border: `1px solid ${T.BORDER}`, color: T.TEXT_PRIMARY, fontFamily: T.FONT_MONO, fontSize: 12, borderRadius: 4, cursor: 'pointer' }}>
                  OPEN →
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add car modal */}
      {showModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 6, width: 380, padding: '28px' }}>
            <div style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.ACCENT_DATA, letterSpacing: '0.12em', marginBottom: 20 }}>ADD CAR</div>
            <form onSubmit={addCar} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {[['Make', 'make'], ['Model', 'model']].map(([label, key]) => (
                <div key={key}>
                  <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginBottom: 4, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</div>
                  <input style={inp} required value={(form as Record<string, string | number>)[key] as string} onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))} />
                </div>
              ))}
              <div>
                <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginBottom: 4, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Year</div>
                <input style={inp} type="number" required value={form.year} onChange={e => setForm(f => ({ ...f, year: Number(e.target.value) }))} />
              </div>
              <div>
                <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginBottom: 4, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Engine metering
                  <span style={{ color: T.TEXT_MUTED, fontWeight: 400, marginLeft: 6 }} title="MAF vs speed-density affects how vacuum leaks appear to the model">ⓘ</span>
                </div>
                <select style={{ ...inp, appearance: 'none' }} value={form.engine_metering} onChange={e => setForm(f => ({ ...f, engine_metering: e.target.value }))}>
                  <option value="unknown">Unknown</option>
                  <option value="speed_density">Speed-density (MAP)</option>
                  <option value="maf">MAF-based</option>
                </select>
              </div>
              {err && <div style={{ color: T.ACCENT_ALERT, fontFamily: T.FONT_MONO, fontSize: 12 }}>{err}</div>}
              <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                <button type="button" onClick={() => setShowModal(false)} style={{ flex: 1, padding: '9px', background: 'none', border: `1px solid ${T.BORDER}`, color: T.TEXT_SECONDARY, fontFamily: T.FONT_MONO, fontSize: 12, borderRadius: 4, cursor: 'pointer' }}>Cancel</button>
                <button type="submit" disabled={adding} style={{ flex: 2, padding: '9px', background: T.ACCENT_DATA, border: 'none', color: '#fff', fontFamily: T.FONT_MONO, fontSize: 12, borderRadius: 4, cursor: 'pointer' }}>{adding ? 'Adding…' : 'ADD →'}</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
