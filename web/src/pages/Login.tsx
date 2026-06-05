import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth';
import { T } from '../theme';

const inp: React.CSSProperties = {
  width: '100%', padding: '10px 14px', background: T.BG_RAISED,
  border: `1px solid ${T.BORDER}`, borderRadius: 4, color: T.TEXT_PRIMARY,
  fontFamily: T.FONT_MONO, fontSize: 14, boxSizing: 'border-box',
  outline: 'none', transition: 'border-color 0.15s',
};
const btn: React.CSSProperties = {
  width: '100%', padding: '11px', background: T.ACCENT_DATA, border: 'none',
  borderRadius: 4, color: '#fff', fontFamily: T.FONT_MONO, fontSize: 13,
  fontWeight: 700, cursor: 'pointer', letterSpacing: '0.08em',
};

export default function Login() {
  const [tab, setTab] = useState<'signin' | 'signup'>('signin');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, signup } = useAuth();
  const nav = useNavigate();

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      if (tab === 'signin') await login(username, password);
      else await signup(username, password);
      nav('/garage');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally { setLoading(false); }
  };

  const tabStyle = (active: boolean): React.CSSProperties => ({
    flex: 1, padding: '9px', background: active ? T.BG_RAISED : 'transparent',
    border: 'none', borderBottom: `2px solid ${active ? T.ACCENT_DATA : T.BORDER}`,
    color: active ? T.TEXT_PRIMARY : T.TEXT_SECONDARY, fontFamily: T.FONT_MONO,
    fontSize: 12, letterSpacing: '0.1em', cursor: 'pointer', textTransform: 'uppercase',
  });

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: T.BG_BASE }}>
      {/* Subtle grid background */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: `linear-gradient(${T.BORDER} 1px, transparent 1px), linear-gradient(90deg, ${T.BORDER} 1px, transparent 1px)`, backgroundSize: '48px 48px', opacity: 0.25 }} />

      <div style={{ position: 'relative', width: 360, background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, borderRadius: 6, overflow: 'hidden' }}>
        {/* Brand */}
        <div style={{ padding: '28px 28px 20px', textAlign: 'center' }}>
          <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, letterSpacing: '0.2em', color: T.ACCENT_DATA, textTransform: 'uppercase' }}>
            ◈ PREDICTIVE MAINTENANCE
          </div>
          <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginTop: 4, letterSpacing: '0.12em' }}>
            OBD-II FAULT CLASSIFICATION
          </div>
        </div>

        {/* Tab toggle */}
        <div style={{ display: 'flex', borderBottom: `1px solid ${T.BORDER}` }}>
          <button style={tabStyle(tab === 'signin')} onClick={() => setTab('signin')}>Sign In</button>
          <button style={tabStyle(tab === 'signup')} onClick={() => setTab('signup')}>Sign Up</button>
        </div>

        <form onSubmit={submit} style={{ padding: '24px 28px 28px' }}>
          <div style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6 }}>Username</label>
            <input style={inp} value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required />
          </div>
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6 }}>Password</label>
            <input style={inp} type="password" value={password} onChange={e => setPassword(e.target.value)} autoComplete={tab === 'signin' ? 'current-password' : 'new-password'} required />
          </div>
          {error && <div style={{ marginBottom: 14, padding: '8px 12px', background: `${T.ACCENT_ALERT}18`, border: `1px solid ${T.ACCENT_ALERT}40`, borderRadius: 4, color: T.ACCENT_ALERT, fontFamily: T.FONT_MONO, fontSize: 12 }}>{error}</div>}
          <button style={{ ...btn, opacity: loading ? 0.6 : 1 }} type="submit" disabled={loading}>
            {loading ? 'Please wait…' : tab === 'signin' ? 'SIGN IN →' : 'CREATE ACCOUNT →'}
          </button>
        </form>
      </div>
    </div>
  );
}
