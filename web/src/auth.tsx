import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { login as apiLogin, signup as apiSignup, setToken } from './api';

interface AuthState {
  token: string | null;
  username: string | null;
  userId: number | null;
  login: (u: string, p: string) => Promise<void>;
  signup: (u: string, p: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(() => localStorage.getItem('pm_token'));
  const [username, setUsername] = useState<string | null>(() => localStorage.getItem('pm_username'));
  const [userId, setUserId] = useState<number | null>(() => {
    const id = localStorage.getItem('pm_user_id');
    return id ? Number(id) : null;
  });

  // Sync token into api.ts on mount and whenever it changes
  useEffect(() => { setToken(token); }, [token]);

  const _persist = (tok: string, user: { id: number; username: string }) => {
    localStorage.setItem('pm_token', tok);
    localStorage.setItem('pm_username', user.username);
    localStorage.setItem('pm_user_id', String(user.id));
    setToken(tok);
    setTokenState(tok);
    setUsername(user.username);
    setUserId(user.id);
  };

  const login = async (u: string, p: string) => {
    const res = await apiLogin(u, p);
    _persist(res.access_token, res.user);
  };

  const signup = async (u: string, p: string) => {
    const res = await apiSignup(u, p);
    _persist(res.access_token, res.user);
  };

  const logout = () => {
    localStorage.removeItem('pm_token');
    localStorage.removeItem('pm_username');
    localStorage.removeItem('pm_user_id');
    setToken(null);
    setTokenState(null);
    setUsername(null);
    setUserId(null);
  };

  return (
    <AuthContext.Provider value={{ token, username, userId, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
}
