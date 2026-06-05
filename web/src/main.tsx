import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './auth';
import Login from './pages/Login';
import Garage from './pages/Garage';
import CarPage from './pages/CarPage';
import Results from './pages/Results';
import { T } from './theme';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token } = useAuth();
  return token ? <>{children}</> : <Navigate to="/login" replace />;
}

function AppRoutes() {
  const { token } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={token ? <Navigate to="/garage" replace /> : <Login />} />
      <Route path="/garage" element={<ProtectedRoute><Garage /></ProtectedRoute>} />
      <Route path="/cars/:carId" element={<ProtectedRoute><CarPage /></ProtectedRoute>} />
      <Route path="/cars/:carId/recordings/:rid" element={<ProtectedRoute><Results /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to={token ? '/garage' : '/login'} replace />} />
    </Routes>
  );
}

// Apply global base styles
const root = document.getElementById('root')!;
Object.assign(document.body.style, {
  margin: '0', background: T.BG_BASE, color: T.TEXT_PRIMARY, fontFamily: T.FONT_BODY,
});

createRoot(root).render(
  <StrictMode>
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  </StrictMode>
);
