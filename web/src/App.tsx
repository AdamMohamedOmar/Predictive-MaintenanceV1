// Placeholder — replaced in Phase 1 with full routing.
// Background matches the dark cockpit theme from src/dashboard/theme.py.

export default function App() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0d1117",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        color: "#c9d1d9",
      }}
    >
      <div style={{ fontSize: "0.7rem", letterSpacing: "0.15em", color: "#58a6ff", textTransform: "uppercase", marginBottom: "1rem" }}>
        ◈ PREDICTIVE MAINTENANCE
      </div>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: 0, color: "#e6edf3" }}>
        Web App — coming soon
      </h1>
      <p style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "#8b949e" }}>
        Backend is up. Phase 1 (auth + garage) loads here next.
      </p>
    </div>
  );
}
