export default function Footer() {
  return (
    <footer style={{
      borderTop: "1px solid var(--color-border)",
      background: "var(--color-surface)",
      padding: "var(--space-6)",
    }}>
      <style>{`
        .footer-link { color: var(--color-text-muted); text-decoration: none; font-size: var(--text-xs); font-family: var(--font-body); transition: color var(--transition-fast); }
        .footer-link:hover { color: var(--color-text-secondary); }
      `}</style>
      <div style={{
        maxWidth: 1120,
        margin: "0 auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "var(--space-4)",
      }}>
        <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", fontFamily: "var(--font-body)" }}>
          © 2026 OtoScope AI. For research use only. Not a medical device.
        </p>
        <nav style={{ display: "flex", gap: "var(--space-5)" }}>
          {["Privacy Policy", "Terms of Use", "Contact"].map(label => (
            <a key={label} href="#" className="footer-link">{label}</a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
