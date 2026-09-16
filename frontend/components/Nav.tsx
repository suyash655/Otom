"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/",               label: "Home"         },
  { href: "/demo",           label: "Demo"         },
  { href: "/explainability", label: "How It Works" },
  { href: "/evaluation",     label: "Evaluation"   },
  { href: "/batch",          label: "Batch"        },
];

export default function Nav() {
  const path = usePathname();

  return (
    <header style={{
      borderBottom: "1px solid var(--color-border)",
      background: "rgba(250,249,247,0.92)",
      backdropFilter: "blur(12px)",
      position: "sticky",
      top: 0,
      zIndex: 50,
    }}>
      <div style={{
        maxWidth: 1120,
        margin: "0 auto",
        padding: "0 var(--space-6)",
        height: 56,
        display: "flex",
        alignItems: "center",
        gap: "var(--space-8)",
      }}>
        {/* Wordmark */}
        <Link href="/" style={{
          fontFamily: "var(--font-display)",
          fontWeight: 400,
          fontSize: "var(--text-lg)",
          color: "var(--color-text-primary)",
          textDecoration: "none",
          letterSpacing: "-0.01em",
          flexShrink: 0,
        }}>
          OTOM
        </Link>

        {/* Links — centred */}
        <nav style={{ display: "flex", gap: "var(--space-1)", flex: 1, justifyContent: "center" }}>
          {links.map(({ href, label }) => {
            const active = path === href;
            return (
              <Link key={href} href={href} style={{
                padding: "var(--space-2) var(--space-3)",
                borderRadius: "var(--radius-sm)",
                fontSize: "var(--text-sm)",
                fontFamily: "var(--font-body)",
                fontWeight: active ? 500 : 400,
                color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                background: active ? "var(--color-surface-2)" : "transparent",
                textDecoration: "none",
                transition: "color var(--transition-fast), background var(--transition-fast)",
              }}
              onMouseEnter={e => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.color = "var(--color-text-primary)";
                  (e.currentTarget as HTMLElement).style.background = "var(--color-surface-2)";
                }
              }}
              onMouseLeave={e => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.color = "var(--color-text-secondary)";
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                }
              }}>
                {label}
              </Link>
            );
          })}
        </nav>

        {/* CTA */}
        <Link href="/demo" style={{
          background: "var(--color-accent)",
          color: "#fff",
          fontFamily: "var(--font-body)",
          fontWeight: 500,
          fontSize: "var(--text-sm)",
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-sm)",
          textDecoration: "none",
          flexShrink: 0,
          transition: "background var(--transition-fast)",
        }}
        onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = "var(--color-accent-light)"}
        onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = "var(--color-accent)"}>
          Try Demo
        </Link>
      </div>
    </header>
  );
}
