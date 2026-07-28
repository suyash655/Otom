"use client";
import { useState } from "react";

interface Props {
  methods: { label: string; b64: string }[];
}

export default function HeatmapViewer({ methods }: Props) {
  const [active, setActive] = useState(0);
  if (!methods.length) return null;

  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)", marginBottom: "var(--space-3)" }}>
        {methods.map((m, i) => (
          <button
            key={m.label}
            onClick={() => setActive(i)}
            style={{
              padding: "var(--space-1) var(--space-3)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid",
              borderColor: i === active ? "var(--color-accent)" : "var(--color-border)",
              background: i === active ? "var(--color-accent)" : "var(--color-surface-2)",
              color: i === active ? "#fff" : "var(--color-text-secondary)",
              fontFamily: "var(--font-body)",
              fontWeight: 500,
              fontSize: "var(--text-xs)",
              cursor: "pointer",
              transition: "background var(--transition-fast), color var(--transition-fast)",
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div style={{ borderRadius: "var(--radius-md)", overflow: "hidden", border: "1px solid var(--color-border)" }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${methods[active].b64}`}
          alt={methods[active].label}
          style={{ width: "100%", display: "block" }}
        />
      </div>
      <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginTop: "var(--space-2)", textAlign: "center" }}>
        {methods[active].label}
      </p>
    </div>
  );
}
