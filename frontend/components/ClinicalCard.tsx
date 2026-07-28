interface Reasoning {
  summary?: string;
  attention?: string;
  tda_findings?: string[];
  class_context?: string;
  confidence_note?: string;
}

function md(s: string) {
  return s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

export default function ClinicalCard({ reasoning }: { reasoning: Reasoning }) {
  if (!reasoning) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>

      {reasoning.summary && (
        <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", lineHeight: 1.65 }}
           dangerouslySetInnerHTML={{ __html: md(reasoning.summary) }} />
      )}

      {reasoning.attention && (
        <div>
          <p style={{ fontSize: "var(--text-xs)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--color-text-muted)", marginBottom: "var(--space-2)" }}>
            Attention region
          </p>
          <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-secondary)" }}>{reasoning.attention}</p>
        </div>
      )}

      {reasoning.tda_findings && reasoning.tda_findings.length > 0 && (
        <div>
          <p style={{ fontSize: "var(--text-xs)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--color-text-muted)", marginBottom: "var(--space-3)" }}>
            Key topological findings
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
            {reasoning.tda_findings.map((f, i) => (
              <div key={i} style={{
                padding: "var(--space-3)",
                background: "var(--color-surface-2)",
                borderRadius: "var(--radius-sm)",
                fontSize: "var(--text-xs)",
                color: "var(--color-text-secondary)",
                lineHeight: 1.6,
                borderLeft: "2px solid var(--color-accent)",
              }}
                dangerouslySetInnerHTML={{ __html: md(f.slice(0, 200) + (f.length > 200 ? "…" : "")) }}
              />
            ))}
          </div>
        </div>
      )}

      {reasoning.class_context && (
        <div>
          <p style={{ fontSize: "var(--text-xs)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--color-text-muted)", marginBottom: "var(--space-2)" }}>
            Condition overview
          </p>
          <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", lineHeight: 1.65 }}>{reasoning.class_context}</p>
        </div>
      )}

      {reasoning.confidence_note && (
        <div style={{
          borderLeft: "3px solid var(--color-warn-border)",
          background: "var(--color-warn-bg)",
          color: "var(--color-warn-text)",
          padding: "var(--space-3) var(--space-4)",
          borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
          fontSize: "var(--text-xs)",
          lineHeight: 1.5,
        }}>
          {reasoning.confidence_note}
        </div>
      )}
    </div>
  );
}
