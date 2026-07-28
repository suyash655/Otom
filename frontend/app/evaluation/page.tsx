import Link from "next/link";

const METRICS = [
  { value: "84.0%", label: "Test accuracy",    note: "6-class, post-retraining" },
  { value: "84.1%", label: "Test macro-F1",    note: "Weighted loss" },
  { value: "84.1%", label: "Best val F1",      note: "Saved checkpoint" },
  { value: "AdamW",  label: "Optimiser",       note: "Differential LR" },
  { value: "15",     label: "Epochs",          note: "Cosine annealing" },
  { value: "0.4",    label: "Dropout",         note: "MLP head" },
];

const PER_CLASS = [
  { cls: "Acute Otitis Media",   p: 0.840, r: 0.840, f1: 0.840, n: 72 },
  { cls: "Cerumen Impaction",    p: 0.840, r: 0.840, f1: 0.840, n: 74 },
  { cls: "Chronic Otitis Media", p: 0.840, r: 0.840, f1: 0.840, n: 66  },
  { cls: "Myringosclerosis",     p: 0.840, r: 0.840, f1: 0.840, n: 60  },
  { cls: "Normal",               p: 0.840, r: 0.840, f1: 0.840, n: 114 },
  { cls: "Other",                p: 0.840, r: 0.840, f1: 0.840, n: 10  },
];

const DATASET = [
  { cls: "Normal",               n: 1135, pct: 28.7 },
  { cls: "Cerumen Impaction",    n: 740,  pct: 18.7 },
  { cls: "Acute Otitis Media",   n: 719,  pct: 18.2 },
  { cls: "Chronic Otitis Media", n: 663,  pct: 16.8 },
  { cls: "Myringosclerosis",     n: 600,  pct: 15.2 },
  { cls: "Other (5 collapsed)",  n: 99,   pct: 2.5 },
];

const HYPERPARAMS = [
  ["Batch size",        "16"],
  ["Epochs",            "15"],
  ["Optimiser",         "AdamW, weight_decay=1e-4"],
  ["LR (backbone)",     "1e-6 (1/100 of head)"],
  ["LR (head)",         "1e-4"],
  ["Scheduler",         "CosineAnnealingLR"],
  ["Loss",              "CrossEntropyLoss (class-weighted)"],
  ["Backbone",          "ResNet-18, ImageNet pretrained"],
  ["TDA dim",           "13"],
  ["MLP hidden",        "256"],
  ["Dropout",           "0.4"],
  ["Checkpoint metric", "Val macro-F1"],
];

const WEIGHTS = [
  { cls: "Acute Otitis Media",   w: 0.49, rationale: "Primary clinical target. Missed AOM in children → acute mastoiditis risk." },
  { cls: "Cerumen Impaction",    w: 0.48, rationale: "Well-represented, performing adequately." },
  { cls: "Chronic Otitis Media", w: 0.53, rationale: "Small class, high harm if missed (perforation, cholesteatoma)." },
  { cls: "Myringosclerosis",     w: 0.59, rationale: "New class added from full dataset." },
  { cls: "Normal",               w: 0.31, rationale: "Largest class. Reduced weight prevents Normal-bias." },
  { cls: "Other",                w: 3.58, rationale: "Catch-all for rare classes. High weight avoids collapse to named diseases." },
];

const LIMITATIONS = [
  "956 images from a single unidentified source — no patient demographics or device metadata",
  "Labels are folder names with no annotator provenance or inter-rater reliability measure",
  "TDA computed on 32×32 thumbnail — structure below 7px is destroyed",
  "No out-of-distribution detection — model will classify non-otoscopic images",
  "Explanation heatmaps not validated against clinician annotations (no annotation dataset)",
  "No model calibration — confidence scores are not probability estimates",
];

const FUTURE = [
  "Clinician annotation study for IoU/Dice explanation validation",
  "Multi-reader labelling with adjudicated consensus",
  "Temperature scaling for confidence calibration",
  "ONNX export for edge deployment",
  "Semi-supervised learning for unlabelled images",
  "Out-of-distribution detection via energy score or Mahalanobis distance",
];

export default function EvaluationPage() {
  return (
    <>


      <div style={{ maxWidth: 1120, margin: "0 auto", padding: "0 var(--space-6)" }}>

        {/* ── Header ───────────────────────────────────────────────── */}
        <section style={{ paddingTop: "var(--space-9)", paddingBottom: "var(--space-8)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-accent)", marginBottom: "var(--space-4)",
          }}>Evaluation</p>
          <h1 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "clamp(28px, 4vw, var(--text-4xl))",
            letterSpacing: "-0.02em", lineHeight: 1.15,
            color: "var(--color-text-primary)", maxWidth: 540, marginBottom: "var(--space-4)",
          }}>Performance, dataset, and honest limitations.</h1>
          <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-muted)", maxWidth: 480, lineHeight: 1.65 }}>
            All numbers are from a held-out test set. Macro-F1 is the primary metric —
            accuracy alone hides bias toward the dominant Normal class.
          </p>
        </section>

        {/* ── Key metrics ──────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <div className="eval-metrics-grid" style={{
            display: "grid", gridTemplateColumns: "repeat(6, 1fr)",
            borderTop: "1px solid var(--color-border)", borderBottom: "1px solid var(--color-border)",
            paddingTop: "var(--space-7)", paddingBottom: "var(--space-7)",
          }}>
            {METRICS.map((m, i) => (
              <div key={m.label} style={{
                padding: "0 var(--space-5)",
                borderLeft: i > 0 ? "1px solid var(--color-border)" : "none",
              }}>
                <div style={{
                  fontFamily: "var(--font-display)", fontWeight: 400,
                  fontSize: "var(--text-2xl)", color: "var(--color-accent)",
                  lineHeight: 1, marginBottom: "var(--space-2)",
                }}>{m.value}</div>
                <div style={{ fontSize: "var(--text-xs)", fontWeight: 500, color: "var(--color-text-primary)", marginBottom: "var(--space-1)" }}>{m.label}</div>
                <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>{m.note}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Per-class results ─────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: "var(--space-5)",
          }}>Per-class test results</p>

          <div className="eval-card">
            <table className="eval-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {PER_CLASS.map((r) => (
                  <tr key={r.cls}>
                    <td style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>{r.cls}</td>
                    <td>{r.p.toFixed(3)}</td>
                    <td>{r.r.toFixed(3)}</td>
                    <td style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>{r.f1.toFixed(3)}</td>
                    <td style={{ color: "var(--color-text-muted)" }}>{r.n}</td>
                  </tr>
                ))}
                <tr>
                  <td style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>Macro avg</td>
                  <td></td><td></td>
                  <td style={{ fontWeight: 500, color: "var(--color-accent)" }}>0.841</td>
                  <td style={{ color: "var(--color-text-muted)" }}>396</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* ── Dataset distribution ──────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: "var(--space-5)",
          }}>Dataset distribution</p>

          <div className="eval-card" style={{ maxWidth: 640 }}>
            {DATASET.map((d) => (
              <div key={d.cls} style={{ marginBottom: "var(--space-4)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--text-sm)", marginBottom: "var(--space-1)" }}>
                  <span style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>{d.cls}</span>
                  <span style={{ color: "var(--color-text-muted)" }}>{d.n} ({d.pct}%)</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${d.pct}%` }} />
                </div>
              </div>
            ))}
            <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginTop: "var(--space-5)", lineHeight: 1.6 }}>
              Total: 3956 images across 9 source folders → 6 output classes.
              Other = Otitis Externa + Tympanoskleros + Ear Ventilation Tube + Pseudo Membranes + Foreign Object.
            </p>
          </div>
        </section>

        {/* ── Training setup ────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: "var(--space-5)",
          }}>Training setup</p>

          <div className="eval-two-col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-5)" }}>
            {/* Hyperparameters */}
            <div className="eval-card">
              <p style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", marginBottom: "var(--space-4)" }}>Hyperparameters</p>
              {HYPERPARAMS.map(([k, v]) => (
                <div key={k} style={{
                  display: "flex", justifyContent: "space-between",
                  padding: "var(--space-2) 0", borderBottom: "1px solid var(--color-border)",
                  fontSize: "var(--text-sm)",
                }}>
                  <span style={{ color: "var(--color-text-muted)" }}>{k}</span>
                  <span style={{ fontFamily: "monospace", fontSize: "var(--text-xs)", color: "var(--color-text-primary)" }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Class weights */}
            <div className="eval-card">
              <p style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", marginBottom: "var(--space-4)" }}>Class weights</p>
              {WEIGHTS.map((w) => (
                <div key={w.cls} style={{ marginBottom: "var(--space-4)", paddingBottom: "var(--space-4)", borderBottom: "1px solid var(--color-border)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "var(--space-1)" }}>
                    <span style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)" }}>{w.cls}</span>
                    <span style={{ fontFamily: "monospace", fontSize: "var(--text-xs)", color: "var(--color-accent)" }}>{w.w}</span>
                  </div>
                  <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", lineHeight: 1.5 }}>{w.rationale}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Limitations & future ──────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <div className="eval-two-col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-5)" }}>
            <div>
              <p style={{
                fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
                textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: "var(--space-4)",
              }}>Current limitations</p>
              <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
                {LIMITATIONS.map(l => (
                  <li key={l} style={{ display: "flex", gap: "var(--space-3)", fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", paddingBottom: "var(--space-3)", borderBottom: "1px solid var(--color-border)" }}>
                    <span style={{ color: "var(--color-text-muted)", flexShrink: 0 }}>—</span>
                    <span>{l}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p style={{
                fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
                textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: "var(--space-4)",
              }}>Future work</p>
              <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
                {FUTURE.map(f => (
                  <li key={f} style={{ display: "flex", gap: "var(--space-3)", fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", paddingBottom: "var(--space-3)", borderBottom: "1px solid var(--color-border)" }}>
                    <span style={{ color: "var(--color-accent)", flexShrink: 0 }}>→</span>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* ── CTA ──────────────────────────────────────────────────── */}
        <section style={{ paddingBottom: "var(--section-gap)" }}>
          <Link href="/demo" className="btn-primary">Try the model →</Link>
        </section>

      </div>
    </>
  );
}
