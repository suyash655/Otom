import Link from "next/link";

const METRICS = [
  { value: "84.1%", label: "Test macro-F1",   note: "Weighted, 6 classes" },
  { value: "84.0%", label: "Test accuracy",   note: "Post-retraining" },
  { value: "13",    label: "TDA features",    note: "H0 + H1 homology" },
  { value: "3956",  label: "Labelled images", note: "Full dataset" },
];

const STEPS = [
  { n: "01", title: "Upload",         body: "Any otoscopic PNG or JPEG. Preprocessed with ImageNet normalisation at 224×224." },
  { n: "02", title: "CNN backbone",   body: "ResNet-18 extracts a 512-d visual feature vector from the image." },
  { n: "03", title: "TDA extraction", body: "Persistent homology computes 13 topological features from a grayscale thumbnail." },
  { n: "04", title: "Fusion",         body: "CNN and TDA features are concatenated and passed to a weighted MLP classifier." },
  { n: "05", title: "Explanation",    body: "Grad-CAM++, Integrated Gradients, and feature occlusion generate attribution maps." },
];

const CONDITIONS = [
  { name: "Normal",               desc: "Healthy pearly-grey tympanic membrane, light reflex intact." },
  { name: "Acute Otitis Media",   desc: "Bulging, erythematous membrane — bacterial or viral middle ear infection." },
  { name: "Cerumen Impaction",    desc: "Earwax mass obstructing the canal, varying texture and colour." },
  { name: "Chronic Otitis Media", desc: "Persistent infection, often with visible perforation or discharge." },
  { name: "Myringosclerosis",     desc: "Calcification of the tympanic membrane, appearing as white patches." },
  { name: "Other / Atypical",     desc: "Otitis externa, tympanosclerosis, ventilation tube, pseudo membranes, or foreign object." },
];

const TDA_FEATURES = [
  "Persistence Entropy H0 / H1",
  "Bottleneck Amplitude H0 / H1",
  "Wasserstein Amplitude H0 / H1",
  "Betti Numbers at 3 thresholds",
  "Persistence Landscape H0",
];

export default function Home() {
  return (
    <>


      <div style={{ maxWidth: 1120, margin: "0 auto", padding: "0 var(--space-6)" }}>

        {/* ── Research disclaimer ─────────────────────────────────── */}
        <div className="animate-in delay-1" style={{
          marginTop: "var(--space-7)",
          borderLeft: "3px solid var(--color-warn-border)",
          background: "var(--color-warn-bg)",
          color: "var(--color-warn-text)",
          padding: "var(--space-3) var(--space-5)",
          borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
          fontSize: "var(--text-sm)",
          lineHeight: 1.5,
        }}>
          <strong>Research use only.</strong>{" "}
          This prototype has not been clinically validated and must not inform diagnosis, treatment, or patient care.
        </div>

        {/* ── Hero ─────────────────────────────────────────────────── */}
        <section style={{ paddingTop: "var(--space-9)", paddingBottom: "var(--space-9)" }}>
          <p className="animate-in delay-1" style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-accent)",
            marginBottom: "var(--space-4)",
          }}>
            Hybrid CNN + Topological Data Analysis
          </p>
          <h1 className="animate-in delay-2" style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "clamp(32px, 5vw, var(--text-4xl))",
            lineHeight: 1.15, letterSpacing: "-0.02em",
            color: "var(--color-text-primary)", maxWidth: 600,
            marginBottom: "var(--space-5)",
          }}>
            Otoscopy classification that explains its reasoning.
          </h1>
          <p className="animate-in delay-3" style={{
            fontSize: "var(--text-lg)", color: "var(--color-text-secondary)",
            maxWidth: 480, lineHeight: 1.65, marginBottom: "var(--space-7)",
          }}>
            A research prototype combining deep visual features with persistent homology
            to classify ear conditions — and surface the topological evidence behind each prediction.
          </p>
          <div className="animate-in delay-4 home-hero-btns" style={{ display: "flex", gap: "var(--space-3)" }}>
            <Link href="/demo" className="btn-primary">Try the demo</Link>
            <Link href="/explainability" className="btn-secondary">How it works</Link>
          </div>
        </section>

        {/* ── Metrics ──────────────────────────────────────────────── */}
        <section className="animate-in delay-3 home-metrics" style={{
          borderTop: "1px solid var(--color-border)",
          borderBottom: "1px solid var(--color-border)",
          paddingTop: "var(--space-7)", paddingBottom: "var(--space-7)",
          display: "grid", gridTemplateColumns: "repeat(4, 1fr)",
          marginBottom: "var(--section-gap)",
        }}>
          {METRICS.map((m, i) => (
            <div key={m.label} style={{
              padding: "0 var(--space-6)",
              borderLeft: i > 0 ? "1px solid var(--color-border)" : "none",
            }}>
              <div style={{
                fontFamily: "var(--font-display)", fontWeight: 400,
                fontSize: "var(--text-3xl)", color: "var(--color-accent)",
                lineHeight: 1, marginBottom: "var(--space-2)",
              }}>{m.value}</div>
              <div style={{ fontSize: "var(--text-sm)", fontWeight: 500, color: "var(--color-text-primary)", marginBottom: "var(--space-1)" }}>{m.label}</div>
              <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>{m.note}</div>
            </div>
          ))}
        </section>

        {/* ── How it works ─────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)",
            marginBottom: "var(--space-4)",
          }}>Pipeline</p>
          <h2 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "var(--text-3xl)", marginBottom: "var(--space-8)",
            letterSpacing: "-0.01em",
          }}>Five steps from image to explanation.</h2>

          <div style={{ display: "flex", flexDirection: "column" }}>
            {STEPS.map((s, i) => (
              <div key={s.n} className="home-step" style={{
                display: "grid",
                gridTemplateColumns: "40px 200px 1fr",
                gap: "var(--space-6)",
                alignItems: "start",
                padding: "var(--space-5) 0",
                borderTop: i === 0 ? "1px solid var(--color-border)" : "none",
                borderBottom: "1px solid var(--color-border)",
              }}>
                <span style={{
                  fontWeight: 500, fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)", paddingTop: 3,
                }}>{s.n}</span>
                <span style={{
                  fontWeight: 500, fontSize: "var(--text-base)",
                  color: "var(--color-text-primary)",
                }}>{s.title}</span>
                <span style={{
                  fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", lineHeight: 1.65,
                }}>{s.body}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ── Conditions ───────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)",
            marginBottom: "var(--space-4)",
          }}>Output classes</p>
          <h2 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "var(--text-3xl)", marginBottom: "var(--space-3)",
            letterSpacing: "-0.01em",
          }}>Six conditions. One honest catch-all.</h2>
          <p style={{
            fontSize: "var(--text-sm)", color: "var(--color-text-muted)",
            maxWidth: 480, marginBottom: "var(--space-7)", lineHeight: 1.65,
          }}>
            Classes with fewer than 50 training images are collapsed into Other
            so the model never silently forces an unknown pathology into a named disease.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
            {CONDITIONS.map((c) => (
              <div key={c.name} style={{
                display: "flex", alignItems: "baseline",
                gap: "var(--space-6)",
                padding: "var(--space-4) 0 var(--space-4) var(--space-4)",
                borderLeft: "2px solid var(--color-accent)",
                borderBottom: "1px solid var(--color-border)",
              }}>
                <span style={{
                  fontWeight: 500, fontSize: "var(--text-sm)",
                  color: "var(--color-text-primary)", minWidth: 200, flexShrink: 0,
                }}>{c.name}</span>
                <span style={{
                  fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", lineHeight: 1.55,
                }}>{c.desc}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ── TDA section ──────────────────────────────────────────── */}
        <section className="home-tda-grid" style={{
          marginBottom: "var(--section-gap)",
          display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: "var(--space-10)", alignItems: "start",
        }}>
          <div>
            <p style={{
              fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
              textTransform: "uppercase", color: "var(--color-text-muted)",
              marginBottom: "var(--space-4)",
            }}>Why TDA</p>
            <h2 style={{
              fontFamily: "var(--font-display)", fontWeight: 300,
              fontSize: "var(--text-3xl)", marginBottom: "var(--space-5)",
              letterSpacing: "-0.01em", lineHeight: 1.2,
            }}>Shape, not just texture.</h2>
            <p style={{
              fontSize: "var(--text-sm)", color: "var(--color-text-secondary)",
              lineHeight: 1.75, marginBottom: "var(--space-6)",
            }}>
              Standard CNNs classify based on texture and colour. Persistent homology captures
              the <em>topology</em> of the tympanic membrane — connected components (H0) and
              loop structures (H1) that correspond to perforations, fluid cavities, and
              structural irregularities invisible to pixel-based features.
            </p>
            <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              {[
                "H0 — connectivity, fragmentation, tissue boundary complexity",
                "H1 — loops, cavities, perforations, annular structures",
                "13 features extracted per image via ripser",
                "Features weighted by clinical harm asymmetry in loss function",
              ].map(t => (
                <li key={t} style={{ display: "flex", gap: "var(--space-3)", fontSize: "var(--text-sm)", color: "var(--color-text-secondary)" }}>
                  <span style={{ color: "var(--color-accent)", flexShrink: 0, marginTop: 2 }}>→</span>
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p style={{
              fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
              textTransform: "uppercase", color: "var(--color-text-muted)",
              marginBottom: "var(--space-5)",
            }}>Feature groups</p>
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              {TDA_FEATURES.map((f, i) => (
                <div key={f} className="tda-feature-card">
                  <span style={{
                    fontFamily: "monospace", fontSize: "var(--text-xs)",
                    color: "var(--color-accent)", marginRight: "var(--space-3)", fontWeight: 500,
                  }}>
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {f}
                </div>
              ))}
            </div>
          </div>
        </section>

      </div>
    </>
  );
}
