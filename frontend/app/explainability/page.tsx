import Link from "next/link";

const XAI_METHODS = [
  {
    name: "Grad-CAM",
    tag: "Class Activation Mapping",
    how: "Computes the gradient of the target class score with respect to ResNet-18 layer4. Gradients are globally average-pooled to weight the feature maps, producing a spatial attention map.",
    strength: "Fast, low memory, works on any CNN.",
    weakness: "Coarse resolution — 7×7 before upsampling.",
  },
  {
    name: "Grad-CAM++",
    tag: "Improved localisation",
    how: "Uses second-order gradient statistics (α weights) to sharpen boundaries. Applies ReLU-filtered positive gradients only.",
    strength: "Sharper spatial precision than standard Grad-CAM.",
    weakness: "Slightly higher compute; still tied to one layer.",
  },
  {
    name: "Integrated Gradients",
    tag: "Axiomatic pixel attribution",
    how: "Integrates gradient w.r.t. each pixel along a path from a black baseline to the input (50 steps). Satisfies completeness and sensitivity axioms.",
    strength: "Theoretically grounded, pixel-level precision.",
    weakness: "50 forward+backward passes per image.",
  },
  {
    name: "Guided Backpropagation",
    tag: "Edge-level attribution",
    how: "Modifies ReLU backward pass to propagate only positive gradients through positive activations. Produces sharp, high-resolution gradient maps.",
    strength: "Pixel-perfect detail, visually sharp.",
    weakness: "Not class-discriminative — same map for any class.",
  },
];

const ARCH_BLOCKS = [
  { label: "Input\n224×224 RGB",  dim: "" },
  { label: "ResNet-18\nBackbone", dim: "" },
  { label: "512-d\nCNN features", dim: "512" },
  { label: "TDA\nExtractor",      dim: "" },
  { label: "13-d\nTDA features",  dim: "13" },
  { label: "Concat\n525-d",       dim: "525" },
  { label: "MLP\n256 → 5",        dim: "" },
  { label: "Prediction\n+ probs", dim: "" },
];

const TDA_GROUPS = [
  {
    label: "H0 — Connected components",
    features: [
      { name: "Persistence Entropy H0",   meaning: "Complexity of tissue regions. High → irregular, fragmented membrane." },
      { name: "Bottleneck Amplitude H0",   meaning: "Largest dominant region. High → consolidated opacity (earwax, exudate)." },
      { name: "Wasserstein Amplitude H0",  meaning: "Total tissue complexity. High → multiple distinct regions (chronic OM)." },
      { name: "Betti H0 @ t1 / t2 / t3",  meaning: "Component count at three filtration levels. Tracks tissue connectivity." },
      { name: "Persistence Landscape H0",  meaning: "Total component persistence weight. High → extensive tissue involvement." },
    ],
  },
  {
    label: "H1 — Loops & cavities",
    features: [
      { name: "Persistence Entropy H1",   meaning: "Loop complexity. Elevated → fluid pockets or middle ear effusion." },
      { name: "Bottleneck Amplitude H1",   meaning: "Dominant circular feature. High → annulus or perforation boundary." },
      { name: "Wasserstein Amplitude H1",  meaning: "Total loop energy. High → tympanosclerosis ring structures." },
      { name: "Betti H1 @ t1 / t2 / t3",  meaning: "Loop count across filtration. Non-zero at t3 → persistent perforation." },
    ],
  },
];

const QUALITY_METRICS = [
  { name: "AUC-Deletion",  desc: "Remove top-k% attended pixels progressively. Confidence should drop sharply. Lower AUC = better explanation." },
  { name: "AUC-Insertion", desc: "Start from blank, insert most-attended pixels progressively. Confidence should rise quickly. Higher AUC = better." },
  { name: "Stability",     desc: "Pearson correlation between original and noise-perturbed (σ=0.05) explanations. Higher = more robust." },
  { name: "IoU / Dice",    desc: "Spatial overlap between binarised heatmap and clinician annotation mask. Not yet computed — no annotation dataset." },
];

export default function ExplainabilityPage() {
  return (
    <>
      <style>{`
        .xai-method { padding: var(--space-6); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); box-shadow: var(--shadow-card); }
        .arch-block { background: var(--color-surface-2); border: 1px solid var(--color-border); border-radius: var(--radius-sm); padding: var(--space-3) var(--space-4); text-align: center; white-space: pre; font-size: var(--text-xs); font-family: var(--font-body); color: var(--color-text-secondary); line-height: 1.5; min-width: 80px; }
        .arch-arrow { color: var(--color-text-muted); font-size: var(--text-sm); flex-shrink: 0; }
        .tda-feature-row { display: flex; gap: var(--space-5); padding: var(--space-3) 0; border-bottom: 1px solid var(--color-border); align-items: baseline; }
        .btn-primary { background: var(--color-accent); color: #fff; font-family: var(--font-body); font-weight: 500; font-size: var(--text-sm); padding: var(--space-3) var(--space-6); border-radius: var(--radius-sm); text-decoration: none; transition: background var(--transition-fast); display: inline-block; }
        .btn-primary:hover { background: var(--color-accent-light); }
        @media (max-width: 768px) {
          .xai-grid { grid-template-columns: 1fr !important; }
          .tda-grid  { grid-template-columns: 1fr !important; }
          .metric-grid { grid-template-columns: 1fr !important; }
          .arch-row { flex-wrap: wrap; }
        }
      `}</style>

      <div style={{ maxWidth: 1120, margin: "0 auto", padding: "0 var(--space-6)" }}>

        {/* ── Header ───────────────────────────────────────────────── */}
        <section style={{ paddingTop: "var(--space-9)", paddingBottom: "var(--space-8)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-accent)",
            marginBottom: "var(--space-4)",
          }}>Explainability</p>
          <h1 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "clamp(28px, 4vw, var(--text-4xl))",
            letterSpacing: "-0.02em", lineHeight: 1.15,
            color: "var(--color-text-primary)", maxWidth: 560,
            marginBottom: "var(--space-4)",
          }}>
            Four methods. One shared goal.
          </h1>
          <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-muted)", maxWidth: 480, lineHeight: 1.65 }}>
            Every prediction is accompanied by visual attribution maps and topological feature scores
            that show which image regions and structural features drove the classification.
          </p>
        </section>

        {/* ── Architecture ─────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)",
            marginBottom: "var(--space-5)",
          }}>Model architecture</p>

          <div className="arch-row" style={{
            display: "flex", alignItems: "center",
            gap: "var(--space-2)", flexWrap: "wrap",
            padding: "var(--space-6)",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            boxShadow: "var(--shadow-card)",
          }}>
            {ARCH_BLOCKS.map((b, i) => (
              <div key={b.label} style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <div className="arch-block">{b.label}</div>
                {i < ARCH_BLOCKS.length - 1 && <span className="arch-arrow">→</span>}
              </div>
            ))}
          </div>
        </section>

        {/* ── XAI methods ──────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)",
            marginBottom: "var(--space-4)",
          }}>Visual attribution</p>
          <h2 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "var(--text-3xl)", marginBottom: "var(--space-7)",
            letterSpacing: "-0.01em",
          }}>Visual explanation methods.</h2>

          <div className="xai-grid" style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-4)",
          }}>
            {XAI_METHODS.map((m) => (
              <div key={m.name} className="xai-method">
                <p style={{
                  fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
                  textTransform: "uppercase", color: "var(--color-accent)",
                  marginBottom: "var(--space-2)",
                }}>{m.tag}</p>
                <h3 style={{
                  fontFamily: "var(--font-display)", fontWeight: 400,
                  fontSize: "var(--text-xl)", color: "var(--color-text-primary)",
                  marginBottom: "var(--space-3)",
                }}>{m.name}</h3>
                <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", lineHeight: 1.65, marginBottom: "var(--space-4)" }}>{m.how}</p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-3)" }}>
                  <div style={{ background: "var(--color-surface-2)", borderRadius: "var(--radius-sm)", padding: "var(--space-3)" }}>
                    <p style={{ fontSize: "var(--text-xs)", fontWeight: 500, color: "var(--color-accent)", marginBottom: "var(--space-1)" }}>Strength</p>
                    <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-secondary)" }}>{m.strength}</p>
                  </div>
                  <div style={{ background: "var(--color-surface-2)", borderRadius: "var(--radius-sm)", padding: "var(--space-3)" }}>
                    <p style={{ fontSize: "var(--text-xs)", fontWeight: 500, color: "var(--color-text-muted)", marginBottom: "var(--space-1)" }}>Limitation</p>
                    <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-secondary)" }}>{m.weakness}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── TDA features ─────────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)",
            marginBottom: "var(--space-4)",
          }}>Topological features</p>
          <h2 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "var(--text-3xl)", marginBottom: "var(--space-3)",
            letterSpacing: "-0.01em",
          }}>What the topology measures.</h2>
          <p style={{
            fontSize: "var(--text-sm)", color: "var(--color-text-muted)",
            maxWidth: 480, marginBottom: "var(--space-7)", lineHeight: 1.65,
          }}>
            Each feature is scored by input occlusion — masking it to zero and
            measuring the resulting drop in prediction confidence.
          </p>

          <div className="tda-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-8)" }}>
            {TDA_GROUPS.map((g) => (
              <div key={g.label}>
                <p style={{
                  fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
                  textTransform: "uppercase", color: "var(--color-accent)",
                  marginBottom: "var(--space-4)",
                }}>{g.label}</p>
                {g.features.map((f) => (
                  <div key={f.name} className="tda-feature-row">
                    <span style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", minWidth: 200, flexShrink: 0 }}>{f.name}</span>
                    <span style={{ fontSize: "var(--text-sm)", color: "var(--color-text-muted)", lineHeight: 1.5 }}>{f.meaning}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </section>

        {/* ── Quality metrics ───────────────────────────────────────── */}
        <section style={{ marginBottom: "var(--section-gap)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-text-muted)",
            marginBottom: "var(--space-4)",
          }}>Explanation quality</p>
          <h2 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "var(--text-3xl)", marginBottom: "var(--space-7)",
            letterSpacing: "-0.01em",
          }}>How we measure whether explanations are faithful.</h2>

          <div className="metric-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-4)" }}>
            {QUALITY_METRICS.map((m) => (
              <div key={m.name} style={{
                padding: "var(--space-5)",
                background: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                boxShadow: "var(--shadow-card)",
              }}>
                <h3 style={{ fontWeight: 500, fontSize: "var(--text-base)", color: "var(--color-text-primary)", marginBottom: "var(--space-2)" }}>{m.name}</h3>
                <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-secondary)", lineHeight: 1.65 }}>{m.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── CTA ──────────────────────────────────────────────────── */}
        <section style={{ paddingBottom: "var(--section-gap)" }}>
          <Link href="/demo" className="btn-primary">See it in the demo →</Link>
        </section>

      </div>
    </>
  );
}
