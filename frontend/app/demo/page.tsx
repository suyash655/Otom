"use client";

import { useCallback, useState } from "react";
import { useExplanation } from "../hooks/useApi";
import { useFileHandler } from "../hooks/useFileHandler";
import { useHealthCheck } from "../hooks/useHealthCheck";
import HeatmapViewer from "../../components/HeatmapViewer";
import ClinicalCard from "../../components/ClinicalCard";
import ErrorBoundary from "../../components/ErrorBoundary";
import { CLASS_ORDER, CLASS_COLOR_HEX } from "../../lib/config";

export default function DemoPage() {
  const { status, result, error, explainImage, reset: resetExplanation } = useExplanation();
  const { file, preview, dragging, inputRef, onFileChange, onDrop, onDragOver, onDragLeave, reset: resetFile, triggerFileSelect } = useFileHandler();
  const apiOk = useHealthCheck();

  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const handleAnalyze = async () => {
    if (!file) return;
    const startTime = performance.now();
    await explainImage(file);
    setLatencyMs(performance.now() - startTime);
  };

  const handleReset = () => {
    resetExplanation();
    resetFile();
  };

  // Reset explanation when file changes
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFileChange(e);
    resetExplanation();
  };

  // Reset explanation when file is dropped
  const handleDrop = (e: React.DragEvent) => {
    onDrop(e);
    resetExplanation();
  };

  const heatmapMethods = result ? [
    { label: "Grad-CAM",               b64: result.heatmap_gradcam },
    { label: "Grad-CAM++",             b64: result.heatmap_gradcam_pp },
    { label: "Integrated Gradients",   b64: result.heatmap_int_grads },
    { label: "Guided Backpropagation", b64: result.heatmap_guided_bp },
  ] : [];

  const flagForReview = () => { alert("Image flagged for clinical review."); };

  const exportPdf = async () => {
    try {
      const { jsPDF } = await import("jspdf");
      const html2canvas = (await import("html2canvas")).default;
      const element = document.getElementById("demo-results");
      if (!element) return;
      const canvas = await html2canvas(element);
      const pdf = new jsPDF("p", "mm", "a4");
      pdf.addImage(canvas.toDataURL("image/png"), "PNG", 0, 0, pdf.internal.pageSize.getWidth(), (canvas.height * pdf.internal.pageSize.getWidth()) / canvas.width);
      pdf.save(`OtoScope_Report_${Date.now()}.pdf`);
    } catch (e) { console.error("PDF export failed:", e); }
  };

  const sortedTDA = result
    ? Object.entries(result.tda_importance).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <>


      <div style={{ maxWidth: 1120, margin: "0 auto", padding: "var(--space-8) var(--space-6)" }}>

        {/* Header */}
        <div style={{ marginBottom: "var(--space-7)" }}>
          <p style={{
            fontWeight: 500, fontSize: "var(--text-xs)", letterSpacing: "0.08em",
            textTransform: "uppercase", color: "var(--color-accent)", marginBottom: "var(--space-3)",
          }}>Demo</p>
          <h1 style={{
            fontFamily: "var(--font-display)", fontWeight: 300,
            fontSize: "var(--text-3xl)", letterSpacing: "-0.01em",
            color: "var(--color-text-primary)", marginBottom: "var(--space-3)",
          }}>Upload an otoscopic image.</h1>
          <p style={{ fontSize: "var(--text-sm)", color: "var(--color-text-muted)" }}>
            PNG or JPEG. The model returns a prediction, attribution heatmaps, and TDA feature scores.
          </p>
        </div>

        {/* Research disclaimer */}
        <div className="warn-bar" style={{ marginBottom: "var(--space-5)" }}>
          <strong>Research use only.</strong>{" "}
          Output has not been clinically validated and must not inform diagnosis or patient care.
        </div>

        {/* Backend status */}
        {apiOk === false && (
          <div style={{
            marginBottom: "var(--space-5)", padding: "var(--space-3) var(--space-4)",
            background: "#FEF2F2", border: "1px solid #FECACA",
            borderRadius: "var(--radius-sm)", fontSize: "var(--text-sm)", color: "#991B1B",
          }}>
            ⚠ Backend not reachable at localhost:8000.
            Run: <code style={{ fontFamily: "monospace", fontSize: "var(--text-xs)" }}>.venv\Scripts\uvicorn backend.app:app --reload --port 8000</code>
          </div>
        )}
        {apiOk === true && (
          <div style={{
            marginBottom: "var(--space-5)", padding: "var(--space-2) var(--space-4)",
            background: "#F0FDF4", border: "1px solid #BBF7D0",
            borderRadius: "var(--radius-sm)", fontSize: "var(--text-xs)", color: "#166534",
          }}>
            ✓ Backend connected
          </div>
        )}

        {/* Main grid */}
        <div className="demo-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "var(--space-6)" }}>

          {/* ── Left: upload + prediction ─────────────────────────── */}
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>

            {/* Drop zone */}
            <div
              className={`demo-dropzone${dragging ? " dragging" : ""}`}
              onDrop={handleDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={triggerFileSelect}
            >
              <input ref={inputRef} type="file" accept="image/*, .dcm" style={{ display: "none" }} onChange={handleFileChange} />
              {preview ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img src={preview} alt="Preview" style={{ maxHeight: 240, margin: "0 auto", display: "block", borderRadius: "var(--radius-md)", objectFit: "contain" }} />
              ) : (
                <div>
                  <p style={{ fontSize: "var(--text-sm)", fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: "var(--space-2)" }}>
                    Drop image here
                  </p>
                  <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>or click to browse · PNG / JPEG</p>
                </div>
              )}
            </div>

            {file && (
              <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>
                {file.name} · {(file.size / 1024).toFixed(1)} KB
              </p>
            )}

            <button
              className="demo-btn"
              onClick={handleAnalyze}
              disabled={!file || status === "loading"}
            >
              {status === "loading" ? "Analysing…" : "Analyse image"}
            </button>

            {status !== "idle" && (
              <button
                className="demo-btn"
                onClick={handleReset}
                style={{ background: "var(--color-surface-2)", color: "var(--color-text-secondary)" }}
              >
                Reset
              </button>
            )}

            {status === "error" && (
              <div style={{
                padding: "var(--space-3) var(--space-4)",
                background: "#FEF2F2", border: "1px solid #FECACA",
                borderRadius: "var(--radius-sm)", fontSize: "var(--text-sm)", color: "#991B1B",
              }}>{error}</div>
            )}

            {/* Prediction card */}
            {result && (
              <div className="panel">
                {/* Predicted class */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "var(--space-4)" }}>
                  <div>
                    <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginBottom: "var(--space-1)" }}>Prediction</p>
                    <p style={{
                      fontFamily: "var(--font-display)", fontWeight: 400,
                      fontSize: "var(--text-xl)", color: "var(--color-text-primary)",
                    }}>{result.predicted_class}</p>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginBottom: "var(--space-1)" }}>Confidence</p>
                    <p style={{
                      fontFamily: "var(--font-display)", fontWeight: 400,
                      fontSize: "var(--text-2xl)", color: "var(--color-accent)",
                    }}>{(result.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>

                {result.uncertainty !== undefined && (
                  <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginBottom: "var(--space-4)" }}>
                    MC Dropout uncertainty ±{(result.uncertainty * 100).toFixed(2)}%
                  </p>
                )}

                {/* Safety flags */}
                {result.low_confidence && (
                  <div className="low-conf-bar" style={{ marginBottom: "var(--space-4)" }}>
                    <strong>Low confidence ({(result.confidence * 100).toFixed(1)}%)</strong> — do not act on this result without clinical examination.
                  </div>
                )}
                {result.uncertain && !result.low_confidence && (
                  <div className="low-conf-bar" style={{ marginBottom: "var(--space-4)" }}>
                    <strong>High uncertainty</strong> — prediction unstable across MC Dropout passes. Image may be out-of-distribution.
                  </div>
                )}

                {/* Confidence bars */}
                <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
                  {CLASS_ORDER.map((cls) => {
                    const prob = result.class_probs[cls] ?? 0;
                    const active = cls === result.predicted_class;
                    const color = CLASS_COLOR_HEX[cls] ?? "#A8A29E";
                    return (
                      <div key={cls}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--text-xs)" }}>
                          <span style={{ color: active ? "var(--color-text-primary)" : "var(--color-text-muted)", fontWeight: active ? 500 : 400 }}>{cls}</span>
                          <span style={{ color: active ? "var(--color-accent)" : "var(--color-text-muted)" }}>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="conf-bar-track">
                          <div className="conf-bar" style={{ width: `${prob * 100}%`, height: "100%", background: color, borderRadius: 3, opacity: active ? 1 : 0.5 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* ── Right: heatmaps + TDA + clinical ─────────────────── */}
          {status === "loading" ? (
            <div className="flex flex-col gap-4 animate-pulse">
              <div className="panel h-64 bg-slate-100 rounded-md"></div>
              <div className="panel h-48 bg-slate-100 rounded-md"></div>
              <div className="panel h-32 bg-slate-100 rounded-md"></div>
            </div>
          ) : result ? (
            <ErrorBoundary fallback={<div className="panel text-red-500">Failed to render results.</div>}>
              <div id="demo-results" style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
                <div style={{ display: "flex", gap: "var(--space-3)", marginBottom: "var(--space-2)" }}>
                   <button onClick={flagForReview} className="demo-btn" style={{ background: "#FEF2F2", color: "#991B1B", border: "1px solid #FECACA", flex: 1 }}>🚩 Flag for Review</button>
                   <button onClick={exportPdf} className="demo-btn" style={{ background: "#EFF6FF", color: "#1D4ED8", border: "1px solid #BFDBFE", flex: 1 }}>📄 Export PDF Report</button>
                </div>

                {/* Heatmaps */}
                <div className="panel">
                  <p style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", marginBottom: "var(--space-3)" }}>
                    Attribution heatmaps
                  </p>
                  <HeatmapViewer methods={heatmapMethods} />
                </div>

                {/* TDA importance */}
                <div className="panel">
                  <p style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", marginBottom: "var(--space-1)" }}>
                    TDA feature importance
                  </p>
                  <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginBottom: "var(--space-4)" }}>
                    Scored by occlusion — how much does zeroing each feature drop confidence?
                  </p>
                  <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
                    {sortedTDA.slice(0, 8).map(([name, imp]) => (
                      <div key={name}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--text-xs)" }}>
                          <span style={{ fontFamily: "monospace", color: "var(--color-text-secondary)" }}>{name}</span>
                          <span style={{ color: "var(--color-accent)", fontWeight: 500 }}>{(imp * 100).toFixed(0)}%</span>
                        </div>
                        <div className="tda-bar-track">
                          <div style={{ width: `${imp * 100}%`, height: "100%", background: name.includes("H1") ? "#9B59B6" : "var(--color-accent)", borderRadius: 2, opacity: 0.75 }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", marginTop: "var(--space-3)" }}>
                    <span style={{ color: "var(--color-accent)" }}>■</span> H0 connected components &nbsp;
                    <span style={{ color: "#9B59B6" }}>■</span> H1 loops / cavities
                  </p>
                </div>

                {/* Clinical reasoning */}
                <div className="panel">
                  <p style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", marginBottom: "var(--space-3)" }}>
                    Clinical reasoning
                  </p>
                  {/* Disclaimer inline */}
                  <div className="warn-bar" style={{ marginBottom: "var(--space-4)" }}>
                    Algorithmically generated from feature values. Not validated by clinicians.
                  </div>
                  <ClinicalCard reasoning={result.clinical_reasoning} />
                </div>

              </div>
            </ErrorBoundary>
          ) : (
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "center",
              minHeight: 320, color: "var(--color-text-muted)", fontSize: "var(--text-sm)",
              background: "var(--color-surface)", border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
            }}>
              Results appear here after analysis
            </div>
          )}
        </div>

        {/* Full comparison chart */}
        {result && (
          <div className="panel" style={{ marginTop: "var(--space-6)" }}>
            <p style={{ fontWeight: 500, fontSize: "var(--text-sm)", color: "var(--color-text-primary)", marginBottom: "var(--space-3)" }}>
              All methods comparison
            </p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`data:image/png;base64,${result.chart_comparison}`} alt="Comparison" style={{ width: "100%", borderRadius: "var(--radius-sm)" }} />
          </div>
        )}

      </div>
    </>
  );
}
