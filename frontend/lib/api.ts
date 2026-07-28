import { API_BASE_URL, API_TIMEOUT } from './constants';

const BASE = API_BASE_URL;

export interface PredictionResult {
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  class_probs: Record<string, number>;
  uncertainty?: number;
  low_confidence: boolean;
  uncertain: boolean;
  research_disclaimer: string;
}

export interface ExplainResult extends PredictionResult {
  heatmap_gradcam:    string;
  heatmap_gradcam_pp: string;
  heatmap_int_grads:  string;
  heatmap_guided_bp:  string;
  chart_comparison:   string;
  chart_tda:          string;
  chart_confidence:   string;
  tda_importance:     Record<string, number>;
  tda_values:         Record<string, number>;
  clinical_reasoning: {
    summary?: string;
    attention?: string;
    tda_findings?: string[];
    class_context?: string;
    confidence_note?: string;
  };
}

export async function predict(file: File): Promise<PredictionResult> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Prediction failed");
  }
  return res.json();
}

export async function explain(file: File): Promise<ExplainResult> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/explain`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Explanation failed");
  }
  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { 
      signal: AbortSignal.timeout(3000) // 3 second timeout for health check
    });
    return res.ok;
  } catch {
    return false;
  }
}
