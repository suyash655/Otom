/**
 * Centralized constants for the frontend.
 * 
 * This file contains all hardcoded values that should be consistent
 * across the application.
 */

// ── API Configuration ───────────────────────────────────────────────────────────
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const API_TIMEOUT = 30000; // 30 seconds
export const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

// ── Class Configuration (must match backend) ────────────────────────────────────
export const CLASS_ORDER = [
  "Normal",
  "Acute Otitis Media",
  "Cerumen Impaction",
  "Chronic Otitis Media",
  "Myringosclerosis",
  "Other",
] as const;

export const CLASS_NAMES = CLASS_ORDER;

// Tailwind hex mapping for components that use inline styles
export const CLASS_COLOR_HEX: Record<string, string> = {
  "Normal":               "#10B981", // emerald-500
  "Acute Otitis Media":   "#F43F5E", // rose-500
  "Cerumen Impaction":    "#F59E0B", // amber-500
  "Chronic Otitis Media": "#F97316", // orange-500
  "Myringosclerosis":     "#A855F7", // purple-500
  "Other":                "#94A3B8", // slate-400
};

// Tailwind utility classes for components that use classNames
export const CLASS_COLOR_TW: Record<string, string> = {
  "Normal":               "bg-emerald-500",
  "Acute Otitis Media":   "bg-rose-500",
  "Cerumen Impaction":    "bg-amber-500",
  "Chronic Otitis Media": "bg-orange-500",
  "Myringosclerosis":     "bg-purple-500",
  "Other":                "bg-slate-400",
};

// ── Safety Thresholds (must match backend) ─────────────────────────────────────
export const LOW_CONFIDENCE_THRESHOLD = 0.5;
export const UNCERTAINTY_THRESHOLD = 0.15;

// ── XAI Methods ───────────────────────────────────────────────────────────────
export const XAI_METHODS = [
  "gradcam",
  "gradcam_pp", 
  "integrated_gradients",
  "guided_backprop",
] as const;

export const XAI_METHOD_LABELS: Record<string, string> = {
  "gradcam": "Grad-CAM",
  "gradcam_pp": "Grad-CAM++",
  "integrated_gradients": "Integrated Gradients",
  "guided_backprop": "Guided Backpropagation",
};

// ── File Upload Configuration ─────────────────────────────────────────────────
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const ALLOWED_FILE_TYPES = ["image/jpeg", "image/png", "image/jpg"];
export const ALLOWED_FILE_EXTENSIONS = [".jpg", ".jpeg", ".png"];

// ── UI Configuration ───────────────────────────────────────────────────────────
export const TOAST_DURATION = 5000; // 5 seconds
export const DEBOUNCE_DELAY = 300; // 300ms

// ── Feature Display Configuration ─────────────────────────────────────────────
export const TDA_FEATURE_GROUPS = {
  H0: "Connected components",
  H1: "Loops & cavities",
};

export const TDA_FEATURE_COLORS = {
  H0: "#3B82F6", // blue-500
  H1: "#EC4899", // pink-500
};

// ── Research Disclaimer ───────────────────────────────────────────────────────
export const RESEARCH_DISCLAIMER = "Research use only. This prototype has not been clinically validated and must not inform diagnosis, treatment, or patient care.";

export const CLINICAL_DISCLAIMER = "Algorithmically generated from feature values. Not validated by clinicians.";