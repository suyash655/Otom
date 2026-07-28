# API Endpoints

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## GET `/health`

**Tags:** System

Health check — verifies the server is running and returns device + class info.

### Response `200 OK`
```json
{
  "status": "ok",
  "device": "cpu",
  "classes": ["Normal", "Acute Otitis Media", "..."]
}
```

---

## POST `/predict`

**Tags:** Inference  
**Content-Type:** `multipart/form-data`

Classify an uploaded otoscopic image.

### Request
| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | PNG or JPEG image |

### Response `200 OK` — `PredictionResponse`
```json
{
  "predicted_class": "Normal",
  "predicted_index": 0,
  "confidence": 0.9213,
  "class_probs": {
    "Normal": 0.9213,
    "Acute Otitis Media": 0.0312,
    "...": "..."
  },
  "uncertainty": 0.0421,
  "low_confidence": false,
  "uncertain": false,
  "research_disclaimer": "RESEARCH ONLY. ..."
}
```

### Safety Flags
| Flag | Condition |
|---|---|
| `low_confidence` | `max(softmax) < 0.5` |
| `uncertain` | `MC Dropout std > 0.15` |

---

## POST `/explain`

**Tags:** Explainability  
**Content-Type:** `multipart/form-data`

Classify + return full XAI explanations.

### Request
| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | PNG or JPEG image |

### Response `200 OK` — `ExplainResponse`

Includes all fields from `PredictionResponse` plus:

| Field | Type | Description |
|---|---|---|
| `heatmap_gradcam` | `string` | Base64 PNG — Grad-CAM overlay |
| `heatmap_gradcam_pp` | `string` | Base64 PNG — Grad-CAM++ overlay |
| `heatmap_int_grads` | `string` | Base64 PNG — Integrated Gradients |
| `heatmap_guided_bp` | `string` | Base64 PNG — Guided Backpropagation |
| `chart_comparison` | `string` | Base64 PNG — 4-panel comparison |
| `chart_tda` | `string` | Base64 PNG — TDA feature importance bar chart |
| `chart_confidence` | `string` | Base64 PNG — Class confidence bars |
| `tda_importance` | `dict[str, float]` | Per-feature importance scores |
| `tda_values` | `dict[str, float]` | Raw TDA feature values |
| `clinical_reasoning` | `dict` | Structured clinical narrative |

### Error Responses
| Code | Condition |
|---|---|
| `422` | Image loading / decoding failed |
| `500` | Model inference or explanation generation failed |
