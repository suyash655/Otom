# Architecture Overview — OtoScope AI (Nexus Supply Chain)

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client / Browser                    │
│                     (Next.js Frontend)                   │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP REST
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                          │
│  backend/app/                                            │
│  ├── main.py          App factory + middleware           │
│  ├── core/            Config, Device, Dependencies       │
│  ├── routers/         /health  /predict  /explain        │
│  ├── schemas/         Pydantic request/response models   │
│  ├── services/        Business logic                     │
│  │   ├── data_ingestion.py   Dataset + transforms        │
│  │   ├── ml_service.py       HybridExplainer             │
│  │   ├── xai_service.py      Audit, inputs, plots        │
│  │   ├── copilot_service.py  Clinical reasoning text     │
│  │   └── graph_service.py    (reserved)                  │
│  ├── engines/         Heavy computation                  │
│  │   ├── analyzer.py         HybridModel (CNN+TDA)       │
│  │   ├── graph_engine.py     TDA feature extraction      │
│  │   └── forecaster.py       Training utilities          │
│  └── utils/           Shared helpers                     │
│      ├── plot_helpers.py     Visualization               │
│      └── metrics.py          Evaluation metrics          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              ML Components                               │
│  • EfficientNet-B0 backbone (ImageNet pretrained)        │
│  • TDA branch: persistent homology (Ripser)              │
│  • Late fusion classifier head                           │
│  • MC Dropout uncertainty estimation                     │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| CNN + TDA hybrid | CNN captures spatial texture; TDA captures topological shape invariants |
| MC Dropout uncertainty | Flags low-confidence predictions for clinical safety |
| Differential LR (1/100 backbone) | Prevents destroying ImageNet features on small dataset |
| Macro-F1 checkpoint selection | Avoids Normal-class bias in imbalanced dataset |
| Audit log (JSONL) | Privacy-preserving (SHA-256 hash only) traceability |

## Data Flow

```
Image Upload
    │
    ▼
PIL decode + resize (224×224)
    │
    ├──► CNN backbone → feature vector (1280-d)
    │
    └──► TDA pipeline:
           grayscale → cubical complex → Ripser → Betti numbers → 13-d vector
    │
    ▼
Late fusion → classifier head → softmax
    │
    ├──► MC Dropout × 10 → mean ± std uncertainty
    │
    └──► XAI: Grad-CAM / Grad-CAM++ / IntGrad / Guided-BP
```
