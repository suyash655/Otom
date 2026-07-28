# OtoScope AI: Hybrid CNN + Topological Data Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16+-black?logo=nextjs&logoColor=white)

⚠️ **RESEARCH PROTOTYPE ONLY** — Not clinically validated. Must not be used for diagnosis or patient care.

> Hybrid AI system combining deep convolutional features with **persistent homology** for robust otoscopy classification with topological explainability.

</div>

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 84.0% |
| **Macro-F1** | 84.1% |
| **Improvement over CNN** | +2.0pp |
| **TDA Features** | 13 (H0/H1 topological) |
| **Dataset Size** | 3,956 labeled images |
| **Inference Speed** | ~180ms (CPU) |

---

## 🎯 What This Project Does

OtoScope AI diagnoses **6 ear conditions** from otoscopic images using a hybrid approach:

1. **CNN Branch** (ResNet-18) → 512-d visual features
2. **TDA Branch** (Ripser) → 13-d topological features
3. **Late Fusion Classifier** → Combined prediction
4. **4 Explainability Methods** → Heatmaps + feature importance
5. **MC Dropout Uncertainty** → Confidence quantification

### Diagnosed Conditions
- ✓ Normal
- ✓ Acute Otitis Media (AOM)
- ✓ Chronic Otitis Media (CSOM)
- ✓ Cerumen Impaction (earwax)
- ✓ Myringosclerosis (calcification)
- ✓ Other/Atypical presentations

---

## 🚀 Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m ml.training.train --dummy        # Create checkpoint
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
📚 **API Docs:** http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
```
🎨 **UI:** http://localhost:3000

---

## 🏗️ Why Hybrid CNN + TDA?

### The Problem with CNN Alone
- Only learns visual textures
- Fails on poor quality images
- Hard to interpret
- Baseline accuracy: 82%

### The TDA Insight
Persistent homology captures **topological structure** invisible to CNNs:
- **H0:** Connected components, fragmentation
- **H1:** Loops, cavities, perforations

### Result: 84% Accuracy
Late fusion combines both feature types for +2.0pp improvement:

```
[CNN: 512-d]  ──┐
                ├─→ Concat (525-d) ──→ MLP ──→ [6 Classes]
[TDA: 13-d]  ──┘
```

---

## 📁 Project Structure

```
otoscope-ai/
├── README.md                   ← You are here (quick start)
├── README_PROFESSIONAL.md      ← Full technical documentation ⭐
├── backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI entry point
│   │   ├── core/              ← Config, device, logging
│   │   ├── routers/           ← /health, /predict, /explain
│   │   ├── schemas/           ← Pydantic models
│   │   ├── services/          ← Business logic
│   │   │   ├── ml_service.py  ← Model inference
│   │   │   ├── xai_service.py ← Explainability
│   │   │   └── copilot_service.py ← Clinical reasoning
│   │   ├── engines/           ← Heavy computation
│   │   │   ├── analyzer.py    ← HybridModel
│   │   │   └── graph_engine.py ← TDA extraction
│   │   └── utils/             ← Helpers
│   ├── requirements.txt
│   └── tests/
├── frontend/                  ← Next.js + React UI
│   ├── app/
│   │   ├── page.tsx           ← Landing page
│   │   ├── demo/              ← Interactive demo
│   │   ├── explainability/    ← XAI dashboard
│   │   ├── evaluation/        ← Results page
│   │   ├── hooks/             ← Custom React hooks
│   │   └── components/        ← UI components
│   ├── lib/
│   │   ├── api.ts             ← Backend client
│   │   └── constants.ts       ← Config
│   └── package.json
├── ml/                        ← Canonical ML module
│   ├── models/
│   │   └── hybrid_model.py    ← CNN + TDA architecture
│   ├── preprocessing/
│   │   └── dataset.py         ← PyTorch Dataset
│   ├── training/
│   │   └── train.py           ← Training script
│   ├── tda/
│   │   └── tda_extract.py     ← Persistent homology
│   ├── xai/
│   │   ├── gradcam.py         ← Attribution methods
│   │   └── clinical_reasoning.py
│   ├── inference/
│   │   └── predictor.py       ← Wrapper
│   └── evaluation/
│       └── metrics.py         ← F1, confusion, etc.
├── data/
│   ├── raw/
│   │   └── Otoscopic_Data/    ← Images (train/test)
│   ├── features/              ← TDA features cache
│   └── splits.json            ← Data indices
├── checkpoints/               ← Saved models
├── logs/                      ← Audit log
└── docs/                      ← Technical guides
    ├── architecture.md        ← System design
    ├── api-endpoints.md       ← REST reference
    ├── data-model.md          ← Schema definitions
    └── evaluation.md          ← Metrics & baselines
```

---

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict (Fast)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@scan.jpg"
```

### Explain (Full XAI Suite)
```bash
curl -X POST http://localhost:8000/explain \
  -F "file=@scan.jpg"
```

📚 **Full Docs:** http://localhost:8000/docs

---

## 🔬 Technical Highlights

### 1. Persistent Homology Features
13 topological features extracted via **Ripser**:
- Betti numbers (H0, H1) at 3 thresholds
- Persistence entropy (H0, H1)
- Bottleneck & Wasserstein distances (H0, H1)
- Persistence landscape (H0)

### 2. MC Dropout Uncertainty
30 stochastic forward passes → confidence bounds

### 3. Multi-Method Explainability
- **Grad-CAM++** — Class activation maps
- **Integrated Gradients** — Feature attribution
- **Guided Backpropagation** — Local structure
- **Feature Occlusion** — Ablation importance

### 4. Imbalanced Classification
- Weighted loss (inverse class frequency)
- Macro-F1 optimization
- Stratified train/test split

---

## 📊 Performance

| Model | Accuracy | Macro-F1 | Δ |
|---|---|---|---|
| CNN only | 82.0% | 81.5% | — |
| **Hybrid** | **84.0%** | **84.1%** | **+2.0pp** |

---

## 🧪 Training from Scratch

```bash
python -m ml.preprocessing.generate_splits        # First time only
python -m ml.training.train --train --epochs 50 --device cuda
python -m ml.training.train --eval --checkpoint checkpoints/hybrid_best.pth
```

---

## ⚖️ Important Disclaimers

- ⚠️ **NOT** FDA-approved
- ⚠️ **NOT** clinically validated  
- ⚠️ **NOT** for diagnosis or patient care
- ✅ Research prototype with uncertainty quantification
- ✅ Audit logging for traceability

---

## 📚 Full Documentation

For comprehensive technical details, system architecture, methodology, training procedures, and more:

### 👉 **[See: README_PROFESSIONAL.md](./README_PROFESSIONAL.md)**

This professional README includes:
- ✅ Complete architecture diagrams & data flow
- ✅ Why hybrid CNN + TDA works (deep dive)
- ✅ All 4 explainability methods explained
- ✅ Persistent homology theory & features
- ✅ MC Dropout uncertainty estimation
- ✅ Dataset composition & preprocessing
- ✅ Full training & evaluation procedures
- ✅ Ablation study results
- ✅ Per-class performance metrics
- ✅ Docker deployment instructions
- ✅ Future research directions
- ✅ Complete API reference
- ✅ References to key papers

---

## 🤝 Contributing

Found an issue? Have a suggestion? Open a GitHub issue or PR!

---

## 📝 Citation

```bibtex
@software{otoscope_ai_2026,
  title = {OtoScope AI: Hybrid CNN + TDA for Otoscopy Classification},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/otoscope-ai},
  note = {Research prototype}
}
```

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com)
- **Email**: your-email@institution.edu

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

---

<div align="center">

**Built with ❤️ for explainable medical AI research**

*Combining deep learning with topological data analysis to advance medical imaging classification.*

**[Quick Start](#-quick-start) • [Full Docs](./README_PROFESSIONAL.md) • [API Docs](http://localhost:8000/docs)**

</div>

Full API docs at `http://localhost:8000/docs`