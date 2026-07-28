# OtoScope AI Refactor Report

## Phase 1: Repository Audit

**Date:** 2026-07-09  
**Auditor:** Devin AI  
**Scope:** Full codebase audit for production-quality refactor

---

## Executive Summary

The audit revealed **critical structural issues** with nearly complete code duplication between `src/` and `ml/` directories, architectural violations where ML code exists in the backend, and numerous hardcoded values that should be configuration-driven. The most urgent issue is the duplicate codebase which creates maintenance burden and inconsistency risks.

---

## Critical Issues

### 1. DUPLICATE FILES (HIGH SEVERITY)

**Issue:** Complete Code Duplication Between src/ and ml/ Directories

**Files Affected:**
- `src/data/dataset.py` ↔ `ml/preprocessing/dataset.py` (353 lines, 100% identical)
- `src/data/generate_splits.py` ↔ `ml/preprocessing/generate_splits.py` (90 lines, 100% identical)
- `src/features/tda_extract.py` ↔ `ml/tda/tda_extract.py` (335 lines, 100% identical)
- `src/models/cnn_baseline.py` ↔ `ml/models/cnn_baseline.py` (59 lines, 100% identical)
- `src/models/hybrid_model.py` ↔ `ml/models/hybrid_model.py` (184 lines, 100% identical)
- `src/train.py` ↔ `ml/training/train.py` (459 lines, 100% identical)
- `src/xai/explainer.py` ↔ `ml/xai/explainer.py` (461 lines, 100% identical)
- `src/xai/clinical_reasoning.py` ↔ `ml/xai/clinical_reasoning.py` (297 lines, 100% identical)
- `src/xai/plots.py` ↔ `ml/xai/plots.py` (534 lines, 100% identical)
- `src/xai/run_explanation.py` ↔ `ml/xai/run_explanation.py` (245 lines, 100% identical)

**Description:** The entire ML codebase exists in two locations with identical content. This represents approximately 3,000+ lines of duplicated code.

**Recommendation:** 
1. **Immediate:** Decide on canonical location (recommend `ml/` as it's better organized)
2. Delete `src/` directory entirely after confirming all imports are updated
3. Update all import statements from `src.*` to `ml.*`
4. Update documentation and training scripts to reference `ml/` path

---

### 2. ARCHITECTURAL VIOLATIONS (HIGH SEVERITY)

**Issue 2.1:** ML/XAI Code in Backend Services Layer

**File:** `backend/app/services/ml_service.py` (lines 1-461)

**Description:** The file `ml_service.py` contains complete XAI explainer implementations (GradCAM, GradCAMPlusPlus, IntegratedGradients, GuidedBackpropagation, TDAFeatureImportance) which are ML model artifacts, not backend service logic. This code should live in the ML module, not the FastAPI backend.

**Recommendation:**
1. Move `backend/app/services/ml_service.py` to `ml/xai/explainer.py` (if not already there)
2. Backend should import XAI classes from ML module, not define them
3. Keep only service orchestration logic in backend (loading models, preparing inputs)

---

**Issue 2.2:** Duplicate Model Definitions

**Files:** 
- `backend/app/services/data_ingestion.py` (lines 45-51)
- `ml/preprocessing/dataset.py` (lines 45-52)

**Description:** CLASS_NAMES are defined in both locations with **different class orders**:
- Backend: 5 classes (Missing "Myringosclerosis")
- ML/dataset: 6 classes (Includes "Myringosclerosis")

**Recommendation:**
1. **Critical:** Single source of truth for CLASS_NAMES in `ml/preprocessing/dataset.py`
2. Backend should import CLASS_NAMES from ML module
3. Update model checkpoints to match the canonical class definition

---

**Issue 2.3:** Inconsistent Class Weights

**Files:**
- `backend/app/services/data_ingestion.py` (lines 88-94)
- `ml/preprocessing/dataset.py` (lines 90-97)

**Description:** CLASS_WEIGHTS are different between backend and ML module, which will cause incorrect model behavior.

**Recommendation:** Use single source of truth in `ml/preprocessing/dataset.py`

---

### 3. MISSING MODEL FILE (HIGH SEVERITY)

**Issue:** Missing analyzer.py in backend/app/engines/

**File:** Referenced in `backend/app/core/dependencies.py` line 66: `from backend.app.engines.analyzer import HybridModel`

**Description:** The file `backend/app/engines/analyzer.py` does not exist but is imported in dependencies.py. This will cause runtime errors.

**Recommendation:** 
1. Create `backend/app/engines/analyzer.py` with HybridModel import from `ml.models.hybrid_model`
2. Or update imports to directly use `ml.models.hybrid_model`

---

## High Priority Issues

### 4. HARDCODED VALUES (MEDIUM SEVERITY)

**Issue 4.1:** Hardcoded Data Paths

**Files Affected:**
- `ml/preprocessing/dataset.py` (line 168): `data_dir: str = "data/raw/Otoscopic_Data"`
- `ml/preprocessing/generate_splits.py` (line 9): `data_dir = Path("data/raw/Otoscopic_Data")`
- `ml/tda/tda_extract.py` (line 246): `data_dir: str = "data/raw/Otoscopic_Data"`
- `ml/training/train.py` (line 212): `data_dir: str = "data/raw/Otoscopic_Data"`
- `scripts/setup/train.py` (line 212): `data_dir: str = "data/raw/Otoscopic_Data"`
- `scripts/maintenance/smoke_test.py` (line 19): `ds = OtoscopyDataset('data/raw/Otoscopic_Data', ...)`

**Recommendation:** Use environment variable or config:
```python
DATA_DIR = os.getenv("OTOSCOPIC_DATA_DIR", "data/raw/Otoscopic_Data")
```

---

**Issue 4.2:** Hardcoded API URL

**File:** Frontend configuration

**Description:** API URL is hardcoded with fallback to localhost

**Recommendation:** Ensure `NEXT_PUBLIC_API_URL` is properly set in production environment variables

---

**Issue 4.3:** Hardcoded Checkpoint Paths

**Files:**
- `backend/app/core/config.py` (lines 34-36)
- `ml/training/train.py` (line 322)

**Recommendation:** Use environment variable `CHECKPOINTS_DIR` or config file

---

**Issue 4.4:** Hardcoded Feature Dimensions

**Files:**
- `ml/xai/run_explanation.py` (line 65): `tda_feature_dim=14`
- `ml/training/train.py` (line 164): `tda_feats = torch.zeros(images.shape[0], 13, device=device)`

**Description:** Inconsistent TDA feature dimensions (13 vs 14) throughout codebase

**Recommendation:** Define constant `TDA_FEATURE_DIM = 13` in config and use everywhere

---

### 5. INCONSISTENT NAMING (MEDIUM SEVERITY)

**Issue 5.1:** Directory Structure Inconsistency

**Description:** Similar functionality organized differently:
- `src/data/` vs `ml/preprocessing/` (both handle datasets)
- `src/features/` vs `ml/tda/` (both handle TDA features)
- `src/xai/` vs `ml/xai/` (both handle explainability)

**Recommendation:** After removing `src/`, standardize on:
- `ml/preprocessing/` for data loading
- `ml/tda/` for topological features
- `ml/xai/` for explainability
- `ml/training/` for training scripts
- `ml/evaluation/` for metrics

---

**Issue 5.2:** Inconsistent Import Patterns

**Files:** Multiple files in backend use `importlib.util` for dynamic imports

**Description:** Backend uses complex dynamic imports instead of standard Python imports

**Recommendation:** Use standard Python imports after resolving duplicate directories

---

## Medium Priority Issues

### 6. DEAD CODE (LOW-MEDIUM SEVERITY)

**Issue 6.1:** Debug Scripts in Production Code

**Files:**
- `scripts/debug/check_tda.py`
- `scripts/debug/check_metrics.py`
- `scripts/debug/fix_tda.py`
- `src/features/check_tda.py`

**Description:** Debug/fix scripts should not be in production repository structure

**Recommendation:** Move to separate `tools/` directory outside main source, or remove if no longer needed

---

**Issue 6.2:** Unused Imports

**Files:** Multiple files have unused imports

**Recommendation:** Run `autoflake` or `pylint` to remove unused imports automatically

---

**Issue 6.3:** Commented Code

**File:** `ml/preprocessing/dataset.py` (line 291-292)

**Recommendation:** Remove commented code or convert to proper documentation

---

### 7. MISSING CONFIGURATION (MEDIUM SEVERITY)

**Issue 7.1:** No Central Configuration File

**Description:** Configuration scattered across:
- `backend/app/core/config.py`
- `.env.example` (minimal)
- Hardcoded values in individual files

**Recommendation:** Create comprehensive `config/settings.py` with Pydantic settings

---

**Issue 7.2:** Missing Environment Variables

**File:** `.env.example` (needs review)

**Recommendation:** Add comprehensive environment variables to `.env.example`

---

## Low Priority Issues

### 8. TEMPORARY/DEBUG FILES (LOW SEVERITY)

**Issue 8.1:** Debug Directory

**Directory:** `scripts/debug/`

**Files:** `check_tda.py`, `check_metrics.py`, `fix_tda.py`

**Recommendation:** 
1. If still needed: Move to project root `tools/` directory
2. If obsolete: Delete entirely

---

**Issue 8.2:** Maintenance Scripts

**Directory:** `scripts/maintenance/`

**Files:** `check_ckpt.py`, `smoke_test.py`

**Recommendation:** Keep these as they serve operational purposes, but document their usage in README

---

### 9. FRONTEND ISSUES (LOW-MEDIUM SEVERITY)

**Issue 9.1:** UI Components with Direct API Calls

**File:** Frontend demo component

**Description:** Component directly imports and uses API functions

**Recommendation:** This is acceptable for client-side components, but consider adding a service layer for better testability

---

**Issue 9.2:** Duplicate Class Configuration

**Files:**
- `frontend/lib/config.ts` (CLASS_ORDER, CLASS_COLOR_HEX)
- Backend CLASS_NAMES

**Description:** Class definitions exist in both frontend and backend

**Recommendation:** Frontend should fetch class names from backend `/health` endpoint rather than hardcoding

---

## Summary Statistics

- **Total duplicate files:** 10 files (~3,000+ lines)
- **Architectural violations:** 3 major issues
- **Hardcoded values:** 15+ instances across codebase
- **Naming inconsistencies:** 5+ instances
- **Dead code/commented code:** 3+ instances
- **Missing configuration:** Critical need for central config
- **Temporary files:** 5 debug scripts

**Estimated refactoring effort:** 2-3 weeks for full cleanup, with critical issues resolvable in 3-5 days.

---

## Prioritized Action Plan

### Immediate (Critical - Phase 2)
1. **Resolve duplicate directories:** Delete `src/` directory after confirming `ml/` is canonical
2. **Fix CLASS_NAMES inconsistency:** Ensure single source of truth in `ml/preprocessing/dataset.py`
3. **Fix CLASS_WEIGHTS inconsistency:** Use single source of truth
4. **Move ML code from backend:** Remove XAI implementations from `backend/app/services/ml_service.py`
5. **Fix missing analyzer.py:** Create proper model import structure

### High Priority (Phase 3-4)
6. **Centralize configuration:** Create `config/settings.py` with Pydantic
7. **Remove hardcoded paths:** Use environment variables for data/checkpoint directories
8. **Fix TDA feature dimension inconsistency:** Use constant `TDA_FEATURE_DIM = 13`
9. **Clean up debug scripts:** Move or remove `scripts/debug/`

### Medium Priority (Phase 5-6)
10. **Standardize directory structure:** Document and enforce ML module organization
11. **Remove unused imports:** Run automated cleanup
12. **Frontend configuration:** Fetch class names from backend
13. **Add environment variable documentation:** Update `.env.example`

### Low Priority (Phase 7-8)
14. **Standardize import patterns:** Remove dynamic imports where possible
15. **Remove commented code:** Clean up code comments
16. **Add type hints:** Improve type coverage in ML module
