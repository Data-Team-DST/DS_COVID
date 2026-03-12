# 📋 CODE INVENTORY ANALYSIS - Migration Plan

**Generated:** 12/03/2026  
**Status:** Ready for migration to ml-backend DDD structure

---

## 🎯 Summary

Current code in `src/` needs to be organized into DDD layers in `ml-backend/src/ds_covid_backend/`

Total Python files found: **44 files** across 7 modules

---

## 📦 MODULE BREAKDOWN

### **MODULE 1: Core ML Models** ⭐ CRITICAL
**Location:** `src/ds_covid/`  
**Priority:** HIGH - needs migration immediately

| File | Type | Key Classes/Functions | DDD Layer | Notes |
|------|------|-------------------|----------|-------|
| `models.py` | ML Model | `build_baseline_cnn()`, Custom models | **domain/models/** | Core TensorFlow models - business logic |
| `features.py` | Features | Feature extraction | **domain/features/** | Feature engineering |
| `visualization.py` | Utils | Visualization functions | **infrastructure/visualization/** | External-facing utilities |
| `cli.py` | CLI | Old CLI entry point | **ARCHIVE** | Replaced by FastAPI |

**To Migrate:**
```
src/ds_covid/models.py → ml-backend/src/ds_covid_backend/domain/models/covid_model.py
src/ds_covid/features.py → ml-backend/src/ds_covid_backend/domain/features/
```

---

### **MODULE 2: Data Processing Pipeline** ⭐ CRITICAL
**Location:** `src/explorationdata/pipeline/`  
**Priority:** HIGH - needed for data loading

| File | Purpose | Classes | DDD Layer | Dependency |
|------|---------|---------|----------|-----------|
| `data_loader.py` | Load data | DataLoader | **infrastructure/data_loader.py** | pandas, numpy |
| `pipeline_runner.py` | Orchestration | PipelineRunner | **application/pipeline_service.py** | Coordinates data flow |
| `dimensionality_reducer.py` | Preprocessing | DimensionalityReducer | **infrastructure/preprocessing.py** | sklearn |
| `clustering_analyzer.py` | Analysis | ClusteringAnalyzer | **infrastructure/analysis.py** | sklearn |
| `visualizer.py` | Output visualization | Visualizer | **infrastructure/visualization.py** | matplotlib |

**To Migrate:**
```
src/explorationdata/pipeline/data_loader.py → ml-backend/src/ds_covid_backend/infrastructure/data_loader.py
src/explorationdata/pipeline/pipeline_runner.py → ml-backend/src/ds_covid_backend/application/pipeline_service.py
```

---

### **MODULE 3: Utilities** ⭐ CRITICAL
**Location:** `src/utils/`  
**Priority:** HIGH - used everywhere

| File | Purpose | Key Functions | DDD Layer | Status |
|------|---------|--------------|----------|--------|
| `data_utils.py` | Data helpers | normalize(), validate() | **infrastructure/** | MIGRATE |
| `model_builders.py` | Model factory | build_model(), load_model() | **application/services/** | MIGRATE |
| `training_utils.py` | Training logic | train(), evaluate() | **application/training_service.py** | MIGRATE |
| `visualization_utils.py` | For plots | plot_metrics() | **infrastructure/visualization.py** | MIGRATE |
| `config.py` | Configuration | Config class | **config/settings.py** | MIGRATE |
| `interpretability_utils.py` | Explainability | SHAP/LIME helpers | **infrastructure/explainability.py** | MIGRATE |

**To Migrate:**
```
src/utils/data_utils.py → ml-backend/src/ds_covid_backend/infrastructure/
src/utils/model_builders.py → ml-backend/src/ds_covid_backend/application/services/
src/utils/training_utils.py → ml-backend/src/ds_covid_backend/application/training_service.py
src/utils/config.py → ml-backend/src/ds_covid_backend/config/settings.py
```

---

### **MODULE 4: Image Processing** 📷
**Location:** `src/features/Pipelines/transformateurs/`  
**Priority:** MEDIUM - specific to image handling

| File | Purpose | Classes | DDD Layer |
|------|---------|---------|----------|
| `image_loaders.py` | Load images | ImageLoader | **infrastructure/image_loader.py** |
| `image_preprocessing.py` | Preprocess images | ImagePreprocessor | **infrastructure/image_processor.py** |
| `image_augmentation.py` | Data augmentation | Augmentor | **infrastructure/augmentation.py** |
| `image_features.py` | Image features | FeatureExtractor | **domain/features/** |
| `utilities.py` | Helpers | Various | **infrastructure/utils.py** |

---

### **MODULE 5: Interpretability** 🔍
**Location:** `src/interpretability/`  
**Priority:** LOW - for explainability

| File | Purpose | Classes | DDD Layer |
|------|---------|---------|----------|
| `gradcam.py` | Grad-CAM explainability | GradCAM | **infrastructure/explainability/gradcam.py** |
| `lime_explainer.py` | LIME explainability | LimeExplainer | **infrastructure/explainability/lime.py** |
| `shap_explainer.py` | SHAP explainability | ShapExplainer | **infrastructure/explainability/shap.py** |

---

### **MODULE 6: EDA Pipeline** 📊
**Location:** `src/explorationdata/`  
**Priority:** LOW - exploratory, not production

| File | Purpose | Status |
|------|---------|--------|
| `run_eda_pipeline.py` | Run EDA | KEEP in src/ (exploratory) |
| `generate_report.py` | Generate reports | KEEP in src/ (exploratory) |
| `test_pipeline.py` | Test runner | KEEP (testing utilities) |

---

### **MODULE 7: Tests** 🧪
**Location:** `src/test/`  
**Priority:** LOW - old tests

| File | Purpose | Status |
|------|---------|--------|
| `test_*.py` | Old unit tests | REWRITE in ml-backend/tests/ |

---

## 🗺️ MIGRATION ROADMAP

### **Phase 2A: Infrastructure Layer** (Day 3-4, ~4h)
```
infrastructure/
├── data_loader.py          ← from src/explorationdata/pipeline/data_loader.py
├── image_loader.py         ← from src/features/.../image_loaders.py
├── image_processor.py      ← from src/features/.../image_preprocessing.py
├── preprocessing.py        ← from src/utils/data_utils.py + dimensionality_reducer.py
├── visualization.py        ← from src/ds_covid/visualization.py
├── explainability/         ← from src/interpretability/
│   ├── gradcam.py
│   ├── lime.py
│   └── shap.py
└── augmentation.py         ← from src/features/.../image_augmentation.py
```

**Estimated Lines:** ~2000 lines to port  
**Dependencies:** pandas, numpy, scikit-learn, TensorFlow

---

### **Phase 2B: Domain Layer** (Day 4, ~2h)
```
domain/
├── models/
│   └── covid_model.py      ← from src/ds_covid/models.py
├── features/
│   └── feature_extractor.py ← from src/ds_covid/features.py + image_features.py
└── entities/
    ├── prediction.py       ← new: Prediction entity
    └── image.py            ← new: Image entity
```

**Estimated Lines:** ~1500 lines to port  
**Key Files:** models.py (50+ lines of TensorFlow)

---

### **Phase 2C: Application Layer** (Day 5, ~2h)
```
application/
├── services/
│   ├── prediction_service.py   ← orchestrates inference
│   ├── training_service.py     ← from src/utils/training_utils.py
│   └── model_service.py        ← from src/utils/model_builders.py
└── use_cases/
    ├── predict.py              ← PredictionUseCase
    └── train.py                ← TrainingUseCase
```

**Estimated Lines:** ~1000 lines to port  
**Creates:** Business logic layer

---

### **Phase 3: API Layer** (Day 6, ~2h)
```
api/
├── routes/
│   ├── predictions.py      ← POST /predict
│   ├── models.py           ← GET /models, POST /train
│   └── health.py           ← GET /health
└── schemas/
    ├── prediction.py       ← Pydantic schemas
    └── image.py
```

**To Create:** 3+ endpoints  
**Dependencies:** FastAPI, Pydantic

---

## 📊 MIGRATION STATISTICS

| Metric | Value |
|--------|-------|
| **Total Python files** | 44 |
| **Files to migrate** | 25-30 |
| **Files to archive** | 6-8 |
| **Files to keep (EDA)** | 5-6 |
| **Estimated LOC to port** | 3500-4000 |
| **New code to write** | 500-1000 |
| **Total effort** | ~15-20 hours (Jour 2-8) |

---

## 🔄 IMPORT DEPENDENCIES

**Critical Packages Currently Used:**
```
tensorflow      (Core ML)
scikit-learn    (Preprocessing, metrics)
numpy           (Numerical ops)
pandas          (Data loading)
opencv-python   (Image processing) - NOT IN CURRENT requirements.txt
matplotlib      (Visualization)
```

**Action:** May need to add OpenCV to requirements.txt

---

## ⚠️ BLOCKERS & NOTES

1. **TensorFlow Not in requirements.txt** - Need to add
2. **OpenCV Missing** - Check if needed for image loading
3. **Old imports** - Many files import from old structure, all need updating
4. **Test coverage low** - Old tests in src/test/ will be rewritten
5. **No FastAPI code yet** - Will create new in api/

---

## 📋 ACTION ITEMS FOR JOUR 2

- [ ] Confirm all dependencies (esp. TensorFlow, OpenCV)
- [ ] Create JOUR_2_INVENTORY_DETAILED.md with full file list
- [ ] Plan import changes (update all `from src.` to `from ds_covid_backend.`)
- [ ] Identify circular dependencies (if any)
- [ ] Create migration timeline (specific file order)

---

## ✅ NEXT STEPS

**Jour 3 (Tomorrow):**
1. Start migrating infrastructure/ layer
2. Port data_loader.py first (foundation)
3. Add tests for each module
4. Verify imports work

---

*Generated by Code Analysis Tool*  
*Last Updated: Phase 2 Planning*
