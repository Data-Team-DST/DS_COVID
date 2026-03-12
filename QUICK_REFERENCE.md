# 📚 DS_COVID Quick Reference Guide

**One-page summary of codebase state and refactoring priorities**

---

## WHAT THE PROJECT DOES

**COVID-19 Detection from X-ray Images**
- Multi-class classifier (COVID, Normal, Lung_Opacity, Viral Pneumonia)
- Deep Learning (CNN) + Classical ML (Random Forest, XGBoost)
- Transfer Learning (EfficientNet, ResNet, VGG, InceptionV3)
- Model Interpretability (Grad-CAM, LIME, SHAP)
- Interactive web UI (Streamlit) + CLI tools

---

## CODEBASE SNAPSHOT

```
Total Size:     ~16,600 LOC in src/
Files:          77 Python modules, 21 Jupyter notebooks
Main Package:   src/ (ds_covid, utils, features, interpretability, explorationdata)
Web App:        9-page Streamlit interface in page/
Notebooks:      Complete EDA, transfer learning pipelines, baseline models
Documentation:  README, CONFIG.md, INSTALLATION.md, glossary
```

---

## WHAT EXISTS ✅

| Component | Status | Quality |
|-----------|--------|---------|
| **Architecture** | ✅ Exists | Good - modular, clear separation |
| **ML Models** | ✅ Code defined | Good - CNN, Transfer Learning, Classical ML |
| **Data Pipeline** | ✅ Code defined | Good - with masking support |
| **Interpretability** | ✅ Implemented | Good - Grad-CAM, LIME, SHAP |
| **EDA Pipeline** | ✅ Implemented | Good - comprehensive analysis |
| **Streamlit UI** | ✅ Built | Good - polished, well-organized |
| **CLI Tools** | ✅ Built | Good - image preprocessing, dataset creation |
| **Configuration** | ✅ 3 systems | ⚠️ **NEEDS CONSOLIDATION** |
| **Logging** | ❌ Missing | 🔴 **432 print() statements** |
| **Tests** | ❌ Critical gap | 🔴 **0% coverage** |
| **API Server** | ❌ Missing | 🟠 Only Streamlit, no REST API |
| **Docker/K8s** | ❌ Missing | 🟠 Not containerized |
| **CI/CD** | ❌ Missing | 🔴 No GitHub Actions |

---

## TRAINED MODELS & DATA

| Item | Status | Notes |
|------|--------|-------|
| **Model Weights** | ❌ None | Architecture code exists, must train from scratch |
| **Checkpoints** | ❌ None | No .h5, .pkl, or .pth files found |
| **Raw Data** | ⚠️ In gitignore | Expected at `data/raw/COVID-19_Radiography_Dataset/` |
| **Code for Data** | ✅ Ready | Dataset loading and preprocessing implemented |

---

## CRITICAL ISSUES (FIX FIRST) 🔥

### 1. Configuration Chaos (3 Systems)
**Files:** 
- `src/config.py` (System #1: .env-based)
- `src/ds_covid/config.py` (System #2: dataclasses)
- `src/features/raf/utils/config.py` (System #3: Colab-specific)

**Fix:** Keep 1, delete 2 others, migrate imports  
**Effort:** 3-5 days  
**Impact:** Stability, reduced bugs

### 2. No Structured Logging
**Problem:** 432 `print()` calls, no `logging` module  
**Where:** Every Python file has print() statements  
**Fix:** Create `src/utils/logging_config.py`, replace all prints  
**Effort:** 3-4 days  
**Impact:** Debuggability, production monitoring

### 3. Zero Tests
**Status:** 0% coverage, only import checks exist  
**Impact:** Can't refactor safely, high regression risk  
**Fix:** Create `tests/` directory with 30-40% coverage target  
**Effort:** 5-7 days  
**Impact:** Safety, confidence in changes

### 4. No Production API
**Current:** Streamlit only (fine for exploration, not for production)  
**Fix:** Add FastAPI server for inference endpoints  
**Effort:** 5-7 days  
**Impact:** Deployability, scalability

---

## MODULE DIRECTORY

### Core ML Modules

| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| `src/ds_covid/models.py` | 150 | CNN architecture, MaskApplicator | ✅ |
| `src/ds_covid/features.py` | 80 | Data loading, flattening | ✅ |
| `src/utils/model_builders.py` | ~100 | Transfer learning factory | ✅ |
| `src/utils/training_utils.py` | ~100 | Training loop, evaluation | ✅ |
| `src/utils/data_utils.py` | 150 | Dataset loading, generators | ✅ |

### Image Processing

| File | Purpose |
|------|---------|
| `image_processor.py` | Batch preprocessing with masking |
| `dataset_factory.py` | Create datasets with variants |
| `src/features/apply_masks.py` | Mask application logic |
| `src/features/Pipelines/transformateurs/` | Modular pipeline components |

### Interpretability (Excellent)

| File | Purpose |
|------|---------|
| `src/interpretability/gradcam.py` | CAM visualization |
| `src/interpretability/shap_explainer.py` | SHAP values |
| `src/interpretability/lime_explainer.py` | LIME explanations |

### EDA Pipeline

| File | Purpose |
|------|---------|
| `src/explorationdata/pipeline_runner.py` | Main orchestrator |
| `src/explorationdata/pipeline/data_loader.py` | Data loading |
| `src/explorationdata/pipeline/embedding_extractor.py` | Hidden representations |
| `src/explorationdata/pipeline/dimensionality_reducer.py` | PCA/UMAP/t-SNE |
| `src/explorationdata/pipeline/clustering_analyzer.py` | K-means, DBSCAN |

### Web UI

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main entry, page router |
| `page/01_accueil.py` | Welcome page |
| `page/02_donnees.py` | Data exploration |
| `page/03-08_*.py` | Various analysis pages |

---

## DEPENDENCIES AT A GLANCE

```
Deep Learning:     TensorFlow 2.18+, PyTorch 2.9+
Classical ML:      scikit-learn 1.5+
Data:             NumPy 2.0+, Pandas 2.2+, Pillow 11.0+
Visualization:     Matplotlib, Seaborn, Plotly
Interpretability: SHAP, LIME
Web:              Streamlit 1.30+
CLI:              Typer
Quality:          Pytest, Ruff, Mypy

Missing:          XGBoost, LightGBM, CatBoost (used but not listed)
```

---

## WHAT TO KEEP VS DELETE

### KEEP ✅ (Production-Ready)

```
✅ src/interpretability/          (Grad-CAM, SHAP, LIME)
✅ src/explorationdata/           (EDA pipeline, comprehensive)
✅ src/features/Pipelines/        (Well-designed pipeline pattern)
✅ src/utils/training_utils.py    (Training loops, well-documented)
✅ src/utils/data_utils.py        (Data loading utilities)
✅ src/utils/model_builders.py    (Transfer learning factory)
✅ Streamlit app structure        (Multi-page is elegant)
✅ image_processor.py             (Solid preprocessing)
✅ cli.py                         (Good CLI interface)
```

### REFACTOR ⚠️ (Needs Work)

```
⚠️ src/utils/config.py           (Consolidate 3 systems into 1)
⚠️ All src/                       (Replace 432 prints with logging)
⚠️ Large files (>500 LOC)         (Break into smaller modules)
⚠️ src/features/apply_masks.py    (Needs better documentation)
```

### DELETE 🗑️ (Obsolete)

```
🗑️ src/features/OLD/              (Legacy code)
🗑️ notebooks/modelebaseline/old/  (Old notebooks)
🗑️ Any *_backup.py, *_old.py     (Cleanup)
🗑️ Duplicate config files          (Keep only 1 unified system)
```

---

## 30-SECOND ACTION ITEMS

### Immediate (Next Week)
1. ✋ Create `tests/` directory structure
2. ✋ Consolidate config: Keep `src/utils/config.py`, delete other 2
3. ✋ Start replacing 432 prints with logging module

### Short Term (Weeks 2-3)
1. 🧪 Write 30+ unit tests (30% coverage target)
2. 🔧 Refactor files >500 LOC into modules
3. 📝 Add docstrings to public API

### Medium Term (Weeks 4-6)
1. 🚀 Create FastAPI server for inference
2. 🐳 Write Dockerfile + docker-compose.yml
3. ⚙️ Setup GitHub Actions CI/CD

---

## KEY FILES TO UNDERSTAND FIRST

**If you have 30 minutes:**
1. `README.md` - Project overview
2. `src/utils/config.py` - Configuration system
3. `src/ds_covid/models.py` - Model architecture
4. `streamlit_app.py` - Web UI entry

**If you have 2 hours:**
1. `COMPREHENSIVE_CODEBASE_ANALYSIS.md` - Full details
2. `REFACTORING_ACTION_PLAN.md` - Step-by-step fixes
3. `src/utils/data_utils.py` - Data pipeline
4. `src/explorationdata/pipeline/` - EDA system

**If you have 4+ hours:**
1. All modules in `src/`
2. Notebook files for context
3. `page/` directory for UI patterns

---

## COMMON TASKS - HOW TO...

### Load a Dataset
```python
from src.utils.data_utils import load_dataset
from pathlib import Path

image_paths, mask_paths, labels, labels_int = load_dataset(
    data_dir=Path('data/raw/COVID-19_Radiography_Dataset'),
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    load_masks=True
)
```

### Build & Train a Model
```python
from src.ds_covid.models import build_baseline_cnn
from src.utils.training_utils import train_model

model = build_baseline_cnn(input_shape=(256,256,1), num_classes=4)
history = train_model(model, train_gen, val_gen, epochs=50)
```

### Create Data Generators
```python
from src.utils.data_utils import create_data_generators

train_gen, val_gen, test_gen = create_data_generators(
    data_dir=Path('data/raw'),
    batch_size=32
)
```

### Explain a Prediction
```python
from src.interpretability.gradcam import GradCAM

gcam = GradCAM(model, layer_name='conv5_block3_out')
heatmap = gcam.compute_heatmap(image, class_idx=0)
gcam.visualize_gradcam(image, heatmap)
```

### Run EDA Pipeline
```python
from src.explorationdata.pipeline.pipeline_runner import EDAPipeline

pipeline = EDAPipeline(
    base_path='data/raw/COVID-19_Radiography_Dataset',
    metadata_path='metadata/',
    output_dir='outputs',
    max_images_per_class=100
)
pipeline.run()
```

### Start Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Imports fail | Run `pip install -e .` from project root |
| No data found | Ensure `data/raw/COVID-19_Radiography_Dataset/` exists |
| OOM errors | Reduce batch_size in config or max_images_per_class |
| Model not found | Models must be trained first, no pre-trained weights |
| Streamlit slow | Use @st.cache for expensive operations |
| Many print outputs | Will be replaced with logging (WIP) |

---

## METRICS TO TRACK

### Code Quality
- Test coverage: 0% → Target 40%
- Logging: 0% → Target 100% (replace prints)
- Config systems: 3 → Target 1
- Max file size: 3,886 → Target <500 LOC

### Performance
- Model accuracy: (to be measured)
- Inference time: (to be measured)
- Training time: (to be measured)

### Deployment
- API endpoints: 0 → Target 5+
- Container readiness: 0% → Target 100%
- CI/CD coverage: 0% → Target 80%

---

## USEFUL COMMANDS

```bash
# Development setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html

# Lint code
ruff check src/

# Type checking
mypy src/ --ignore-missing-imports

# Run Streamlit
streamlit run streamlit_app.py

# Run EDA pipeline
python -m src.explorationdata.run_eda_pipeline \
  --base-path data/raw/COVID-19_Radiography_Dataset \
  --metadata-path metadata \
  --output-dir outputs

# Format imports (future)
ruff check src/ --fix
```

---

## RESOURCES

- **Project Repo:** https://github.com/L-Poca/DS_COVID
- **COVID-19 Dataset:** Kaggle COVID-19 Radiography Dataset
- **TensorFlow Docs:** https://tensorflow.org
- **SHAP Documentation:** https://shap.readthedocs.io
- **Streamlit Docs:** https://docs.streamlit.io

---

**Last Updated:** March 11, 2026  
**Prepared for:** DS_COVID Team  
**Status:** Ready for Refactoring
