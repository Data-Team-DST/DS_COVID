# 🎯 DS_COVID Refactoring Action Plan

**Prepared:** March 11, 2026  
**Estimated Duration:** 4-6 weeks (1-2 developers)  
**Priority Level:** HIGH - Critical issues blocking maintainability

---

## EXECUTIVE SUMMARY

Your codebase has a **solid foundation** but suffers from **4 critical blockers** that prevent safe maintenance and deployment:

1. **♻️ Multiple Config Systems** - 3 competing systems causing inconsistency
2. **📝 No Structured Logging** - 432 print() statements vs proper logging
3. **❌ Zero Tests** - 0% coverage, blocking safe refactoring
4. **📦 No Production Deployment** - Missing API, Docker, CI/CD

**Good News:** These are all **fixable** with focused effort. The underlying architecture is sound.

---

## PHASE 1: CRITICAL FIXES (Weeks 1-2) 🔥

### Task 1.1: Consolidate Configuration Systems
**Effort:** 3-5 days | **Impact:** Stability  
**Owner:** Senior Developer

#### Current State
```
src/config.py                          (System #1: .env-based)
src/ds_covid/config.py                 (System #2: dataclasses)
src/features/raf/utils/config.py       (System #3: Colab-specific)
```

#### Action

**Step 1.1.1: Keep One System**
```
DECISION: Keep src/utils/config.py as the SINGLE source of truth
REASON: Most complete, best structure, already used by notebooks
```

**Step 1.1.2: Enhance Kept System**
Add to `src/utils/config.py`:
```python
# NEW: Environment detection
def detect_environment() -> str:
    """Auto-detect: local, colab, wsl"""
    if 'COLAB_RELEASE_TAG' in os.environ:
        return 'colab'
    # ... etc
    return 'local'

# NEW: Config factory
def load_config(env: str = None) -> Config:
    """Load config based on environment"""
    env = env or detect_environment()
    if env == 'colab':
        return load_colab_config()
    elif env == 'wsl':
        return load_wsl_config()
    else:
        return load_local_config()
```

**Step 1.1.3: Migrate Imports**
```bash
# Find all imports of old configs
grep -r "from src.ds_covid.config import" --include="*.py"
grep -r "from src.features.raf.utils.config import" --include="*.py"

# Replace with:
# from src.utils.config import Config, load_config
```

**Step 1.1.4: Delete Duplicate Systems**
```bash
rm src/ds_covid/config.py
rm src/features/raf/utils/config.py
```

**Step 1.1.5: Create Environment-Specific Fixture Files**
```
config/
├── default.json         (local development)
├── colab.json          (Google Colab)
├── wsl.json            (Windows Subsystem for Linux)
└── production.json     (future)
```

**Testing:**
```python
# Add to test_config.py
def test_config_consolidation():
    # Verify no file imports old systems
    from src.utils.config import Config, load_config
    
    config = load_config('colab')
    assert config is not None
    assert hasattr(config, 'img_width')
```

---

### Task 1.2: Implement Structured Logging
**Effort:** 3-4 days | **Impact:** Debuggability  
**Owner:** Mid-level Developer

#### Current State
```
432 print() statements → 0 logging module usage
```

#### Action

**Step 1.2.1: Create Logging Configuration**
New file: `src/utils/logging_config.py`
```python
import logging
import logging.config
import json
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Setup structured logging with optional file output
    
    Args:
        level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_file: Optional file path for logging
    """
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'json': {
                '()': 'pythonjsonlogger.jsonlogger.JsonFormatter'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': level,
            'formatter': 'standard',
            'filename': log_file
        }
        config['root']['handlers'].append('file')
    
    logging.config.dictConfig(config)

# Get logger
logger = logging.getLogger(__name__)
```

**Step 1.2.2: Replace print() Statements**
Create a script to identify and replace:
```bash
# Find print statements
grep -rn "print(" src/ --include="*.py" | head -20

# Pattern to replace:
# print("Loading data...")
# →
# logger.info("Loading data...")

# print("ERROR: Invalid path")
# →
# logger.error("Invalid path")

# print("DEBUG: Processing...")
# →
# logger.debug("Processing...")
```

**Step 1.2.3: Update All Modules**
Focus areas (highest print density):
1. `image_processor.py` - Replace console output
2. `dataset_factory.py` - Replace progress prints
3. Streamlit pages - Keep st.write() but add logger calls
4. `explorationdata/pipeline_runner.py` - Already has some logging, enhance it

**Step 1.2.4: Add Logging to Key Functions**
Example refactor for `image_processor.py`:
```python
# BEFORE
def process(self, dry_run: bool = False) -> dict:
    print(f"🚀 Début du prétraitement")
    print(f"Source: {self.source_path}")
    ...

# AFTER
import logging
logger = logging.getLogger(__name__)

def process(self, dry_run: bool = False) -> dict:
    logger.info("Starting preprocessing")
    logger.info(f"Source: {self.source_path}")
    logger.debug(f"Target size: {self.target_size}")
    ...
```

**Testing:**
```python
def test_logging_integration(caplog):
    """Verify logging is captured properly"""
    logger.info("Test message")
    assert "Test message" in caplog.text
```

---

### Task 1.3: Create Basic Test Suite
**Effort:** 5-7 days | **Impact:** Safety  
**Owner:** Mid-level Developer (or two juniors)  
**Goal:** Reach 30% coverage

#### Strategy

**Step 1.3.1: Create Test Structure**
```
tests/  (create new at workspace root)
├── unit/
│   ├── test_config.py
│   ├── test_data_utils.py
│   ├── test_training_utils.py
│   └── test_models.py
├── integration/
│   ├── test_preprocessing_pipeline.py
│   └── test_eda_pipeline.py
├── conftest.py  (pytest fixtures)
└── __init__.py
```

**Step 1.3.2: Write Fixtures** (`tests/conftest.py`)
```python
import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

@pytest.fixture
def temp_data_dir():
    """Create temporary dataset directory with sample images"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample structure
        for class_name in ["COVID", "Normal"]:
            class_dir = tmpdir / class_name / "images"
            class_dir.mkdir(parents=True)
            
            # Create 2 sample images per class
            for i in range(2):
                img = Image.fromarray(
                    np.random.randint(0, 255, (256, 256), dtype=np.uint8),
                    mode='L'
                )
                img.save(class_dir / f"image_{i}.png")
        
        yield tmpdir

@pytest.fixture
def sample_config():
    """Create sample configuration"""
    from src.utils.config import Config
    from pathlib import Path
    
    return Config(
        project_root=Path("/tmp/test"),
        data_dir=Path("/tmp/test/data"),
        models_dir=Path("/tmp/test/models"),
        results_dir=Path("/tmp/test/results"),
        img_width=256,
        img_height=256,
        batch_size=32,
        epochs=10
    )
```

**Step 1.3.3: Write Unit Tests**

**File: `tests/unit/test_config.py`**
```python
from src.utils.config import Config, load_config
from pathlib import Path

def test_config_initialization(sample_config):
    """Test Config dataclass initialization"""
    assert sample_config.img_width == 256
    assert sample_config.batch_size == 32

def test_config_from_json(tmp_path):
    """Test loading config from JSON file"""
    config_file = tmp_path / "test_config.json"
    config_file.write_text("""{
        "img_width": 512,
        "img_height": 512,
        "batch_size": 64
    }""")
    
    # Implement load_config method
    config = load_config(config_file)
    assert config.img_width == 512
```

**File: `tests/unit/test_data_utils.py`**
```python
from src.utils.data_utils import load_dataset, create_data_generators
from pathlib import Path

def test_load_dataset(temp_data_dir):
    """Test dataset loading"""
    image_paths, mask_paths, labels, labels_int = load_dataset(
        data_dir=temp_data_dir,
        categories=["COVID", "Normal"]
    )
    
    assert len(image_paths) == 4  # 2 classes × 2 images
    assert len(labels) == 4
    assert set(labels) == {"COVID", "Normal"}

def test_create_data_generators(temp_data_dir, sample_config):
    """Test data generator creation"""
    train_gen, val_gen, test_gen = create_data_generators(
        data_dir=temp_data_dir,
        batch_size=32
    )
    
    assert train_gen is not None
    assert val_gen is not None
```

**File: `tests/unit/test_models.py`**
```python
from src.ds_covid.models import build_baseline_cnn

def test_baseline_cnn_architecture():
    """Test CNN model builds correctly"""
    model = build_baseline_cnn(
        input_shape=(256, 256, 1),
        num_classes=4,
        dropout_rate=0.5
    )
    
    assert model is not None
    assert len(model.layers) > 0
    
    # Check output shape
    assert model.output_shape == (None, 4)

def test_baseline_cnn_compilation():
    """Test CNN is properly compiled"""
    model = build_baseline_cnn()
    
    assert model.optimizer is not None
    assert model.loss is not None
    assert 'accuracy' in [m.name for m in model.metrics]
```

**Step 1.3.4: Write Integration Tests**

**File: `tests/integration/test_preprocessing_pipeline.py`**
```python
from src.features.Pipelines.transformateurs.image_loaders import ImageLoader
from src.features.Pipelines.transformateurs.image_preprocessing import ImageResizer
from pathlib import Path
import numpy as np

def test_full_preprocessing_pipeline(temp_data_dir):
    """Test complete preprocessing pipeline"""
    # Load
    loader = ImageLoader(temp_data_dir / "COVID" / "images")
    images = loader.load()
    
    assert len(images) == 2
    
    # Resize
    resizer = ImageResizer(target_size=(256, 256))
    resized = [resizer.apply(img) for img in images]
    
    assert all(img.size == (256, 256) for img in resized)
```

**Step 1.3.5: Run Tests and Measure Coverage**
```bash
# Install test requirements
pip install pytest pytest-cov pytest-mock

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Target output:
# Name           Stmts   Miss  Cover
# ────────────────────────────────────
# src/config.py    80     10   87%
# src/utils/      150     30   80%
# src/ds_covid/    200     60   70%
# ────────────────────────────────────
# TOTAL            700    200   71%  ← (target: 30%+)
```

**Step 1.3.6: Add pytest Configuration**
File: `pyproject.toml` (update)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--strict-markers --tb=short"
markers = [
    "unit: unit tests",
    "integration: integration tests",
    "slow: slow running tests"
]
```

---

## PHASE 2: STRUCTURAL IMPROVEMENTS (Weeks 3-4) 🏗️

### Task 2.1: Refactor Large Files
**Effort:** 5-7 days | **Impact:** Maintainability

If file >500 lines found (like `images_tab.py`), break into modules:
```
images_analysis/
├── __init__.py
├── loader.py          (image loading logic)
├── preprocessor.py    (image preprocessing)
├── visualizer.py      (visualization functions)
├── analyzer.py        (analysis functions)
└── utils.py           (helper functions)
```

Each file <200 lines, single responsibility.

### Task 2.2: Clean Legacy Code
**Effort:** 2-3 days | **Impact:** Clarity

```bash
# Backup first
cp -r notebooks/modelebaseline/old notebooks/modelebaseline/old.backup

# Delete confirmed obsolete:
rm -rf src/features/OLD/
rm -rf notebooks/modelebaseline/old/
```

---

## PHASE 3: PRODUCTION READINESS (Weeks 5-6) 🚀

### Task 3.1: Create REST API Server
**Effort:** 5-7 days | **Impact:** Deployability

New structure:
```
api/
├── __init__.py
├── main.py            (FastAPI app)
├── models.py          (Pydantic models)
├── routes/
│   ├── __init__.py
│   ├── predict.py     (inference endpoint)
│   └── health.py      (health check)
└── config.py          (API config)
```

Example `api/main.py`:
```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Load model at startup
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    global model
    model = load_trained_model('models/best_model.h5')
    yield
    # Cleanup
    model = None

app = FastAPI(
    title="COVID-19 Radiography API",
    version="0.1.0",
    lifespan=lifespan
)

@app.post("/predict")
async def predict(file: UploadFile):
    """Predict COVID status from radiography image"""
    img = await file.read()
    prediction = model.predict(img)
    return {"class": prediction, "confidence": float(confidence)}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Task 3.2: Containerization
**Effort:** 2-3 days

**File: `Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY api/ ./api
COPY config/ ./config

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File: `docker-compose.yml`**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - LOG_LEVEL=INFO
  
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
```

### Task 3.3: CI/CD Pipeline
**Effort:** 2-3 days

**File: `.github/workflows/tests.yml`**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      
      - name: Lint
        run: ruff check src/
      
      - name: Type check
        run: mypy src/ --ignore-missing-imports
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## PARALLEL WORK STREAMS

Could assign multiple people:

| Stream | Person | Duration | Tasks |
|--------|--------|----------|-------|
| **Logging** | Dev A | 3-4d | Replace prints, setup logging config |
| **Config** | Dev B | 3-5d | Consolidate configs, migrate imports |
| **Tests** | Dev A+B | 5-7d | Write test suite, measure coverage |
| **Refactor** | Dev C | 5-7d | Break large files, clean legacy code |
| **API** | Dev D | 5-7d | FastAPI setup, endpoints, deployment |

---

## MILESTONES & DELIVERABLES

### Milestone 1 (End of Week 1)
- ✅ Single config system in place
- ✅ Logging module integrated
- ✅ 15+ unit tests written

### Milestone 2 (End of Week 2)
- ✅ 30% test coverage achieved
- ✅ All critical files refactored
- ✅ Legacy code removed

### Milestone 3 (End of Week 4)
- ✅ FastAPI server running
- ✅ Docker image builds successfully
- ✅ Endpoints documented

### Milestone 4 (End of Week 6)
- ✅ CI/CD pipeline live
- ✅ 50%+ test coverage
- ✅ Production-ready documentation

---

## SUCCESS CRITERIA

### Code Quality
- ❌→✅ 0% → 40% test coverage
- ❌→✅ 432 prints → 0 prints (all logging)
- ❌→✅ 3 configs → 1 unified config
- ❌→✅ Max file 3,886 LOC → Max <500 LOC

### Deployment
- ❌→✅ No API → FastAPI server running
- ❌→✅ No Docker → Containerized & tested
- ❌→✅ No CI/CD → GitHub Actions automated

### Documentation
- ✅ API documentation (Swagger)
- ✅ Architecture decision records (ADR)
- ✅ Deployment guide

---

## RISK MITIGATION

**Risk:** Breaking existing functionality during refactoring

**Mitigation:** 
1. Create feature branch for each task
2. Comprehensive tests written FIRST (TDD approach)
3. Weekly integration testing
4. Senior code reviews on all refactoring PRs

---

## COST ESTIMATION

| Phase | Effort | Cost | Dependencies |
|-------|--------|------|--------------|
| **Phase 1** | 11-15 days | $3-4.5k | None |
| **Phase 2** | 7-10 days | $2-3k | Phase 1 complete |
| **Phase 3** | 9-13 days | $2.5-4k | Phases 1-2 complete |
| **TOTAL** | 27-38 days | $7.5-11.5k | 6-8 weeks |

*Assumes ~$200-300/day senior dev, $100-150/day mid dev*

---

## NEXT STEPS

1. **Week 1:** Assign owners, create feature branches
2. **Weekly:** Sync meetings to unblock issues
3. **End of each phase:** Integration testing, code review
4. **Continuous:** Update test coverage dashboard

---

**Prepared by:** GitHub Copilot  
**Last Updated:** March 11, 2026
