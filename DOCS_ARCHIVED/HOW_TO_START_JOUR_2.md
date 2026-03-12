# 🎯 How to Start Jour 2 (Code Migration)

**Once you're ready to migrate code to DDD layers, follow this.**

---

## 📋 Pre-Migration Checklist

Before you start migrating code on Jour 2, verify:

- [ ] Backend runs: `python ml-backend/app.py` → Port 8000 OK
- [ ] You can `curl http://localhost:8000/health` → Returns `{"status":"ok"}`
- [ ] Validation passes: `python validate_phase_1.py` → Green report
- [ ] You've read [CODE_INVENTORY.md](CODE_INVENTORY.md) → Know what to migrate
- [ ] You've read your project's Phase 2 guide → Understand the approach

---

## 🗺️ Code Migration Strategy

### Priority Order (Migrate These First)

**CRITICAL (Do First) - Jour 2-3**
1. **infrastructure/data/** - Data loading pipeline
   - From: `src/explorationdata/pipeline/data_loader.py`
   - To: `ml-backend/src/ds_covid_backend/infrastructure/data/`
   - Tests: Unit tests for loading

2. **infrastructure/image/** - Image processing
   - From: `src/features/Pipelines/transformateurs/*.py`
   - To: `ml-backend/src/ds_covid_backend/infrastructure/image/`
   - Tests: Image augmentation tests

**HIGH (Do Next) - Jour 3-4**
3. **domain/models/** - ML models
   - From: `src/ds_covid/models.py`
   - To: `ml-backend/src/ds_covid_backend/domain/models/`
   - Tests: TensorFlow CNN tests

4. **domain/entities/** - Business objects
   - From: `src/ds_covid/features.py`
   - To: `ml-backend/src/ds_covid_backend/domain/entities/`
   - Tests: Entity validation tests

**MEDIUM (Do Last) - Jour 5-8**
5. **application/services/** - Higher-level logic
   - From: `src/explorationdata/pipeline/pipeline_runner.py`
   - To: `ml-backend/src/ds_covid_backend/application/services/`
   - Tests: Integration tests

6. **infrastructure/tensorflow/** - Framework wrappers
   - From: ML framework usages
   - To: `ml-backend/src/ds_covid_backend/infrastructure/tensorflow/`
   - Tests: Framework wrapper tests

---

## 📝 Migration Template

When migrating each file, follow this pattern:

```python
# 1. NEW FILE: ml-backend/src/ds_covid_backend/infrastructure/data/data_loader.py

from typing import List, Tuple
import numpy as np
import pandas as pd

class DataLoader:
    """Load COVID-19 dataset from disk."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_images(self) -> np.ndarray:
        """Load images from dataset."""
        # YOUR CODE HERE FROM OLD data_loader.py
        pass
    
    def load_labels(self) -> List[str]:
        """Load labels from dataset."""
        # YOUR CODE HERE FROM OLD data_loader.py
        pass
```

```python
# 2. NEW FILE: ml-backend/tests/unit/infrastructure/test_data_loader.py

import pytest
from ds_covid_backend.infrastructure.data.data_loader import DataLoader

class TestDataLoader:
    def test_load_images(self):
        # YOUR TEST HERE
        assert True
```

---

## ✅ Migration Checklist (Per File)

For each file you migrate:

- [ ] Create target file in DDD structure
- [ ] Copy code from old location
- [ ] Fix imports (they'll be different now)
- [ ] Add type hints where missing
- [ ] Write unit tests (2-3 per function)
- [ ] Run tests until passing
- [ ] Delete old file
- [ ] Commit to git: `git add . && git commit -m "Migrate [module] to DDD"`

---

## 🧪 Testing Strategy

### Unit Tests (40% Coverage Target)
```python
# For each function, write tests for:
# 1. Happy path (normal input)
# 2. Edge case (empty input)
# 3. Error case (bad input)
```

### Run Tests
```powershell
cd ml-backend
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Coverage Report
```
Expected: >= 40% coverage of src/
Minimum: No uncovered critical paths
```

---

## 🐍 Python Style Guide

For **all migrated code**, follow:

1. **Type Hints**: Every function
   ```python
   def load_data(path: str) -> pd.DataFrame:
   ```

2. **Docstrings**: Every class & public function
   ```python
   """Load COVID-19 dataset from CSV."""
   ```

3. **Error Handling**: Expected errors
   ```python
   if not path.exists():
       raise FileNotFoundError(f"Dataset not found: {path}")
   ```

4. **Logging**: Important operations
   ```python
   logger = logging.getLogger(__name__)
   logger.info(f"Loaded {len(data)} samples")
   ```

---

## 📦 Imports Cleanup

When moving files, update imports:

```python
# OLD (in src/explorationdata/)
from ...ds_covid.models import CovidModel
from ...utils.config import load_config

# NEW (in ml-backend/src/ds_covid_backend/infrastructure/)
from src.ds_covid_backend.domain.models import CovidModel
from src.ds_covid_backend.config import load_config
```

---

## 🗂️ Target Structure (After Migration)

```
ml-backend/src/ds_covid_backend/
├── api/
│   └── routes/
│       ├── __init__.py
│       ├── health.py        ✅ Already has /health endpoint
│       ├── predict.py       (Coming Jour 6)
│       └── models.py        (Coming Jour 6)
│
├── domain/
│   ├── __init__.py
│   ├── models/              (Migrate from src/ds_covid/models.py)
│   │   ├── __init__.py
│   │   └── covid_model.py   (TensorFlow CNN)
│   │
│   └── entities/            (Migrate from src/ds_covid/features.py)
│       ├── __init__.py
│       └── prediction.py    (Business objects)
│
├── application/
│   ├── __init__.py
│   ├── services/            (Migrate from pipeline_runner.py)
│   │   ├── __init__.py
│   │   └── prediction.py    (PredictionService)
│   │
│   └── use_cases/           (New high-level logic)
│       ├── __init__.py
│       └── predict.py       (Prediction use case)
│
├── infrastructure/
│   ├── __init__.py
│   ├── data/                (Migrate from explorationdata/pipeline/)
│   │   ├── __init__.py
│   │   └── loader.py        (DataLoader)
│   │
│   ├── image/               (Migrate from features/Pipelines/)
│   │   ├── __init__.py
│   │   ├── loader.py        (ImageLoader)
│   │   ├── preprocessing.py (ImagePreprocessor)
│   │   └── augmentation.py  (ImageAugmenter)
│   │
│   └── tensorflow/          (New, wrapper for TensorFlow)
│       ├── __init__.py
│       └── model_runner.py  (TensorFlow wrapper)
│
└── config/
    ├── __init__.py
    └── settings.py          ✅ Already created

tests/
├── unit/
│   ├── infrastructure/
│   │   ├── test_data_loader.py
│   │   └── test_image_processor.py
│   ├── domain/
│   │   └── test_models.py
│   └── application/
│       └── test_services.py
│
└── integration/
    └── test_end_to_end.py
```

---

## 🎓 Daily Schedule (Jour 2-8)

### Jour 2: Infrastructure/Data
- [ ] Migrate data_loader.py
- [ ] Migrate pipeline_runner.py structure
- [ ] Write data loader tests
- **Deliverable:** DataLoader class with 70% test coverage

### Jour 3: Infrastructure/Image
- [ ] Migrate image_loaders.py
- [ ] Migrate image_preprocessing.py
- [ ] Migrate image_augmentation.py
- **Deliverable:** Image pipeline with tests

### Jour 4: Domain/Models
- [ ] Migrate models.py (TensorFlow CNN)
- [ ] Create model factory
- [ ] Write model tests
- **Deliverable:** CovidModel class + tests

### Jour 5: Domain/Entities
- [ ] Create business entities
- [ ] Migrate features.py logic
- [ ] Write entity tests
- **Deliverable:** Entity classes + validators

### Jour 6: Application/Services
- [ ] Create PredictionService
- [ ] Create TrainingService
- [ ] Write service tests
- **Deliverable:** Service layer + integration tests

### Jour 7: Clean Up
- [ ] Remove old files
- [ ] Fix remaining imports
- [ ] Reach 40% test coverage
- **Deliverable:** Ready for API integration

### Jour 8: Review & Plan
- [ ] Code review
- [ ] Performance testing
- [ ] Doc update
- **Deliverable:** Phase 2 complete, Phase 3 ready

---

## 🔧 Daily Workflow

```
9:00 AM  - Review tasks for day (this day's schedule)
9:15 AM  - Migrate one file
10:00 AM - Write tests for that file
11:00 AM - Run tests until passing
12:00 PM - Commit to git
1:00 PM  - Lunch break
2:00 PM  - Next file (repeat 3x)
5:00 PM  - Code review & documentation
5:30 PM  - Update progress notes
6:00 PM  - Done for day
```

---

## ✨ Success Indicators

By end of Jour 2:
- [ ] data_loader.py migrated and tested
- [ ] 30+ unit tests written
- [ ] Zero import errors
- [ ] Test coverage >= 50% for infrastructure/

By end of Jour 4:
- [ ] Models migrated and tested
- [ ] 60+ unit tests written
- [ ] All imports working
- [ ] Test coverage >= 40% overall

By end of Jour 8:
- [ ] All 44 files migrated
- [ ] 100+ unit tests
- [ ] Zero breaking changes
- [ ] Test coverage = 40%
- [ ] **Phase 2 COMPLETE** ✅

---

## 🆘 Common Migration Issues

| Issue | Solution |
|-------|----------|
| Import Error | Update sys.path imports |
| Missing dependency | Add to requirements.txt |
| Test fails | Check data path, adjust fixtures |
| TensorFlow error | Verify model serialization |
| Memory issue | Add batch processing |

---

## 📞 Need Help?

When stuck:
1. Check [CODE_INVENTORY.md](CODE_INVENTORY.md) for file location
2. Look at similar migrated file
3. See `_REFACTORING_MICROSERVICE_/01_ARCHITECTURE/` for patterns
4. Read `_REFACTORING_MICROSERVICE_/04_GUIDES_JOUR_PAR_JOUR/Jour_X.md`

---

## 🚀 Ready?

When ready to start Jour 2:

1. ✅ Verify backend is running
2. ✅ Read [CODE_INVENTORY.md](CODE_INVENTORY.md)
3. ✅ Read your day's guide in `04_GUIDES_JOUR_PAR_JOUR/`
4. ✅ Create your first file
5. ✅ Write tests
6. ✅ Run tests
7. ✅ Commit
8. 🎯 **PROCEED TO JOUR 2!**

---

**You've got this! 💪**

*Good luck with Phase 2!*
