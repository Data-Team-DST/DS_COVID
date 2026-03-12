# 📖 JOUR 2-5: Migration Code vers Architecture Microservice

**Objectif:** Migrer le code existant depuis `src/`, `notebooks/` vers `ml-backend/src/ds_covid_backend/` avec structure DDD

**Durée:** 5 jours (Jour 2-6)

**Livrable:** Code existant réorganisé dans DDD layers + premiers tests

---

## 🎯 Vue d'ensemble Jour 2-5

| Jour | Tâche | Durée | Outcome |
|------|-------|-------|---------|
| **Jour 2** | Explorer code existant + identifier modules réutilisables | 1-2h | Liste de modules à migrer |
| **Jour 3** | Migrer data_utils, ML models vers infrastructure/ + domain/ | 2-3h | Data pipeline working |
| **Jour 4** | Créer premier service (UseCase) dans application/ | 1-2h | Inference service |
| **Jour 5** | Ajouter premiers endpoints FastAPI | 2h | 3+ endpoints |
| **Jour 6** | Écrire tests (target 40%) | 2-3h | 40% coverage |

---

## JOUR 2: Explorer & Mapper Code Existant

### 🔍 Étape 1: Examiner structure actuelle

```bash
# Explorer src/ old
cd migration_backup/src_old/

# Voir les modules
ls -la

# Voir taille des fichiers
du -sh *
```

**Fichiers à chercher:**
- `*.py` files (modules Python)
- `__init__.py` (packages)
- Classes principales (Models, DataLoaders, etc.)
- Fichiers d'utilitaires (utils, helpers)

### 📋 Étape 2: Créer une liste d'inventaire

Ouvre un éditeur et crée `JOUR_2_CODE_INVENTORY.md`:

```markdown
# Code Inventory - Migration Plan

## Modules à Migrer

### 1. Data Pipeline
- [ ] data_utils.py → infrastructure/data_loader.py
  - Classes: DataLoader, DataValidator
  - Size: ~500 lines
  - Dependencies: pandas, numpy

- [ ] preprocessing.py → infrastructure/preprocessor.py
  - Classes: Preprocessor, Normalizer
  - Dependencies: scikit-learn

### 2. ML Models
- [ ] model_builders.py → domain/models.py
  - Classes: CovidModel, ModelFactory
  - Size: ~300 lines

- [ ] training_utils.py → application/training_service.py
  - Functions: train(), evaluate()
  - Dependencies: TensorFlow

### 3. Utilities
- [ ] visualization_utils.py → infrastructure/visualization.py
  - For EDA/debugging
  - Keep in infrastructure (external-facing)

- [ ] config.py → config/settings.py
  - Configuration management
  - Environment variables

### 4. NOT to MIGRATE
- [ ] Jupyter notebooks → Keep in analysis/
- [ ] Old test files → Rewrite with new structure
```

### 🗂️ Étape 3: Map de DDD (Rappel)

```
STRUCTURE DDD:

API Layer (api/)
  ↓
  FastAPI routes call

Application Layer (application/)
  ↓
  UseCases / Services
  Example: PredictionService.predict(X)

Domain Layer (domain/)
  ↓
  Business Logic + Models
  Example: CovidModel, PredictionEntity

Infrastructure Layer (infrastructure/)
  ↓
  Data Access + External Services
  Example: DataLoader, TensorFlow wrapper
```

### 🔀 Étape 4: Mappings Clés

```python
OLD LOCATIONS → NEW LOCATIONS

src/ds_covid/features.py
  ├─ FeatureExtractor class → domain/entities/feature.py
  └─ feature_utils functions → infrastructure/feature_processor.py

src/ds_covid/models.py
  ├─ CovidModel → domain/models/covid_model.py
  ├─ ModelLoader → infrastructure/model_loader.py
  └─ ModelFactory → application/services/model_service.py

src/ds_covid/utils/
  ├─ data_utils.py → infrastructure/data_loader.py
  ├─ training_utils.py → application/services/training_service.py
  ├─ visualization_utils.py → infrastructure/visualization.py
  └─ config.py → config/settings.py

notebooks/ (Python files)
  ├─ EDA notebooks → Keep as-is
  ├─ Training pipeline → Refactor to application/
  └─ Prediction notebook → Refactor to api/
```

### ⚠️ Imports à Adapter

Quand tu migres, TOUS les imports changent:

```python
# OLD
from src.ds_covid.models import CovidModel
from src.ds_covid.utils.data_utils import DataLoader

# NEW
from ds_covid_backend.domain.models import CovidModel
from ds_covid_backend.infrastructure.data_loader import DataLoader

# In app.py or tests:
# (because ml-backend/src is added to PYTHONPATH)
```

---

## Next Steps After Jour 2

### Jour 3 Tasks
1. Créer `ml-backend/src/ds_covid_backend/infrastructure/data_loader.py`
2. Copier/refactoriser data loading logic
3. Tester imports

### Jour 4 Tasks
1. Créer `ml-backend/src/ds_covid_backend/domain/models.py`
2. Copier modèles ML
3. Adapter dépendances

### Jour 5 Tasks
1. Créer service dans `application/`
2. Ajouter première API route
3. Tests simples

### Jour 6 Tasks
1. Ajouter plus d'endpoints
2. Écrire tests pour 40% coverage
3. Fix bugs détectés

---

## 🛠️ Tools für Jour 2

### Terminal Commands

```bash
# Explorer structure
tree src_old/ -L 2

# Count lines
find src_old -name "*.py" | xargs wc -l

# Find imports
grep -r "^import\|^from" src_old/ | head -20

# Find classes
grep -r "^class " src_old/
```

### Python Script (analyse_modules.py)

```python
import os
import ast

def analyze_module(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    return {
        'classes': classes,
        'functions': functions,
        'lines': len(open(filepath).readlines())
    }

# Usage
for root, dirs, files in os.walk('migration_backup/src_old'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            print(f"\n{path}")
            print(analyze_module(path))
```

---

## 📋 Checklist Jour 2

- [ ] Explorer migration_backup/src_old/
- [ ] Lire tous les fichiers Python
- [ ] Identifier classes principales
- [ ] Créer JOUR_2_CODE_INVENTORY.md
- [ ] Lire dependencies (imports)
- [ ] Créer mapping: OLD → NEW locations
- [ ] Valider aucune dépendance circulaire
- [ ] Planifier ordre de migration (bottom-up)

**Done?** → Prêt pour Jour 3! 💪

---

## Exemple: Migration réelle

### Before - OLD CODE (src/ds_covid/utils/data_utils.py)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None
    
    def load(self):
        self.data = pd.read_csv(self.path)
        return self.data
    
    def validate(self):
        assert not self.data.isnull().sum().any()
        return True
```

### After - NEW CODE (ml-backend/src/ds_covid_backend/infrastructure/data_loader.py)

```python
# File: ml-backend/src/ds_covid_backend/infrastructure/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class DataLoader:
    """Load and validate COVID-19 dataset."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: pd.DataFrame = None
    
    def load(self) -> pd.DataFrame:
        """Load CSV data."""
        self.data = pd.read_csv(self.path)
        return self.data
    
    def validate(self) -> bool:
        """Validate data integrity."""
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        if self.data.isnull().sum().any():
            raise ValueError("Data contains null values")
        
        return True
```

### Using NEW CODE (in tests)

```python
# File: ml-backend/tests/unit/test_data_loader.py

from ds_covid_backend.infrastructure.data_loader import DataLoader

def test_load_data():
    loader = DataLoader('data/covid.csv')
    data = loader.load()
    assert data is not None
    assert loader.validate()
```

---

## 🎓 Key Principles

1. **Move incrementally** - Migrate one module at a time
2. **Test as you go** - Add tests after each migration
3. **Refactor, don't copy** - Improve code while moving
4. **Type hints** - Add type hints for clarity
5. **Documentation** - Document why code is where it is

---

**Next:** Jour 3 - Infrastructure Layer (data loading)
