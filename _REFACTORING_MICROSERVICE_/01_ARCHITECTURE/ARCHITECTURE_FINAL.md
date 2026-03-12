# Architecture Finale - Microservice avec src/

C'est la **BONNE structure** avec `src/` (DDD-compliant + Production-ready)

## 📐 Structure Finale (À Créer)

```
DS_COVID/
│
├── 📂 ml-backend/                          # Service Backend (FastAPI)
│   ├── 📂 src/
│   │   └── 📂 ds_covid_backend/            # Package principal (racine importable)
│   │       ├── __init__.py
│   │       ├── 📂 api/                     # Couche API (entrée HTTP)
│   │       │   ├── __init__.py
│   │       │   ├── 📂 errors/              # Exceptions HTTP
│   │       │   ├── 📂 middlewares/         # CORS, logging, etc
│   │       │   ├── 📂 schemas/             # Pydantic models (request/response)
│   │       │   │   ├── __init__.py
│   │       │   │   ├── request.py
│   │       │   │   └── response.py
│   │       │   └── 📂 routes/              # Endpoints
│   │       │       ├── __init__.py
│   │       │       ├── health.py           # /health
│   │       │       ├── predict.py          # /predict
│   │       │       └── metrics.py          # /metrics
│   │       │
│   │       ├── 📂 domain/                  # Logic métier (pur, pas de dépendances)
│   │       │   ├── __init__.py
│   │       │   ├── 📂 models/              # Entités (datamodels)
│   │       │   │   ├── __init__.py
│   │       │   │   ├── prediction.py       # Prediction entity
│   │       │   │   └── image.py            # Image entity
│   │       │   └── 📂 repositories/        # Interfaces (abstractions)
│   │       │       ├── __init__.py
│   │       │       └── ml_model_repository.py
│   │       │
│   │       ├── 📂 application/             # Use cases / Business logic
│   │       │   ├── __init__.py
│   │       │   ├── 📂 predict_service/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── predict_service.py  # Logique prédiction
│   │       │   │   └── validators.py
│   │       │   ├── 📂 data_processor/      # Prétraitement
│   │       │   │   ├── __init__.py
│   │       │   │   ├── preprocessing.py
│   │       │   │   ├── augmentation.py
│   │       │   │   └── utils.py
│   │       │   └── 📂 training_service/    # Training logic (futur)
│   │       │       ├── __init__.py
│   │       │       └── trainer.py
│   │       │
│   │       ├── 📂 infrastructure/          # Implémentations concrètes
│   │       │   ├── __init__.py
│   │       │   ├── 📂 ml_models/           # Chargement modèles
│   │       │   │   ├── __init__.py
│   │       │   │   ├── model_loader.py
│   │       │   │   └── tensorflow_model.py
│   │       │   ├── 📂 storage/             # Base de données, cache
│   │       │   │   ├── __init__.py
│   │       │   │   ├── sqlite_repo.py
│   │       │   │   └── memory_cache.py
│   │       │   └── 📂 logging/             # Logging structuré
│   │       │       ├── __init__.py
│   │       │       └── logger.py
│   │       │
│   │       ├── 📂 config/                  # Configuration
│   │       │   ├── __init__.py
│   │       │   ├── settings.py             # Config centralisée (Pydantic)
│   │       │   └── 📂 templates/
│   │       │
│   │       └── main.py                     # Point d'entrée (FastAPI app)
│   │
│   ├── 📂 tests/                           # Tests (mirror src/ structure)
│   │   ├── __init__.py
│   │   ├── conftest.py                     # Pytest fixtures
│   │   ├── 📂 unit/                        # Unit tests
│   │   │   ├── __init__.py
│   │   │   ├── test_preprocessing.py
│   │   │   ├── test_predict_service.py
│   │   │   └── test_model_loader.py
│   │   ├── 📂 integration/                 # Integration tests
│   │   │   ├── __init__.py
│   │   │   ├── test_api_endpoints.py
│   │   │   └── test_predict_flow.py
│   │   └── 📂 fixtures/                    # Données de test
│   │       ├── __init__.py
│   │       ├── dummy_images/
│   │       └── dummy_data.py
│   │
│   ├── 📂 notebooks/                       # Dev/Exploration (Git LFS)
│   │   ├── 01_eda.ipynb
│   │   ├── 02_model_baseline.ipynb
│   │   └── 03_model_training.ipynb
│   │
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── README.md
│   ├── .env.example
│   └── .gitignore
│
├── 📂 frontend/                            # Service Frontend
│   ├── 📂 src/
│   │   └── 📂 app/
│   │       ├── streamlit_app.py
│   │       ├── 📂 pages/
│   │       ├── 📂 components/
│   │       └── config/
│   ├── requirements.txt
│   └── Dockerfile
│
├── 📂 infrastructure/                      # Docker + K8s
│   ├── docker-compose.yml
│   ├── 📂 kubernetes/
│   └── 📂 scripts/
│
├── 📂 migration_backup/                    # 🔑 Anciens fichiers (à supprimer)
│   ├── 📂 src_old/                         # Ancien src/ original
│   ├── 📂 notebooks_old/                   # Vieux notebooks
│   ├── 📂 pages_old/                       # Vieilles pages Streamlit
│   └── .gitkeep
│
├── 📂 docs/                                # Documentation projet
│   ├── SETUP.md
│   ├── SPECIFICATION.md
│   ├── API.md
│   └── DEPLOYMENT.md
│
├── 📂 docs_guides/                         # 🔑 Guides de refactoring (à supprimer après)
│   ├── START_HERE.md
│   ├── RESUME_EXECUTIF.md
│   ├── ARCHITECTURE_FINAL.md               # Ce fichier!
│   ├── PLAN_D_ACTION_DETAILLE.md
│   ├── CHECKLIST_PROGRESSION.md
│   └── INDEX_DOCS.md
│
├── 📂 data/                                # Données (gitignore)
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
│
├── 📂 models/                              # Artefacts (gitignore)
│   ├── trained/
│   ├── checkpoints/
│   └── .gitkeep
│
├── .gitignore                              # À créer
├── pyproject.toml                          # Root config (optionnel)
└── README.md                               # Main project README
```

---

## 🎯 Que Mettre Où?

### `domain/` - Pur métier (0 dépendances externes)
```python
# NE PAS importer FastAPI, TensorFlow, etc ici!
# Juste la logic métier

# ✅ BON
class Prediction:
    """Entité prédiction"""
    class_name: str
    confidence: float
    
def validate_prediction(pred: Prediction) -> bool:
    return pred.confidence > 0.5
```

### `application/` - Use cases (logique applicative)
```python
# Utilise domain/ + infrastructure/
# Les "règles métier" appliquées

from domain.models import Prediction
from infrastructure.ml_models import ModelLoader

class PredictionService:
    def predict(self, image) -> Prediction:
        model = ModelLoader().load()
        result = model.predict(image)
        return Prediction(...)
```

### `infrastructure/` - Implémentations
```python
# Concrétisations: TensorFlow, BD, cache, etc
# Tout ce qui est "technique"

import tensorflow as tf
from domain.repositories import MLModelRepository

class TensorflowModel(MLModelRepository):
    def load(self):
        return tf.keras.models.load_model(...)
```

### `api/` - Points d'entrée HTTP
```python
# Reçoit requête HTTP → appelle service → rend réponse

from fastapi import APIRouter
from application.predict_service import PredictionService

router = APIRouter()

@router.post("/predict")
def predict(image: UploadFile):
    service = PredictionService()
    result = service.predict(image)
    return result
```

---

## 📁 Créer la Structure (Commands)

```bash
# D'abord, créer la structure skeleton
cd ml-backend

# Package principal
mkdir -p src/ds_covid_backend/{api,domain,application,infrastructure,config}
mkdir -p src/ds_covid_backend/api/{routes,schemas,errors,middlewares}
mkdir -p src/ds_covid_backend/domain/{models,repositories}
mkdir -p src/ds_covid_backend/application/{predict_service,data_processor,training_service}
mkdir -p src/ds_covid_backend/infrastructure/{ml_models,storage,logging}

# Tests
mkdir -p tests/{unit,integration,fixtures}

# Notebooks
mkdir -p notebooks

# Créer __init__.py everywhere
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

# Vérifier structure
tree -d src/
```

---

## 🔄 Structure Migration

```
AVANT (Chaos):              APRÈS (DDD):
├─ src/                     ├─ src/
│  ├─ ds_covid/             │  └─ ds_covid_backend/
│  │  ├─ config.py          │     ├─ api/
│  │  ├─ models.py          │     ├─ domain/
│  │  ├─ features.py        │     ├─ application/
│  │  └─ cli.py             │     └─ infrastructure/
│  ├─ explorationdata/      ├─ tests/
│  ├─ features/             ├─ notebooks/
│  ├─ interpretability/     └─ migration_backup/
│  └─ utils/                   ├─ src_old/ ← Old files
│  
└─ notebooks/
```

---

## ✅ .gitignore (Important!)

```bash
# migration_backup/ - On le supprime après refactorisation
migration_backup/
*_backup/
*_old/

# __pycache__ et compilés
__pycache__/
*.pyc
*.pyo
*.egg-info/
.Python

# Données et modèles (trop gros)
data/raw/
data/processed/
models/trained/
models/checkpoints/
*.h5
*.pkl
notebooks/.ipynb_checkpoints/

# Logs
logs/
*.log

# Env
.env
.env.local
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Tests
.pytest_cache/
.coverage
htmlcov/
```

---

## 🎯 Microservice vs Non-Microservice

| Aspect | Ton Archi Actuelle | Avec `src/` | Microservice |
|--------|-------------------|-----------|-------------|
| **Structure** | 😫 Désorganisée | ✅ DDD-compliant | ✅ Clean |
| **Import** | ❌ Confus | ✅ `from src.ds_covid_backend.api import...` | ✅ Package-based |
| **Testabilité** | ❌ Difficile | ✅ Facile (inversion contrôle) | ✅ Très facile |
| **Scalabilité** | ❌ Non | ✅ Oui | ✅ Oui |
| **Réutilisabilité** | ❌ Non | ✅ Oui (pip install) | ✅ Oui (containerized) |
| **Microservice?** | ❌ Non | ⚠️ Presque | ✅ Oui |

**Réponse:** Avec `src/` = **80% microservice** (manque juste Docker/K8s). Sans `src/` = **0% microservice**.

---

## 📂 docs_guides/ (À SUPPRIMER après refactoring)

```
docs_guides/              # ← Mettre ici pour pouvoir rm facilement
├── START_HERE.md
├── RESUME_EXECUTIF.md
├── ARCHITECTURE_FINAL.md
├── PLAN_D_ACTION_DETAILLE.md
├── CHECKLIST_PROGRESSION.md
└── INDEX_DOCS.md

# Après refactoring:
rm -rf docs_guides/
```

---

## 🔑 Fichiers sur Git

```
À versionner (git add):
├── src/                   ✅ Code source
├── tests/                 ✅ Tests
├── docs/                  ✅ Documentation projet
└── infrastructure/        ✅ Docker + K8s

À .gitignore:
├── migration_backup/      ❌ (temporaire, à supprimer)
├── docs_guides/           ❌ (temporaire, à supprimer après)
├── models/trained/        ❌ (trop gros)
├── data/                  ❌ (data confidentielle)
├── logs/                  ❌ (logs runtime)
└── __pycache__/           ❌ (compilés)
```

---

## 🚀 Avant vs Après

### AVANT (actuellement)
```bash
$ cd DS_COVID
$ python -c "from src.ds_covid.models import Model"
```
❌ Confus: Est-ce `ds_covid.models` ou `src.models`?

### APRÈS (avec structure src/)
```bash
$ cd DS_COVID/ml-backend
$ python -c "from src.ds_covid_backend.infrastructure.ml_models import ModelLoader"
```
✅ Clair: Package → Layer → Module

---

## 📋 Prochaines Étapes

### **Phase 1 (Demain):**
1. Créer structure skeleton (voir commands ci-dessus)
2. Créer `migration_backup/` (pour vieux fichiers)
3. Créer `docs_guides/` (pour guides de refactoring)
4. Créer `.gitignore` (voir plus haut)

### **Phase 2 (Jour +1):**
5. Migrer code existant into structure:
   - `src/ds_covid/models.py` → `src/ds_covid_backend/infrastructure/ml_models/`
   - `src/ds_covid/features.py` → `src/ds_covid_backend/application/data_processor/`
   - Copier anciens files dans `migration_backup/` (juste cas)

### **Phase 3 (Jour +3):**
6. Remplir chaque layer progressivement
7. Tests au fur et à mesure
8. Supprimer `migration_backup/` une fois sûr

---

##  Pourquoi cette structure?

✅ **DDD-Compliant** - Chaque layer a une responsabilité
✅ **Production-Ready** - Comme les vrais microservices
✅ **Testable** - Inversion de dépendances (facile de mocker)
✅ **Scalable** - Facile d'ajouter features
✅ **Cloud-Ready** - Prêt pour Docker + Kubernetes
✅ **Maintenable** - Code organisé, clairement structuré

---

**La VRAIE différence microservice = `src/` + DDD + Docker + Kubernetes**

Tu avais raison! Avec un `src/`, c'est BIEN plus microservice-compliant! 🎯
