# Plan d'Action Détaillé - DS_COVID Refactoring

## 📋 Résumé Exécutif

Pour atteindre les 5 objectifs demandés avec architecture microservice:

| Objectif | Durée | Complexité |
|----------|-------|-----------|
| 1️⃣ Définir objectifs + métriques | 2-3 jours | ⭐ Facile |
| 2️⃣ Environnement reproductible | 3-4 jours | ⭐ Facile |
| 3️⃣ Données + prétraitement | 5-7 jours | ⭐⭐ Moyen |
| 4️⃣ Modèle baseline + tests | 7-10 jours | ⭐⭐⭐ Difficile |
| 5️⃣ API d'inférence | 5-7 jours | ⭐⭐ Moyen |
| 🐳 Docker + Deployment | 5-7 jours | ⭐⭐⭐ Difficile |
| 🧹 Cleanup codebase | 5-7 jours | ⭐⭐ Moyen |

**Total:** 4-5 semaines | **Équipe:** 1-2 développeurs

---

## 🚀 PHASE 1: Foundation (Jours 1-5)

### Jour 1: Objectifs & Métriques ✅

**Fichier à créer:** `docs/SPECIFICATION.md`

```markdown
# Spécification Projet DS_COVID

## 🎯 Objectif Global
Développer une application ML pour diagnostic COVID-19 basée sur images radiologiques
avec API REST production-ready et interface web.

## 📊 Objectifs Spécifiques

### 1. Classification d'Imagerie
- Classifier images radiologiques: COVID-19 vs Normal vs Pneumonie
- Confiance minimale: 85% accuracy

### 2. API d'Inférence 
- Endpoint HTTP `/predict` acceptant images/données
- Temps de réponse < 500ms
- Scalable (potentiellement Kubernetes)

### 3. Interface Utilisateur
- Prédictions interactives
- Analytics/Métriques modèle
- Historique prédictions

### 4. Code Quality
- Tests unitaires ≥ 40% coverage
- Code maintenable et documenté
- CI/CD automatisé

## 🎲 Métriques Clés

### Performance Modèle (Données de test)
```
Métrique              | Target | Calcul
---------------------|--------|--------
Accuracy             | ≥ 85%  | (TP+TN)/(TP+FP+TN+FN)
Sensitivity/Recall   | ≥ 80%  | TP/(TP+FN)
Specificity          | ≥ 90%  | TN/(TN+FP)
Precision            | ≥ 83%  | TP/(TP+FP)
F1-Score             | ≥ 0.83 | 2*(P*R)/(P+R)
AUC-ROC              | ≥ 0.92 | Area under ROC curve
```

### Infrastructure & API
```
Métrique             | Target  | Calcul
---------------------|---------|--------
Latence P95          | < 500ms | perf: predict()
Uptime               | ≥ 99.5% | (Temps up) / (Temps total)
Throughput           | ≥ 10 req/s | Requêtes/seconde
Test Coverage        | ≥ 40%   | Lignes testées
```

### Données
```
Métrique             | Value    | Notes
---------------------|----------|--------
Train Set Size       | TBD      | Dépend données dispo
Val Set Size         | 15%      | De train
Test Set Size        | 15%      | De train
Image Resolution     | 224x224  | Standard CNN
Augmentation         | Enabled  | Rotation, flip, zoom
```

## 🔄 Success Criteria (Definition of Done)

- [ ] Modèle trained et validé
- [ ] API testée avec ≥ 20 test cases
- [ ] Frontend deployable
- [ ] Docker images built
- [ ] CI/CD pipeline green
- [ ] Documentation complète
- [ ] Performance metrics atteints
```

**Durée:** 2-3 jours | **Propriétaire:** Product Owner + Tech Lead

---

### Jours 2-3: Environnement Reproductible 🔧

**À faire:**

#### 2.1 Créer structure de base

```bash
# Depuis le repo root
mkdir -p ml-backend ml-backend/app ml-backend/tests ml-backend/app/api
mkdir -p frontend
mkdir -p infrastructure infrastructure/kubernetes
mkdir -p docs data models

# Créer .gitignore
cat > .gitignore << 'EOF'
# Environment
venv/
env/
.env
.env.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data & Models (important - ne pas commit)
data/raw/
data/processed/
models/trained/
models/checkpoints/
*.h5
*.pkl
*.joblib

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Docker
.dockerignore
EOF
```

#### 2.2 Configuration de base

**ml-backend/pyproject.toml:**
```toml
[project]
name = "ds-covid-backend"
version = "0.1.0"
description = "ML Backend for COVID-19 Detection"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic-settings==2.1.0",
    "tensorflow==2.15.0",
    "numpy==1.24.3",
    "pandas==2.1.3",
    "scikit-learn==1.3.2",
    "pillow==10.1.0",
    "python-multipart==0.0.6",
    "opencv-python==4.8.1.78",
]

[project.optional-dependencies]
dev = [
    "pytest==7.4.3",
    "pytest-cov==4.1.0",
    "black==23.12.0",
    "ruff==0.1.8",
    "mypy==1.7.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --cov=app"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
select = ["E", "F", "W", "I"]
line-length = 100
```

**ml-backend/requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
tensorflow==2.15.0
numpy==1.24.3
pandas==2.1.3
scikit-learn==1.3.2
Pillow==10.1.0
python-multipart==0.0.6
opencv-python==4.8.1.78
joblib==1.3.2
pytest==7.4.3
pytest-cov==4.1.0
```

#### 2.3 Créer structure logging

**ml-backend/app/logging_config.py:**
```python
import logging
import logging.config
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": LOG_DIR / "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "INFO",
            "handlers": ["console", "file"],
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logging.config.dictConfig(LOGGING_CONFIG)
    logging.getLogger().setLevel(level)
    logging.info(f"Logging initialized at level {level}")

def get_logger(name: str) -> logging.Logger:
    """Get named logger"""
    return logging.getLogger(name)
```

#### 2.4 Configuration centralisée

**ml-backend/app/config.py:**
```python
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "DS_COVID_Backend"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model
    MODEL_NAME: str = "covid_detector"
    MODEL_PATH: Path = Path("models/trained/model.h5")
    MODEL_TYPE: Literal["cnn", "xgboost", "ensemble"] = "cnn"
    
    # Data
    DATA_PATH: Path = Path("data/processed")
    IMG_SIZE: tuple = (224, 224)
    IMG_CHANNELS: int = 3
    BATCH_SIZE: int = 32
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("logs/app.log")
    
    # Monitoring
    TRACK_PREDICTIONS: bool = True
    METRICS_FILE: Path = Path("metrics/predictions.json")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow reading from .env
        extra = "allow"

settings = Settings()

# Validate on startup
assert settings.MODEL_PATH.exists() or settings.DEBUG, \
    f"Model path {settings.MODEL_PATH} not found"
```

#### 2.5 Environment de développement

**ml-backend/.env (template):**
```
# Application
PROJECT_NAME=DS_COVID_Backend
DEBUG=true
LOG_LEVEL=DEBUG

# Server
HOST=0.0.0.0
PORT=8000

# Model
MODEL_NAME=covid_detector
MODEL_PATH=models/trained/cnn_covid.h5
MODEL_TYPE=cnn

# Data
DATA_PATH=data/processed
IMG_SIZE=224
BATCH_SIZE=32

# Monitoring
TRACK_PREDICTIONS=true
METRICS_FILE=metrics/predictions.json
```

**Installation:**
```bash
cd ml-backend

# Créer virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -e ".[dev]"
# ou
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Vérifier installation
python -c "import fastapi; print(fastapi.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Durée:** 3-4 jours | **Checklist:**
- [ ] Structure directories créée
- [ ] pyproject.toml dans ml-backend/
- [ ] requirements.txt actualisé
- [ ] logging_config.py créé
- [ ] config.py centralisée créé
- [ ] .env template créé
- [ ] venv testée (`pip list`)
- [ ] Imports basiques testés

---

### Jours 4-5: Nettoyage Codebase 🧹

**À faire:**

#### 4.1 Migrer code existant utile

```bash
# Copier les modules utiles du src/ existant
cp -r src/ds_covid/models.py ml-backend/app/models/
cp -r src/ds_covid/features.py ml-backend/app/features/
cp -r src/utils/training_utils.py ml-backend/app/utils/
cp -r src/utils/visualization_utils.py ml-backend/app/utils/

# Refactoriser imports
# Exemple: from src.ds_covid.models import * 
#       → from app.models import *
```

#### 4.2 Supprimer/restructurer ancien code

```bash
# À supprimer (duplicated/legacy):
rm -rf notebooks/old/
rm -rf src/features/raf/               # Config #3 (gardé config.py seulement)
rm -rf notebooks/modelebaseline/        # Legacy code
rm src/ds_covid/config.py              # Config #2 (gardé app/config.py)

# À garder:
- notebooks/04_apply_masks.ipynb       → ml-backend/notebooks/01_preprocessing.ipynb
- Complete_EDA_COVID_Dataset.ipynb     → ml-backend/notebooks/02_eda.ipynb
- page/*.py                            → frontend/pages/ (adapter pour Streamlit)
```

#### 4.3 Créer README structure

**ml-backend/README.md:**
```markdown
# DS_COVID ML Backend

FastAPI service for COVID-19 detection from radiological images.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Development
uvicorn app.main:app --reload --port 8000

# Tests
pytest tests/ -v --cov=app
```

## Project Structure

```
├── app/
│   ├── main.py          # Entry point
│   ├── config.py        # Configuration
│   ├── logging_config.py # Logging
│   ├── api/            # Routes
│   ├── models/         # ML models
│   ├── features/       # Preprocessing
│   └── utils/          # Utilities
├── tests/              # Test suite
├── notebooks/          # Development
└── requirements.txt    # Dependencies
```

## API Documentation

Once running, visit: http://localhost:8000/docs

## Monitoring

- Logs: `logs/app.log`
- Metrics: `metrics/predictions.json`
- Health: `GET /api/v1/health`
```

**Durée:** 2-3 jours | **Checklist:**
- [ ] Code utile migré
- [ ] Old notebooks supprimés
- [ ] Imports refactorisés
- [ ] README créé
- [ ] `.gitignore` actualisé
- [ ] Structure de base testée

---

## 🔧 PHASE 2: ML Core (Jours 6-15)

### Jour 6-7: Import & Refactoring Données

**ml-backend/app/features/preprocessing.py:**
```python
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Preprocessing pipeline for radiological images"""
    
    def __init__(self, 
                 img_size: tuple = (224, 224),
                 normalize: bool = True):
        self.img_size = img_size
        self.normalize = normalize
    
    def load_image(self, path: str | Path) -> np.ndarray:
        """Load and preprocess single image"""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Convert to RGB (3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        if self.normalize:
            img = img.astype('float32') / 255.0
        
        logger.debug(f"Loaded image {path}: shape={img.shape}")
        return img
    
    def preprocess_batch(self, images: list[str]) -> np.ndarray:
        """Load multiple images"""
        batch = [self.load_image(img) for img in images]
        return np.array(batch)

class DataAugmentation:
    """Image augmentation for training"""
    
    def __init__(self, rotation_range=20, zoom_range=0.2, 
                 horizontal_flip=True):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations"""
        # Implementation with albumentations or TensorFlow
        pass
```

**Durée:** 2-3 jours | **Checklist:**
- [ ] Data loader classes créées
- [ ] Image preprocessing testée
- [ ] Augmentation configurée
- [ ] Dataset mappé (classes)
- [ ] Train/Val/Test split défini

---

### Jour 8-10: Model Training

**ml-backend/app/models/model_builder.py:**
```python
import tensorflow as tf
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class COVIDModelBuilder:
    """Factory for COVID detection models"""
    
    @staticmethod
    def build_cnn(input_shape: Tuple[int, ...], 
                  num_classes: int = 3) -> tf.keras.Model:
        """Build baseline CNN"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Block 3
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Dense layers
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()],
        )
        
        logger.info(f"Built CNN model: {model.count_params()} parameters")
        return model
    
    @staticmethod
    def build_transfer_learning(num_classes: int = 3) -> tf.keras.Model:
        """Build model with transfer learning (MobileNetV2)"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base
        base_model.trainable = False
        
        # Add custom top
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        
        logger.info(f"Built Transfer Learning model")
        return model
```

**ml-backend/notebooks/03_model_training.ipynb:**

Cell 1 - Imports & Setup:
```python
import sys
sys.path.insert(0, '..')

from app.config import settings
from app.features.preprocessing import ImagePreprocessor
from app.models.model_builder import COVIDModelBuilder
from app.logging_config import setup_logging
import tensorflow as tf
import numpy as np
from pathlib import Path

setup_logging()
```

Cell 2 - Load Data:
```python
# Setup (adapter to your data)
DATA_PATH = Path(settings.DATA_PATH)
X_train = ...  # Load your data
y_train = ...
X_val = ...
y_val = ...

print(f"Train shape: {X_train.shape}, classes: {np.unique(y_train)}")
```

Cell 3 - Build & Train:
```python
# Build model
model = COVIDModelBuilder.build_cnn(
    input_shape=(224, 224, 3),
    num_classes=3
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/trained/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
    ]
)

print(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

Cell 4 - Evaluate:
```python
# Test prediction
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_true_classes, y_pred_classes)
prec = precision_score(y_true_classes, y_pred_classes, average='weighted')
rec = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
```

**Durée:** 3-5 jours | **Checklist:**
- [ ] Model builder créé
- [ ] Baseline CNN entraîné
- [ ] Métriques calculées
- [ ] Best model sauvegardé
- [ ] Target metrics atteints (≥85% acc)

---

### Jour 11-15: Unit Tests

**ml-backend/tests/conftest.py:**
```python
import pytest
import numpy as np
from pathlib import Path
from app.config import settings
from app.features.preprocessing import ImagePreprocessor

@pytest.fixture
def preprocessor():
    """Image preprocessor fixture"""
    return ImagePreprocessor(img_size=(224, 224))

@pytest.fixture
def dummy_image() -> np.ndarray:
    """Generate dummy image for testing"""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

@pytest.fixture
def dummy_batch() -> np.ndarray:
    """Generate dummy batch"""
    return np.random.rand(10, 224, 224, 3).astype('float32')

@pytest.fixture
def test_model():
    """Load test model"""
    # Load or create small model for testing
    pass
```

**ml-backend/tests/unit/test_preprocessing.py:**
```python
import pytest
import numpy as np
from app.features.preprocessing import ImagePreprocessor

def test_preprocessor_init():
    prep = ImagePreprocessor(img_size=(224, 224))
    assert prep.img_size == (224, 224)

def test_image_shape(preprocessor, dummy_image, tmp_path):
    """Test image resizing"""
    # Save dummy image
    img_path = tmp_path / "test.png"
    import cv2
    cv2.imwrite(str(img_path), dummy_image)
    
    # Load with preprocessor
    img = preprocessor.load_image(img_path)
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.float32

def test_batch_loading(preprocessor, tmp_path):
    """Test batch loading"""
    import cv2
    # Create test images
    paths = []
    for i in range(5):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        paths.append(str(p))
    
    batch = preprocessor.preprocess_batch(paths)
    assert batch.shape == (5, 224, 224, 3)

def test_normalization(preprocessor):
    """Test image normalization"""
    img = np.ones((224, 224, 3), dtype='float32')
    # Check values are in [0, 1]
    assert np.max(img) <= 1.0
    assert np.min(img) >= 0.0
```

**ml-backend/tests/unit/test_models.py:**
```python
import pytest
import tensorflow as tf
from app.models.model_builder import COVIDModelBuilder

def test_cnn_build():
    """Test CNN model creation"""
    model = COVIDModelBuilder.build_cnn(
        input_shape=(224, 224, 3),
        num_classes=3
    )
    
    assert model is not None
    assert len(model.layers) > 0
    assert model.output_shape == (None, 3)

def test_model_prediction(test_model):
    """Test model inference"""
    import numpy as np
    X = np.random.rand(2, 224, 224, 3).astype('float32')
    
    y = test_model.predict(X, verbose=0)
    assert y.shape == (2, 3)
    assert np.all((y >= 0) & (y <= 1))
    assert np.allclose(y.sum(axis=1), 1.0)
```

**Run tests:**
```bash
pytest tests/ -v --cov=app

# Output:
# tests/unit/test_preprocessing.py::test_preprocessor_init PASSED
# tests/unit/test_models.py::test_cnn_build PASSED
# ...
# ============ 15 passed in 2.34s =============
# Coverage: app 42%
```

**Durée:** 5-7 jours | **Checklist:**
- [ ] conftest.py avec fixtures
- [ ] Unit tests preprocessing
- [ ] Unit tests models
- [ ] ≥40% coverage atteint
- [ ] Tous tests green

---

## 💻 PHASE 3: API Backend (Jours 16-21)

### Jour 16: Request/Response Schemas

**ml-backend/app/schemas/request.py:**
```python
from pydantic import BaseModel, Field
from typing import List, Optional
import base64

class PredictImageRequest(BaseModel):
    """Prediction request with base64 encoded image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    model_name: Optional[str] = "covid_detector"

class PredictBatchRequest(BaseModel):
    """Batch prediction request"""
    images: List[str] = Field(..., description="List of base64 images")

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool
```

**ml-backend/app/schemas/response.py:**
```python
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime

class PredictionResult(BaseModel):
    """Single prediction result"""
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float

class PredictImageResponse(BaseModel):
    """API prediction response"""
    result: PredictionResult
    timestamp: datetime

class ModelInfoResponse(BaseModel):
    """Model information"""
    name: str
    version: str
    input_shape: tuple
    num_classes: int
    classes: List[str]
    accuracy: float
```

### Jour 17-19: API Implementation

**ml-backend/app/main.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.logging_config import setup_logging
from app.api import health, predict

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events"""
    logger.info("🚀 API Starting...")
    yield
    logger.info("⛔ API Shutting down...")

# Create app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging(settings.LOG_LEVEL)

# Include routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
    )
```

**ml-backend/app/api/health.py:**
```python
from fastapi import APIRouter
from app.config import settings
from app.schemas.response import HealthCheckResponse
from pathlib import Path
import logging

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint"""
    model_loaded = settings.MODEL_PATH.exists()
    
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=model_loaded,
    )

@router.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "name": settings.MODEL_NAME,
        "type": settings.MODEL_TYPE,
        "input_shape": settings.IMG_SIZE,
        "num_classes": 3,
        "classes": ["COVID-19", "Normal", "Pneumonie"],
    }
```

**ml-backend/app/api/predict.py:**
```python
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.request import PredictImageRequest
from app.schemas.response import PredictImageResponse, PredictionResult
from app.config import settings
from app.models.model_loader import ModelLoader
import logging
import base64
import cv2
import numpy as np
from datetime import datetime
import time

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)

# Global model loader
model_loader = ModelLoader(settings.MODEL_PATH)

@router.post("/predict", response_model=PredictImageResponse)
async def predict(request: PredictImageRequest):
    """
    Predict COVID-19 from image
    
    Request:
        image_base64: Base64 encoded image
    
    Response:
        {
            "result": {
                "class_name": "COVID-19",
                "confidence": 0.92,
                "probabilities": {...},
                "processing_time_ms": 150
            },
            "timestamp": "2024-01-10T12:30:45"
        }
    """
    start_time = time.time()
    
    try:
        # Decode image
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Preprocess
        img = cv2.resize(img, settings.IMG_SIZE)
        img = img.astype('float32') / 255.0
        
        # Predict
        model = model_loader.get_model()
        predictions = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred = predictions[0]
        
        class_idx = np.argmax(pred)
        class_name = ["COVID-19", "Normal", "Pneumonie"][class_idx]
        confidence = float(pred[class_idx])
        
        # Log
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Prediction: {class_name} ({confidence:.2%})")
        
        return PredictImageResponse(
            result=PredictionResult(
                class_name=class_name,
                confidence=confidence,
                probabilities={
                    "covid19": float(pred[0]),
                    "normal": float(pred[1]),
                    "pneumonie": float(pred[2]),
                },
                processing_time_ms=processing_time,
            ),
            timestamp=datetime.now(),
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Predict from uploaded file"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ... rest of prediction logic
```

**ml-backend/app/models/model_loader.py:**
```python
import tensorflow as tf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage model"""
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model from disk"""
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model(self) -> tf.keras.Model:
        """Get model instance"""
        if self.model is None:
            self._load_model()
        return self.model
```

### Jour 20-21: API Tests

**ml-backend/tests/integration/test_api_endpoints.py:**
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
import base64
import cv2
import numpy as np

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 3
    assert "classes" in data

def test_predict_valid_image():
    """Test prediction with valid image"""
    # Create dummy image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Predict
    response = client.post(
        "/api/v1/predict",
        json={"image_base64": img_base64}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "class_name" in data["result"]
    assert "confidence" in data["result"]

def test_predict_invalid_image():
    """Test with invalid base64"""
    response = client.post(
        "/api/v1/predict",
        json={"image_base64": "not_valid_base64"}
    )
    
    assert response.status_code == 400
```

**Durée:** 6-7 jours | **Checklist:**
- [ ] Schemas créés (request/response)
- [ ] main.py avec routes
- [ ] /health endpoint
- [ ] /predict endpoint
- [ ] /model/info endpoint
- [ ] Model loader créé
- [ ] API tests ≥20 test cases
- [ ] All integration tests green
- [ ] API docs accessible (/docs)

---

## 🐳 PHASE 4: Docker & Deployment (Jours 22-26)

### Jour 22-23: Containerization

**ml-backend/Dockerfile:**
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (including OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app code
COPY app/ ./app/
COPY models/ ./models/
COPY .env .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**frontend/Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**infrastructure/docker-compose.yml:**
```yaml
version: '3.8'

services:
  # ML Backend
  ml-backend:
    build:
      context: ../ml-backend
      dockerfile: Dockerfile
    container_name: covid-ml-backend
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - MODEL_PATH=models/trained/cnn_covid.h5
    volumes:
      - ../ml-backend/logs:/app/logs
      - ../ml-backend/metrics:/app/metrics
      - ../models:/app/models
    networks:
      - covid-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Frontend
  frontend:
    build:
      context: ../frontend
      dockerfile: Dockerfile
    container_name: covid-frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://ml-backend:8000
    depends_on:
      ml-backend:
        condition: service_healthy
    networks:
      - covid-network

networks:
  covid-network:
    driver: bridge
```

### Jour 24-25: CI/CD Pipeline

**.github/workflows/tests.yml:**
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
        cd ml-backend
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        cd ml-backend
        pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./ml-backend/coverage.xml

  lint:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Lint with ruff
      run: |
        pip install ruff
        ruff check ml-backend/app/
```

**.github/workflows/build-docker.yml:**
```yaml
name: Build & Push Docker

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push backend
      uses: docker/build-push-action@v4
      with:
        context: ./ml-backend
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-backend:latest
    
    - name: Build and push frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-frontend:latest
```

### Jour 26: Kubernetes (optionnel)

**infrastructure/kubernetes/backend-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: covid-ml-backend
  labels:
    app: covid-ml-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: covid-ml-backend
  template:
    metadata:
      labels:
        app: covid-ml-backend
    spec:
      containers:
      - name: backend
        image: ghcr.io/repo/ds-covid-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: MODEL_PATH
          value: "/models/cnn_covid.h5"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1024Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: covid-ml-backend
spec:
  type: LoadBalancer
  selector:
    app: covid-ml-backend
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
```

**Deploy avec Kubernetes:**
```bash
kubectl create namespace covid-19
kubectl apply -f infrastructure/kubernetes/ -n covid-19
kubectl get pods -n covid-19
```

**Durée:** 5-7 jours | **Checklist:**
- [ ] Dockerfile backend & frontend
- [ ] docker-compose testé localement
- [ ] .github/workflows/ créé
- [ ] Tests pipeline green
- [ ] Docker images built
- [ ] Kubernetes manifests créés
- [ ] Deploy testé sur cluster

---

## 📊 Résumé Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│           DS_COVID Refactoring Timeline (6 semaines)            │
├─────────────────────────────────────────────────────────────────┤
│ Semaine 1  │ Foundation & Cleanup                               │
│ ├─ J1-3    │ Objectifs, config, logging                         │
│ ├─ J4-5    │ Nettoyage codebase                                 │
│ ├─ J6-7    │ Test suite básico                                  │
│ └─ Status   │ ✅ Core structure prête                           │
├─────────────────────────────────────────────────────────────────┤
│ Semaines 2-3│ ML Core & Training                                │
│ ├─ J8-10   │ Data pipeline + Model training                     │
│ ├─ J11-15  │ Tests (40% coverage)                               │
│ └─ Status   │ ✅ Baseline model trained & validé                │
├─────────────────────────────────────────────────────────────────┤
│ Semaines 4-5│ API Backend & Integration                         │
│ ├─ J16-17  │ Schemas + API setup                                │
│ ├─ J18-19  │ Endpoints (/predict, /health)                      │
│ ├─ J20-21  │ Integration tests                                  │
│ └─ Status   │ ✅ Production-ready API                           │
├─────────────────────────────────────────────────────────────────┤
│ Semaine 6  │ Docker & Deployment                                │
│ ├─ J22-23  │ Dockerfiles + docker-compose                       │
│ ├─ J24-25  │ CI/CD (GitHub Actions)                             │
│ ├─ J26     │ Kubernetes (optionnel)                             │
│ └─ Status   │ ✅ Production deployment ready                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Success Criteria (Final Validation)

- [ ] Tous 5 objectifs atteints
- [ ] ≥ 85% accuracy sur test set
- [ ] API latency < 500ms
- [ ] ≥ 40% test coverage
- [ ] All GitHub Actions green
- [ ] Docker images deployable
- [ ] Documentation complète
- [ ] README & SETUP guides
- [ ] Logs structurés
- [ ] Metrics tracked

---

## 📞 Support & Escalation

Si problèmes:
1. Vérifier logs: `logs/app.log`
2. Tests: `pytest tests/ -v`
3. Linter: `ruff check app/`
4. Health: `curl http://localhost:8000/api/v1/health`

---

**Prêt à commencer Phase 1 ? 🚀**
