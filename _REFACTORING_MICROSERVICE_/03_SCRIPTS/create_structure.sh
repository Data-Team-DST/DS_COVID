#!/bin/bash
# Script: Create ML-Backend Structure with src/
# Usage: bash create_structure.sh

set -e  # Exit on error

echo "🚀 Creating DS_COVID ML Backend Structure..."
echo "=================================================="

# Navigate to ml-backend
mkdir -p ml-backend && cd ml-backend

# 1. Create src/ structure
echo "📦 Creating src/ directory..."
mkdir -p src/ds_covid_backend/{api,domain,application,infrastructure,config}

# 1a. API layer
mkdir -p src/ds_covid_backend/api/{routes,schemas,errors,middlewares}

# 1b. Domain layer
mkdir -p src/ds_covid_backend/domain/{models,repositories}

# 1c. Application layer
mkdir -p src/ds_covid_backend/application/{predict_service,data_processor,training_service}

# 1d. Infrastructure layer
mkdir -p src/ds_covid_backend/infrastructure/{ml_models,storage,logging}

# 2. Create tests/ structure  
echo "🧪 Creating tests/ directory..."
mkdir -p tests/{unit,integration,fixtures}

# 3. Create notebooks/
echo "📓 Creating notebooks/ directory..."
mkdir -p notebooks

# 4. Create other dirs
echo "📂 Creating other directories..."
mkdir -p migration_backup
mkdir -p logs
mkdir -p data/{raw,processed}
mkdir -p models/{trained,checkpoints}

# 5. Create __init__.py everywhere in src/
echo "✨ Creating __init__.py files..."
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

# 6. Create .keepfiles where needed
touch data/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep

# 7. Create main configuration files
echo "🔧 Creating configuration files..."

# .gitignore
cat > .gitignore << 'EOF'
# Migration backup (temporaire)
migration_backup/
_old/
*_backup/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data & Models (gros fichiers)
data/raw/
data/processed/
models/trained/
models/checkpoints/
*.h5
*.pkl
*.joblib
*.onnx

# Jupyter
notebooks/.ipynb_checkpoints/
.ipynb_checkpoints/

# Logs
logs/
*.log

# Environment
.env
.env.local
.env.*.local
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Docker
.dockerignore

# OS
Thumbs.db
.DS_Store
EOF

echo "✅ Created .gitignore"

# .env.example
cat > .env.example << 'EOF'
# Application
PROJECT_NAME=DS_COVID_Backend
DEBUG=false
LOG_LEVEL=INFO

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

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Monitoring
TRACK_PREDICTIONS=true
METRICS_FILE=metrics/predictions.json
EOF

echo "✅ Created .env.example"

# pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "ds-covid-backend"
version = "0.1.0"
description = "ML Backend for COVID-19 Detection"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic==2.5.0",
    "pydantic-settings==2.1.0",
    "tensorflow==2.15.0",
    "numpy==1.24.3",
    "pandas==2.1.3",
    "scikit-learn==1.3.2",
    "Pillow==10.1.0",
    "python-multipart==0.0.6",
    "opencv-python==4.8.1.78",
    "joblib==1.3.2",
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
addopts = "-v --tb=short --cov=src"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
select = ["E", "F", "W", "I"]
line-length = 100
EOF

echo "✅ Created pyproject.toml"

# requirements.txt
cat > requirements.txt << 'EOF'
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
EOF

echo "✅ Created requirements.txt"

# README.md
cat > README.md << 'EOF'
# DS_COVID ML Backend

FastAPI service for COVID-19 detection from radiological images.

## Structure

```
src/ds_covid_backend/
├── api/          # HTTP endpoints
├── domain/       # Business entities
├── application/  # Use cases & services
├── infrastructure/  # Implementations (TensorFlow, BD, etc)
└── config/       # Configuration
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src

# Run API
uvicorn src.ds_covid_backend.main:app --reload --port 8000
```

## API Docs

Once running: http://localhost:8000/docs
EOF

echo "✅ Created README.md"

# 8. Summary
echo ""
echo "=================================================="
echo "✅ Structure Created Successfully!"
echo "=================================================="
echo ""
echo "📊 What was created:"
echo "  ✓ src/ds_covid_backend/  (Package principal)"
echo "    ├─ api/               (Routes HTTP)"
echo "    ├─ domain/            (Entités métier)"
echo "    ├─ application/       (Use cases)"
echo "    ├─ infrastructure/    (Implémentations)"
echo "    └─ config/            (Configuration)"
echo ""
echo "  ✓ tests/                (Tests)"
echo "    ├─ unit/              (Unit tests)"
echo "    ├─ integration/       (Integration tests)"
echo "    └─ fixtures/          (Test data)"
echo ""
echo "  ✓ notebooks/            (Jupyter notebooks)"
echo "  ✓ migration_backup/     (Anciens fichiers - à supprimer)"
echo "  ✓ logs/, data/, models/ (Runtime directories)"
echo ""
echo "📄 Configuration files:"
echo "  ✓ .gitignore      (Ignored files)"
echo "  ✓ .env.example    (Environment variables template)"
echo "  ✓ pyproject.toml  (Python project config)"
echo "  ✓ requirements.txt (Dependencies)"
echo "  ✓ README.md       (Project documentation)"
echo ""
echo "🚀 Next steps:"
echo "  1. cd ml-backend"
echo "  2. python -m venv venv"
echo "  3. source venv/bin/activate  (or venv\Scripts\activate on Windows)"
echo "  4. pip install -r requirements.txt"
echo "  5. mv existing code to migration_backup/"
echo "  6. Follow ARCHITECTURE_FINAL.md to structure your code"
echo ""
echo "View structure:"
echo "  tree -d -L 3 src/"
echo ""
