# Script: Create ML-Backend Structure with src/ (Windows PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File create_structure.ps1

Write-Host "🚀 Creating DS_COVID ML Backend Structure..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""

# Navigate to ml-backend
if (-not (Test-Path "ml-backend")) {
    New-Item -ItemType Directory -Path "ml-backend" | Out-Null
}
Set-Location "ml-backend"

# 1. Create src/ structure
Write-Host "📦 Creating src/ directory..." -ForegroundColor Cyan

$dirs = @(
    "src/ds_covid_backend",
    "src/ds_covid_backend/api/routes",
    "src/ds_covid_backend/api/schemas",
    "src/ds_covid_backend/api/errors",
    "src/ds_covid_backend/api/middlewares",
    "src/ds_covid_backend/domain/models",
    "src/ds_covid_backend/domain/repositories",
    "src/ds_covid_backend/application/predict_service",
    "src/ds_covid_backend/application/data_processor",
    "src/ds_covid_backend/application/training_service",
    "src/ds_covid_backend/infrastructure/ml_models",
    "src/ds_covid_backend/infrastructure/storage",
    "src/ds_covid_backend/infrastructure/logging",
    "src/ds_covid_backend/config",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "notebooks",
    "migration_backup",
    "logs",
    "data/raw",
    "data/processed",
    "models/trained",
    "models/checkpoints"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  ✓ Created $dir" -ForegroundColor Green
    }
}

# 2. Create __init__.py files in src/
Write-Host ""
Write-Host "✨ Creating __init__.py files..." -ForegroundColor Cyan

$pythonDirs = Get-ChildItem -Path "src" -Directory -Recurse
foreach ($dir in $pythonDirs) {
    $initFile = Join-Path $dir.FullName "__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile | Out-Null
    }
}

$testDirs = Get-ChildItem -Path "tests" -Directory -Recurse
foreach ($dir in $testDirs) {
    $initFile = Join-Path $dir.FullName "__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile | Out-Null
    }
}

Write-Host "  ✓ Created __init__.py files" -ForegroundColor Green

# 3. Create .gitkeep files
Write-Host ""
Write-Host "📌 Creating .gitkeep files..." -ForegroundColor Cyan

Create-Item -ItemType File -Path "data/.gitkeep" -Force | Out-Null
Create-Item -ItemType File -Path "models/.gitkeep" -Force | Out-Null
Create-Item -ItemType File -Path "logs/.gitkeep" -Force | Out-Null

Write-Host "  ✓ Created .gitkeep files" -ForegroundColor Green

# 4. Create .gitignore
Write-Host ""
Write-Host "🔧 Creating .gitignore..." -ForegroundColor Cyan

$gitignore = @"
# Migration backup (temporaire)
migration_backup/
_old/
*_backup/

# Python
__pycache__/
*.py[cod]
*`$py.class
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
"@

Set-Content -Path ".gitignore" -Value $gitignore
Write-Host "  ✓ Created .gitignore" -ForegroundColor Green

# 5. Create .env.example
Write-Host "📝 Creating .env.example..." -ForegroundColor Cyan

$envExample = @"
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
"@

Set-Content -Path ".env.example" -Value $envExample
Write-Host "  ✓ Created .env.example" -ForegroundColor Green

# 6. Create pyproject.toml
Write-Host "📄 Creating pyproject.toml..." -ForegroundColor Cyan

$pyproject = @"
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
"@

Set-Content -Path "pyproject.toml" -Value $pyproject
Write-Host "  ✓ Created pyproject.toml" -ForegroundColor Green

# 7. Create requirements.txt
Write-Host "📋 Creating requirements.txt..." -ForegroundColor Cyan

$requirements = @"
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
"@

Set-Content -Path "requirements.txt" -Value $requirements
Write-Host "  ✓ Created requirements.txt" -ForegroundColor Green

# 8. Create README.md
Write-Host "📚 Creating README.md..." -ForegroundColor Cyan

$readme = @"
# DS_COVID ML Backend

FastAPI service for COVID-19 detection from radiological images.

## Structure

\`\`\`
src/ds_covid_backend/
├── api/          # HTTP endpoints
├── domain/       # Business entities
├── application/  # Use cases & services
├── infrastructure/  # Implementations (TensorFlow, BD, etc)
└── config/       # Configuration
\`\`\`

## Quick Start

\`\`\`bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src

# Run API
uvicorn src.ds_covid_backend.main:app --reload --port 8000
\`\`\`

## API Docs

Once running: http://localhost:8000/docs
"@

Set-Content -Path "README.md" -Value $readme
Write-Host "  ✓ Created README.md" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "✅ Structure Created Successfully!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "📊 What was created:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ✓ src/ds_covid_backend/  (Package principal)" -ForegroundColor Green
Write-Host "    ├─ api/               (Routes HTTP)" -ForegroundColor Gray
Write-Host "    ├─ domain/            (Entités métier)" -ForegroundColor Gray
Write-Host "    ├─ application/       (Use cases)" -ForegroundColor Gray
Write-Host "    ├─ infrastructure/    (Implémentations)" -ForegroundColor Gray
Write-Host "    └─ config/            (Configuration)" -ForegroundColor Gray
Write-Host ""
Write-Host "  ✓ tests/                (Tests)" -ForegroundColor Green
Write-Host "    ├─ unit/              (Unit tests)" -ForegroundColor Gray
Write-Host "    ├─ integration/       (Integration tests)" -ForegroundColor Gray
Write-Host "    └─ fixtures/          (Test data)" -ForegroundColor Gray
Write-Host ""
Write-Host "  ✓ notebooks/            (Jupyter notebooks)" -ForegroundColor Green
Write-Host "  ✓ migration_backup/     (Anciens fichiers → à supprimer)" -ForegroundColor Yellow
Write-Host "  ✓ logs/, data/, models/ (Runtime directories)" -ForegroundColor Green
Write-Host ""
Write-Host "📄 Configuration files:" -ForegroundColor Yellow
Write-Host "  ✓ .gitignore      (Ignored files)" -ForegroundColor Green
Write-Host "  ✓ .env.example    (Environment variables template)" -ForegroundColor Green
Write-Host "  ✓ pyproject.toml  (Python project config)" -ForegroundColor Green
Write-Host "  ✓ requirements.txt (Dependencies)" -ForegroundColor Green
Write-Host "  ✓ README.md       (Project documentation)" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. cd ml-backend" -ForegroundColor White
Write-Host "  2. python -m venv venv" -ForegroundColor White
Write-Host "  3. venv\Scripts\activate           (Windows)" -ForegroundColor White
Write-Host "     source venv/bin/activate        (Linux/Mac)" -ForegroundColor White
Write-Host "  4. pip install -r requirements.txt" -ForegroundColor White
Write-Host "  5. Move existing code to migration_backup/" -ForegroundColor White
Write-Host "  6. Follow ARCHITECTURE_FINAL.md to structure your code" -ForegroundColor White
Write-Host ""
Write-Host "View structure:" -ForegroundColor Cyan
Write-Host "  tree /F src\  # (install 'tree' with: choco install tree)" -ForegroundColor Gray
Write-Host ""
