# Create ml-backend structure with DDD architecture
Write-Host "Creating ml-backend structure..." -ForegroundColor Green

# Create main directories
$dirs = @(
    "ml-backend/src/ds_covid_backend/api",
    "ml-backend/src/ds_covid_backend/domain",
    "ml-backend/src/ds_covid_backend/application",
    "ml-backend/src/ds_covid_backend/infrastructure",
    "ml-backend/src/ds_covid_backend/config",
    "ml-backend/tests/unit",
    "ml-backend/tests/integration",
    "ml-backend/tests/fixtures"
)

foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Cyan
    }
}

# Create __init__.py files
$initFiles = @(
    "ml-backend/src/ds_covid_backend/__init__.py",
    "ml-backend/src/ds_covid_backend/api/__init__.py",
    "ml-backend/src/ds_covid_backend/domain/__init__.py",
    "ml-backend/src/ds_covid_backend/application/__init__.py",
    "ml-backend/src/ds_covid_backend/infrastructure/__init__.py",
    "ml-backend/src/ds_covid_backend/config/__init__.py",
    "ml-backend/tests/__init__.py",
    "ml-backend/tests/unit/__init__.py",
    "ml-backend/tests/integration/__init__.py",
    "ml-backend/tests/fixtures/__init__.py"
)

foreach ($file in $initFiles) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created: $file" -ForegroundColor Yellow
    }
}

# Create requirements.txt
$requirements = "# Core Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0

# ML & Data Processing
tensorflow==2.15.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.0.3
opencv-python==4.8.0.76
Pillow==10.0.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.0

# Development
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0

# Documentation
python-multipart==0.0.6"

Set-Content -Path "ml-backend/requirements.txt" -Value $requirements
Write-Host "Created: ml-backend/requirements.txt" -ForegroundColor Yellow

# Create pyproject.toml
$pyproject = "[build-system]
requires = [\"setuptools>=65.0\", \"wheel\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"ds_covid_backend\"
version = \"0.1.0\"
description = \"COVID-19 ML Backend API with FastAPI and TensorFlow\"
requires-python = \">=3.11\"

[tool.setuptools]
packages = [\"ds_covid_backend\"]
package-dir = {\"\" = \"src\"}

[tool.pytest.ini_options]
testpaths = [\"tests\"]
addopts = \"--cov=src/ds_covid_backend --cov-report=term-missing -v\""

Set-Content -Path "ml-backend/pyproject.toml" -Value $pyproject
Write-Host "Created: ml-backend/pyproject.toml" -ForegroundColor Yellow

# Create setup.py
$setup = "from setuptools import setup, find_packages

setup(
    name=\"ds_covid_backend\",
    version=\"0.1.0\",
    package_dir={\"\":\"src\"},
    packages=find_packages(where=\"src\"),
    python_requires=\">=3.11\",
)"

Set-Content -Path "ml-backend/setup.py" -Value $setup
Write-Host "Created: ml-backend/setup.py" -ForegroundColor Yellow

# Create .env.example
$env = "# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false"

Set-Content -Path "ml-backend/.env.example" -Value $env
Write-Host "Created: ml-backend/.env.example" -ForegroundColor Yellow

# Create sample API main.py
$apiMain = "# API Routes
from fastapi import APIRouter

router = APIRouter()

@router.get(\"/health\")
async def health_check():
    return {\"status\": \"ok\"}"

Set-Content -Path "ml-backend/src/ds_covid_backend/api/main.py" -Value $apiMain
Write-Host "Created: ml-backend/src/ds_covid_backend/api/main.py" -ForegroundColor Yellow

# Create sample app.py
$appFile = "from fastapi import FastAPI
from src.ds_covid_backend.api.main import router

app = FastAPI(title=\"COVID-19 ML API\", version=\"0.1.0\")
app.include_router(router)"

Set-Content -Path "ml-backend/app.py" -Value $appFile
Write-Host "Created: ml-backend/app.py" -ForegroundColor Yellow

# Create sample test
$testFile = "def test_sample():
    assert 1 + 1 == 2"

Set-Content -Path "ml-backend/tests/unit/test_sample.py" -Value $testFile
Write-Host "Created: ml-backend/tests/unit/test_sample.py" -ForegroundColor Yellow

Write-Host "`nStructure created successfully!" -ForegroundColor Green
