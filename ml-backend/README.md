# COVID-19 ML Backend

Production-ready microservice backend with FastAPI and TensorFlow.

## Structure

```
ml-backend/
├── src/ds_covid_backend/
│   ├── api/           # HTTP endpoints
│   ├── domain/        # Business logic
│   ├── application/   # Use cases
│   ├── infrastructure/# Data access
│   └── config/        # Configuration
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── app.py             # FastAPI application
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

# Run tests
pytest
```

## Next Steps

1. Migrate code from src/ and notebooks/ 
2. Add more API endpoints
3. Add unit tests (target 40% coverage)
4. Setup CI/CD with GitHub Actions
