# Checklist Refactoring DS_COVID - Suivi de Progression

## 🎯 Vue d'ensemble

**Objectif global:** Transformer le codebase chaos en architecture microservice production-ready  
**Durée estimée:** 4-6 semaines  
**Équipe:** 1-2 développeurs  

---

## ✅ PHASE 1: Foundation & Setup (Jours 1-7)

### 1.1 - Objectifs & Spécification (J1-3)
- [ ] Lire `SPECIFICATION.md` créé
- [ ] Définir métriques cibles pour modèle:
  - [ ] Accuracy target: ≥ 85%
  - [ ] Sensitivity target: ≥ 80%
  - [ ] Specificity target: ≥ 90%
  - [ ] AUC-ROC target: ≥ 0.92
- [ ] Définir contraintes API:
  - [ ] Latence P95 < 500ms
  - [ ] Uptime ≥ 99.5%
  - [ ] Support batch predictions: [ ] Oui [ ] Non
- [ ] Valider avec stakeholders
- [ ] Documenter dans `docs/SPECIFICATION.md`
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 1.2 - Créer structure de base (J2-3)
```bash
# Créer répertoires
mkdir -p ml-backend/{app,tests,notebooks}
mkdir -p frontend/{pages,components,config}
mkdir -p infrastructure/{docker,kubernetes,scripts}
mkdir -p {docs,data,models}
```

- [ ] Structure créée (vérifier avec `ls -la`)
- [ ] `.gitignore` updated
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 1.3 - Configuration centralisée (J3-4)

**ml-backend:**
- [ ] `pyproject.toml` créé avec dépendances
- [ ] `requirements.txt` généré
- [ ] `requirements-dev.txt` créé
- [ ] `app/config.py` avec Settings (Pydantic)
- [ ] `.env.template` créé
- [ ] `app/logging_config.py` créé
- [ ] Tester import: `python -c "from app.config import settings"`
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 1.4 - Environnement reproductible (J4-5)

**ml-backend:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

- [ ] venv créé et activé
- [ ] Dépendances installées sans erreur
- [ ] Vérifier: `pip list` (voir tensorflow, fastapi, etc)
- [ ] Vérifier import TensorFlow: `python -c "import tensorflow as tf; print(tf.__version__)"`
- [ ] Vérifier FastAPI: `python -c "import fastapi; print(fastapi.__version__)"`
- [ ] **Status:** _____ (In Progress | ✅ Done)

**frontend:**
- [ ] `requirements.txt` créé (streamlit, pandas, numpy)
- [ ] venv créé et testé
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 1.5 - Foundation tests (J5-7)

- [ ] `tests/conftest.py` créé avec fixtures de base
- [ ] `tests/unit/test_config.py` créé et passant
- [ ] Pytest configuré: `pytest tests/ -v`
- [ ] Coverage plugin installé: `pytest-cov`
- [ ] Commande test fonctionne: `pytest tests/ --cov=app`
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 1.6 - Nettoyage codebase initiale (J5-7)

**À supprimer (sans perte):**
- [ ] `notebooks/old/` supprimé
- [ ] `src/features/raf/` legacy config supprimé (garder utils)
- [ ] `src/ds_covid/config.py` supprimé (consolidé dans `app/config.py`)
- [ ] `notebooks/modelebaseline/` legacy supprimé

**À déplacer/migrer:**
- [ ] `src/ds_covid/models.py` → `ml-backend/app/models/`
- [ ] `src/ds_covid/features.py` → `ml-backend/app/features/`
- [ ] `src/utils/training_utils.py` → `ml-backend/app/utils/`
- [ ] `src/utils/visualization_utils.py` → `ml-backend/app/utils/`
- [ ] `notebooks/Complete_EDA.ipynb` → `ml-backend/notebooks/01_eda.ipynb`

**À adapter:**
- [ ] `page/*.py` adapté pour nouveau structure (si nécessaire pour frontend)

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 1.7 - Documentation initiale (J6-7)

- [ ] `ml-backend/README.md` créé
- [ ] `docs/SETUP.md` écrit (instructions projet entier)
- [ ] `docs/SPECIFICATION.md` finalisé
- [ ] Main `README.md` updated avec architecture
- [ ] **Status:** _____ (In Progress | ✅ Done)

---

## ✅ PHASE 2: ML Core (Jours 8-15)

### 2.1 - Data Preparation (J8-10)

État données:
- [ ] Données disponibles: [ ] Oui [ ] Non → Action: Collecter dataset COVID
- [ ] Format identifié: [ ] Images (format: _____)
- [ ] Taille dataset: N = _____ images
- [ ] Classes: [ ] 2-way [ ] 3-way [ ] Multi (spécifier: _________)
- [ ] Split défini: Train = ____%, Val = ____%, Test = ____% 

Coding:
- [ ] `app/features/preprocessing.py` créé
  - [ ] Classe `ImagePreprocessor` implémentée
  - [ ] Method `load_image()` fonctionnelle
  - [ ] Method `preprocess_batch()` fonctionnelle
  - [ ] Normalization implémentée
  
- [ ] `app/features/data_loader.py` créé
  - [ ] Classe `DataGenerator` pour training
  - [ ] Support fichiers locaux
  - [ ] Support batch loading

- [ ] Test données:
  - [ ] `tests/unit/test_preprocessing.py` écrit
  - [ ] `tests/unit/test_data_loader.py` écrit
  - [ ] Tests passent: `pytest tests/unit/test_preprocessing.py -v`

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 2.2 - Model Architecture (J10-11)

- [ ] `app/models/model_builder.py` créé avec:
  - [ ] `build_cnn()` - Baseline CNN
  - [ ] `build_transfer_learning()` - MobileNetV2 ou ResNet50
  - [ ] `build_ensemble()` (optionnel pour phase 2)

- [ ] Modèles compillés avec:
  - [ ] Bon optimizer (Adam, SGD)
  - [ ] Bonne loss function (categorical_crossentropy)
  - [ ] Metrics: accuracy, precision, recall, AUC

- [ ] Test modèles:
  - [ ] `tests/unit/test_models.py` écrit
  - [ ] Test shape output: `pytest tests/unit/test_models.py::test_cnn_build -v`
  - [ ] Test prediction shape
  - [ ] Test forward pass (small batch)

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 2.3 - Training Pipeline (J11-14)

Notebook `ml-backend/notebooks/03_model_training.ipynb`:

- [ ] Cell 1: Imports & setup
- [ ] Cell 2: Load training data
  - [ ] X_train shape: _________
  - [ ] y_train shape: _________
  - [ ] X_val shape: _________
  - [ ] Data normalized: [ ] Oui

- [ ] Cell 3: Build & compile model
  - [ ] CNN model created
  - [ ] Compiled avec metrics

- [ ] Cell 4: Train model
  ```python
  history = model.fit(X_train, y_train,
      validation_data=(X_val, y_val),
      epochs=50,
      batch_size=32,
      callbacks=[...])
  ```
  - [ ] Training complété (epochs ≥ 30)
  - [ ] Val loss diminue: [ ] Oui
  - [ ] Best model sauvegardé

- [ ] Cell 5: Evaluate on test set
  - [ ] Test accuracy: ______ (target ≥ 85%)
  - [ ] Test sensitivity: ______ (target ≥ 80%)
  - [ ] Test specificity: ______ (target ≥ 90%)
  - [ ] Test AUC: ______ (target ≥ 0.92)
  - [ ] Confusion matrix généré
  - [ ] Classification report généré

- [ ] Model saved:
  - [ ] Location: `models/trained/best_model.h5`
  - [ ] Size: ______ MB
  - [ ] Loadable: `tf.keras.models.load_model(...)`

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 2.4 - Unit Testing (J14-15)

Coverage targets:
- [ ] `app/features/` coverage: ≥ 80%
- [ ] `app/models/` coverage: ≥ 60%
- [ ] `app/utils/` coverage: ≥ 70%
- [ ] Overall coverage: **≥ 40%**

Tests à avoir:
- [ ] `tests/unit/test_preprocessing.py` (≥ 5 tests)
- [ ] `tests/unit/test_models.py` (≥ 5 tests)
- [ ] `tests/unit/test_data_loader.py` (≥ 5 tests)
- [ ] `tests/unit/test_config.py` (≥ 3 tests)

Runner:
```bash
pytest tests/unit/ -v --cov=app --cov-report=html
# Vérifier: coverage > 40%
```

- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] Coverage rapport généré
- [ ] Coverage ≥ 40%: ______ % actuellement
- [ ] **Status:** _____ (In Progress | ✅ Done)

---

## ✅ PHASE 3: API Backend (Jours 16-21)

### 3.1 - API Schemas (J16)

**Pydantic Models:**

- [ ] `app/schemas/request.py` créé:
  - [ ] `PredictImageRequest` (avec image_base64)
  - [ ] `PredictBatchRequest`
  - [ ] Validation tests passent

- [ ] `app/schemas/response.py` créé:
  - [ ] `PredictionResult` (class_name, confidence, probabilities)
  - [ ] `PredictImageResponse` (result + timestamp)
  - [ ] `ModelInfoResponse`
  - [ ] `HealthCheckResponse`

- [ ] Test schemas:
  - [ ] `tests/unit/test_schemas.py` écrit
  - [ ] Tests passent: `pytest tests/unit/test_schemas.py -v`

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 3.2 - FastAPI Setup (J17)

- [ ] `app/main.py` créé:
  - [ ] FastAPI app initialized
  - [ ] CORS middleware ajouté
  - [ ] Lifespan context managers
  - [ ] Logging setup

- [ ] `app/api/__init__.py` créé
- [ ] Run test:
  ```bash
  uvicorn app.main:app --reload --port 8000
  # Vérifier: http://localhost:8000/docs exists
  ```

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 3.3 - API Endpoints (J17-19)

**Health endpoint (`/api/v1/health`):**

- [ ] `app/api/health.py` créé
- [ ] GET `/api/v1/health` → returns status, version, model_loaded
- [ ] Test: `curl http://localhost:8000/api/v1/health`
- [ ] Response code: 200
- [ ] Response format: JSON ✓

**Model info endpoint (`/api/v1/model/info`):**

- [ ] GET `/api/v1/model/info` → returns model metadata
- [ ] Include: name, type, input_shape, classes, accuracy
- [ ] Test: `curl http://localhost:8000/api/v1/model/info`

**Prediction endpoint (`/api/v1/predict`):**

- [ ] `app/api/predict.py` créé
- [ ] POST `/api/v1/predict`:
  - [ ] Accept base64 encoded image
  - [ ] Preprocess image
  - [ ] Load model via ModelLoader
  - [ ] Run inference
  - [ ] Return class_name + confidence + probabilities
  - [ ] Track processing_time_ms

- [ ] Model loader (`app/models/model_loader.py`):
  - [ ] Classe `ModelLoader` créée
  - [ ] Lazy load model on first request
  - [ ] Cache model in memory
  - [ ] Handle load errors gracefully

- [ ] Test endpoints:
  ```bash
  # Health
  curl http://localhost:8000/api/v1/health
  
  # HealthStatus test with base64 image
  # ... (voir code exemple)
  ```

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 3.4 - Integration Tests (J20-21)

- [ ] `tests/integration/test_api_endpoints.py` créé
- [ ] Tests health endpoint: ✓
- [ ] Tests model info endpoint: ✓
- [ ] Tests prediction endpoint (valid image): ✓
- [ ] Tests prediction endpoint (invalid image): ✓
- [ ] Tests prediction latency < 500ms: ✓
- [ ] Tests batch prediction: ✓

Runner:
```bash
uvicorn app.main:app &
pytest tests/integration/ -v
```

- [ ] All integration tests pass: `pytest tests/integration/ -v`
- [ ] Coverage intégration: ≥ 30%
- [ ] Total coverage (unit + integration): ≥ 40%
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 3.5 - API Documentation (J20-21)

- [ ] Swagger docs générés: `/docs`
- [ ] ReDoc générés: `/redoc`
- [ ] Tous endpoints documentés avec docstrings
- [ ] Examples fournis pour requests/responses
- [ ] `docs/API.md` écrit (OpenAPI spec)
- [ ] **Status:** _____ (In Progress | ✅ Done)

---

## ✅ PHASE 4: Docker & Deployment (Jours 22-26)

### 4.1 - Docker Images (J22-23)

**Backend Dockerfile:**
- [ ] `ml-backend/Dockerfile` créé:
  - [ ] Multi-stage build (builder + runtime)
  - [ ] Dépendances installées
  - [ ] App code COPY'd
  - [ ] HEALTHCHECK configuré
  - [ ] EXPOSE 8000

Test:
```bash
cd ml-backend
docker build -t covid-ml-backend:latest .
docker run -p 8000:8000 covid-ml-backend:latest
# Vérifier: curl http://localhost:8000/api/v1/health
```

- [ ] Build succède: ✓
- [ ] Image runs: ✓
- [ ] Health check passe: ✓
- [ ] API responds: ✓

**Frontend Dockerfile:**
- [ ] `frontend/Dockerfile` créé
- [ ] Build & test succède: ✓

- [ ] **Status:** _____ (In Progress | ✅ Done)

### 4.2 - Docker Compose (J23-24)

- [ ] `infrastructure/docker-compose.yml` créé:
  - [ ] Backend service (port 8000)
  - [ ] Frontend service (port 8501)
  - [ ] Network définit (covid-network)
  - [ ] Health checks
  - [ ] Volumes pour logs/metrics/models
  - [ ] `depends_on` configured

Test:
```bash
cd infrastructure
docker-compose up -d
docker-compose ps
docker logs covid-ml-backend
docker logs covid-frontend
curl http://localhost:8000/api/v1/health
```

- [ ] `docker-compose up` succède: ✓
- [ ] All services healthy: ✓
- [ ] Backend responds: ✓
- [ ] Frontend accessible: ✓ (http://localhost:8501)
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 4.3 - CI/CD Pipeline (J24-25)

GitHub Actions:

- [ ] `.github/workflows/tests.yml` créé:
  - [ ] Trigger: push + pull_request
  - [ ] Run pytest
  - [ ] Linting (ruff, black)
  - [ ] Coverage report
  - [ ] Upload to codecov

- [ ] `.github/workflows/build.yml` créé:
  - [ ] Build Docker images
  - [ ] Push to registry (ghcr.io ou Docker Hub)
  - [ ] Trigger: tags (v*) ou main branch

Test:
```bash
git add .github/
git commit -m "Add CI/CD"
git push
# Vérifier: Actions tab sur GitHub
```

- [ ] Tests workflow runs: ✓
- [ ] Tests workflow passes: ✓ (green checkmark)
- [ ] Build workflow runs: ✓
- [ ] Images pushed to registry: ✓
- [ ] **Status:** _____ (In Progress | ✅ Done)

### 4.4 - Kubernetes (optionnel - J25-26)

- [ ] `infrastructure/kubernetes/namespace.yaml` créé
- [ ] `infrastructure/kubernetes/backend-deployment.yaml` créé
  - [ ] Replicas: 2+
  - [ ] Resource requests/limits
  - [ ] Liveness probe
  - [ ] Env vars

- [ ] `infrastructure/kubernetes/service.yaml` créé
  - [ ] Type: LoadBalancer (ou ClusterIP)
  - [ ] Ports: 8000

- [ ] `infrastructure/kubernetes/ingress.yaml` (optionnel)

Deploy test:
```bash
kubectl create namespace covid-19
kubectl apply -f infrastructure/kubernetes/ -n covid-19
kubectl get pods -n covid-19
kubectl logs -f deployment/covid-ml-backend -n covid-19
```

- [ ] Namespace créé: ✓
- [ ] Deployment créé: ✓
- [ ] Pods running: ✓
- [ ] Service accessible: ✓
- [ ] (optionnel) Ingress working: ✓
- [ ] **Status:** _____ (In Progress | ✅ Done)

---

## ✅ PHASE 5: Final Validation (Jour 27)

### 5.1 - Performance Validation

**Modèle ML:**
- [ ] Accuracy: ______ % (target ≥ 85%) ✓
- [ ] Sensitivity: ______ % (target ≥ 80%) ✓
- [ ] Specificity: ______ % (target ≥ 90%) ✓
- [ ] AUC-ROC: ______ (target ≥ 0.92) ✓
- [ ] F1-Score: ______ (target ≥ 0.83) ✓

**API Performance:**
- [ ] Latency P95: ______ ms (target < 500ms) ✓
- [ ] Throughput: ______ req/s (target ≥ 10) ✓
- [ ] Error rate: ______ % (target < 0.1%) ✓
- [ ] Health check: ✓

**Code Quality:**
- [ ] Test coverage: ______ % (target ≥ 40%) ✓
- [ ] No linting errors: `ruff check --count` = 0 ✓
- [ ] Type safety: `mypy app/` = 0 errors (optionnel) ✓
- [ ] No critical security issues ✓

### 5.2 - Documentation Completeness

- [ ] Main `README.md` updated ✓
- [ ] `docs/SETUP.md` - Setup instructions ✓
- [ ] `docs/SPECIFICATION.md` - Objectifs & métriques ✓
- [ ] `docs/API.md` - OpenAPI spec ✓
- [ ] `docs/DEPLOYMENT.md` - Docker + K8s ✓
- [ ] `ml-backend/README.md` ✓
- [ ] `frontend/README.md` ✓
- [ ] Code comments dans modules clés ✓
- [ ] Docstrings sur classes/functions ✓

### 5.3 - Deployment Readiness

- [ ] Docker images tagged with version ✓
- [ ] docker-compose tested locally ✓
- [ ] CI/CD pipelines all green ✓
- [ ] Kubernetes manifests validated ✓
- [ ] Health checks passing ✓
- [ ] Logs structured and monitored ✓

### 5.4 - Cleanup & Archival

- [ ] Old notebooks archived or deleted ✓
- [ ] Legacy code removed ✓
- [ ] .gitignore verified ✓
- [ ] Sensitive data not committed ✓
- [ ] Large files excluded ✓
- [ ] Repository tags created: `git tag -a v0.1.0 -m "Initial release"` ✓

---

## 📊 Overall Progress

```
Phase 1 [████████░░] 80% (Days 1-7)
Phase 2 [░░░░░░░░░░] 0%  (Days 8-15)
Phase 3 [░░░░░░░░░░] 0%  (Days 16-21)
Phase 4 [░░░░░░░░░░] 0%  (Days 22-26)
Phase 5 [░░░░░░░░░░] 0%  (Day 27)

Overall Progress: ▐░░░░░░░░░ 8% complete
```

**Current Phase:** Phase 1 Foundation  
**Next Milestone:** Phase 1 completion (Day 7)  
**Days Completed:** _____ / 27  
**Days Remaining:** _____ days  

---

## 📝 Notes & Blockers

### Action Items

- [ ] Task 1: _________________________ | Owner: _____ | Due: _____
- [ ] Task 2: _________________________ | Owner: _____ | Due: _____
- [ ] Task 3: _________________________ | Owner: _____ | Due: _____

### Blockers

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| Données COVID pas disponibles | 🔴 Critical | ⏳ Waiting | Action: Collecter dataset |
| TensorFlow GPU setup | 🟡 Medium | ✅ Resolved | Using CPU for Phase 2 |
| Pas assez ressources | 🔴 Critical | 🔍 TBD | Vérifier capacity |

### Decisions

- [ ] Model type à utiliser: [ ] CNN [ ] Transfer Learning [ ] Ensemble
- [ ] Déployer sur: [ ] Local [ ] AWS [ ] Azure [ ] Kubernetes
- [ ] Frontend framework: [ ] Streamlit [ ] React [ ] TBD
- [ ] Database: [ ] Oui [ ] Non ; Type: _______

---

## 🔗 Resources

- Architecture docs: [ARCHITECTURE_MICROSERVICES.md](ARCHITECTURE_MICROSERVICES.md)
- Plan détaillé: [PLAN_D_ACTION_DETAILLE.md](PLAN_D_ACTION_DETAILLE.md)
- Spécification: [docs/SPECIFICATION.md](docs/SPECIFICATION.md)
- API docs: Will be at `http://localhost:8000/docs`
- GitHub Actions: Will monitor at `.github/workflows/`

---

## 📅 Weekly Sync Agenda

**Template for weekly meetings:**

```markdown
## Week X Sync (Monday)

### ✅ Completed This Week
- [ ] Item 1
- [ ] Item 2

### 🚧 In Progress
- [ ] Item 1 (% done)
- [ ] Item 2 (% done)

### 🔴 Blockers
- Blocker 1: Impact, Action, Owner, ETA

### 📚 Next Week
- [ ] Task 1
- [ ] Task 2
```

---

Last Updated: 2024-01-10  
Status: Foundation Phase Started ✓
