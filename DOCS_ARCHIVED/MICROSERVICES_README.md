# 🏗️ COVID-19 Microservice Architecture

**Status:** ✅ Phase 1 Complete - Ready for testing

---

## 🎯 What is a Microservice Architecture?

Instead of one monolithic app, we split functionality into independent services:

```
Traditional Monolith:
┌─────────────────────────────┐
│  Frontend + Backend + DB    │  (All in one place)
└─────────────────────────────┘

Our Microservices:
┌──────────────────┐          ┌──────────────────┐
│   Streamlit      │ ←HTTP→  │   FastAPI        │
│   Frontend       │          │   Backend        │
│   (Port 8501)    │          │   (Port 8000)    │
└──────────────────┘          └──────────────────┘
        🎨                              🧠
(User Interface)            (ML Logic + Data)
```

---

## 📂 Project Structure

```
DS_COVID/
├── ml-backend/                    ← Production Backend Code
│   ├── src/ds_covid_backend/      ← DDD Architecture
│   │   ├── api/                   (FastAPI routes)
│   │   ├── domain/                (Business logic: ML models)
│   │   ├── application/           (Services: Prediction, Training)
│   │   ├── infrastructure/        (Data loading, TensorFlow)
│   │   └── config/                (Settings)
│   ├── tests/                     (Unit & integration tests)
│   ├── venv/                      (Python environment)
│   ├── app.py                     (FastAPI entry point)
│   └── requirements.txt           (Python dependencies)
│
├── streamlit_app.py               ← Frontend (Streamlit)
├── start_services.ps1             ← Launch both services (Windows)
├── start_services.sh              ← Launch both services (Linux/Mac)
├── test_microservices.ps1         ← Test architecture (Windows)
├── test_microservices.sh          ← Test architecture (Linux/Mac)
│
├── _REFACTORING_MICROSERVICE_/    ← Architecture & guides (200+ pages)
├── migration_backup/              ← Backup of old code (local only)
└── _OLD_ROOT_FILES/               ← Archived old files
```

---

## 🚀 Quick Start

### Requirements
- Python 3.11+
- Windows PowerShell / Linux Bash
- Ports 8000 and 8501 available

### Step 1: Start Both Services

#### On Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File start_services.ps1
```

#### On Linux/Mac (Bash):
```bash
bash start_services.sh
```

### Step 2: Access the Services

- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **Health Check:** http://localhost:8000/health

### Step 3: Test Everything Works

#### On Windows:
```powershell
# In another PowerShell window:
powershell -ExecutionPolicy Bypass -File test_microservices.ps1
```

#### On Linux/Mac:
```bash
# In another terminal:
bash test_microservices.sh
```

---

## 🧪 Testing the Microservice Architecture

### Manual Tests

**Test 1: Backend is alive**
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

**Test 2: Frontend is accessible**
```bash
curl http://localhost:8501
# Expected: HTML response from Streamlit
```

**Test 3: Frontend → Backend communication**
- Open http://localhost:8501
- Look for "Backend Status" section
- Should show ✅ Backend is running

### Automated Tests

Run the test script to validate entire architecture:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File test_microservices.ps1

# Linux/Mac
bash test_microservices.sh
```

Expected output:
```
Test 1: Backend Health Check ... ✓ PASS (HTTP 200)
Test 2: Backend Root Endpoint ... ✓ PASS (HTTP 200)
Test 3: Frontend Main Page ... ✓ PASS (HTTP 200)

Test Results: 3 / 3 PASSED
✅ All tests passed!
```

---

## 🔄 How the Microservices Work Together

### Request Flow

```
User opens http://localhost:8501
      ↓
Streamlit Frontend loads
      ↓
User fills form & clicks "Predict"
      ↓
Frontend sends HTTP POST to http://localhost:8000/predict
      ↓
FastAPI Backend processes request
      ↓
TensorFlow model makes prediction
      ↓
Backend returns JSON response {"prediction": "COVID-19"}
      ↓
Frontend displays result to user
```

### Architecture Layers

```
REQUEST → API Layer → Application → Domain → Infrastructure → DATA
RESPONSE ← (JSON)  ← (Services) ← (Logic)   ← (TensorFlow)

Example: Prediction Flow

1. API Layer (api/routes)
   Receives: POST /predict with image

2. Application Layer (application/)
   - PredictionService orchestrates
   - Calls domain logic

3. Domain Layer (domain/)
   - CovidModel: TensorFlow model
   - PredictionEntity: business object

4. Infrastructure Layer (infrastructure/)
   - ImageLoader: loads image file
   - TensorFlow wrapper: runs model
   - Database: saves prediction

5. Returns: JSON to frontend
```

---

## 📝 Frontend Features

The Streamlit dashboard includes:

1. **📊 Dashboard Tab**
   - System status indicators
   - Architecture overview
   - Component information

2. **🏥 Prediction Tab**
   - Upload chest X-ray image
   - Get prediction (COVID/Normal/Pneumonia/Other)

3. **📈 Model Info Tab**
   - Model architecture details
   - Performance metrics
   - Training information

4. **⚙️ System Status Tab**
   - Backend health check
   - API endpoint testing
   - Live communication test

---

## 🔌 API Endpoints

### Currently Implemented

| Method | Endpoint | Status | Purpose |
|--------|----------|--------|---------|
| GET | `/health` | ✅ | Backend health check |
| GET | `/` | ✅ | API root / welcome |

### Coming Soon (Phase 2-3)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Make prediction on image |
| GET | `/models` | List available models |
| POST | `/train` | Start model training |
| GET | `/metrics` | Get performance metrics |

---

## 🛠️ Troubleshooting

### Issue: Port 8000 already in use
```bash
# Kill existing process using port 8000
# Windows: 
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

### Issue: Backend not responding from frontend
- ✅ Check backend is running: `curl http://localhost:8000/health`
- ✅ Check firewall allows localhost connection
- ✅ Check CORS settings in FastAPI (already configured)

### Issue: Streamlit not starting
```bash
# Install Streamlit if missing
pip install streamlit

# Run Streamlit manually
streamlit run streamlit_app.py --server.port 8501
```

### Issue: Services start but tests fail
- Wait 5 seconds after starting
- Re-run test script
- Check Python/pip paths
- Verify venv activation

---

## 📊 Architecture Benefits

✅ **Scalability:** Each service can scale independently  
✅ **Maintainability:** Clear separation of concerns (DDD)  
✅ **Testability:** Can test frontend & backend separately  
✅ **Deployment:** Deploy frontend & backend independently  
✅ **Technology:** Can use different tech for each service  

---

## 🎓 Next Steps

### Phase 2: Code Migration (Jour 2-6)
1. Migrate ML models to `domain/models/`
2. Migrate data pipeline to `infrastructure/`
3. Create services in `application/`
4. Write unit tests (40% coverage target)

### Phase 3: API Endpoints (Jour 7-12)
1. Implement `/predict` endpoint
2. Implement `/models` endpoint
3. Implement `/train` endpoint

### Phase 4: Production Ready (Jour 13-26)
1. Setup CI/CD (GitHub Actions)
2. Docker containerization
3. Kubernetes orchestration
4. Monitoring & logging

---

## 📚 Documentation

Full documentation available in:
```
_REFACTORING_MICROSERVICE_/
├── 00_COMMENCER_ICI/      (Quick start)
├── 01_ARCHITECTURE/       (Technical design)
├── 02_PLANNING/           (Timeline + roadmap)
├── 03_SCRIPTS/            (Automation tools)
├── 04_GUIDES_JOUR_PAR_JOUR/(Day-by-day guides)
└── 05_REFERENCE/          (Quick lookup)
```

---

## 🎯 Success Criteria

- ✅ **Phase 1:** Structure created & environment ready
- ⏳ **Phase 2:** Code migrated, 40% tests pass
- ⏳ **Phase 3:** API endpoints working, 4+ endpoints
- ⏳ **Phase 4:** Production-ready, CI/CD configured

---

## 🚀 You're Ready!

The microservice architecture is set up. You can now:

1. **Test it:** `test_microservices.ps1`
2. **Run it:** `start_services.ps1`
3. **Develop it:** Continue with Jour 2 (code migration)
4. **Monitor it:** Use Streamlit dashboard

---

**Status:** 🟢 **Phase 1 COMPLETE - Ready for Phase 2!**

*See you in Jour 2 for code migration! 💪*
