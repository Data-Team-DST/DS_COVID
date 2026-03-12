# 🦠 DS_COVID - ML Microservice Refactoring

Production-ready microservice architecture for COVID-19 ML prediction model.

## 🚀 Quick Start

```bash
# Activate environment
cd ml-backend
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run API
python app.py

# Run tests
pytest
```

## 📁 Project Structure

```
DS_COVID/
├── ml-backend/                    ← Production-ready backend
│   ├── src/ds_covid_backend/      ← DDD Architecture
│   │   ├── api/                   (FastAPI routes)
│   │   ├── domain/                (Business logic)
│   │   ├── application/           (Use cases)
│   │   ├── infrastructure/        (Data access)
│   │   └── config/                (Configuration)
│   ├── tests/                     (Unit & integration tests)
│   ├── venv/                      (Virtual environment)
│   └── app.py                     (FastAPI app)
│
├── _REFACTORING_MICROSERVICE_/    ← Methodology & guides
│   ├── 00_COMMENCER_ICI/          (Quick start)
│   ├── 01_ARCHITECTURE/           (Technical design)
│   ├── 02_PLANNING/               (Timeline & roadmap)
│   ├── 03_SCRIPTS/                (Automation tools)
│   ├── 04_GUIDES_JOUR_PAR_JOUR/   (Day-by-day instructions)
│   └── 05_REFERENCE/              (Quick lookup)
│
├── migration_backup/              ← Local backup (not committed)
├── docs_guides/                   ← Documentation index
├── _OLD_ROOT_FILES/               ← Archived old files
│
└── src/, notebooks/, page/        ← Original code (to migrate)
```

## 🎯 Architecture: Domain-Driven Design (DDD)

```
API Layer (FastAPI)
    ↓
Application Layer (Use Cases)
    ↓
Domain Layer (Business Logic)
    ↓
Infrastructure Layer (TensorFlow, Storage)
```

## 📚 Documentation

**Begin here:** [`_REFACTORING_MICROSERVICE_/README.md`](_REFACTORING_MICROSERVICE_/README.md)

Quick guides:
- **Immediate action:** `_REFACTORING_MICROSERVICE_/00_COMMENCER_ICI/IMMEDIATE_ACTION.md`
- **Architecture:** `_REFACTORING_MICROSERVICE_/01_ARCHITECTURE/`
- **Day-by-day guide:** `_REFACTORING_MICROSERVICE_/04_GUIDES_JOUR_PAR_JOUR/JOUR_1_STRUCTURE_CREATION.md`
- **Timeline:** `_REFACTORING_MICROSERVICE_/02_PLANNING/`

## 🔄 Refactoring Phases

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** | Days 1-5 | ml-backend structure + environment |
| **Phase 2** | Days 6-12 | ML pipeline + unit tests (40%) |
| **Phase 3** | Days 13-18 | FastAPI endpoints + inference API |
| **Phase 4** | Days 19-26 | CI/CD + Docker + Deployment |

**Current Status:** ✅ Phase 1 Complete (Jour 1)

## 🛠️ Tech Stack

- **Framework:** FastAPI 0.104.1
- **ML:** TensorFlow 2.x (TBD)
- **Testing:** pytest 9.0.2
- **Data:** pandas, numpy, scikit-learn
- **Deployment:** Docker, GitHub Actions

## 📖 For Different Roles

- **👨‍💼 Product Manager** → Read `_REFACTORING_MICROSERVICE_/02_PLANNING/RESUME_EXECUTIF.md`
- **👨‍💻 Developer** → Go to `ml-backend/` + `_REFACTORING_MICROSERVICE_/01_ARCHITECTURE/`
- **🧪 QA/Tester** → Check `_REFACTORING_MICROSERVICE_/02_PLANNING/CHECKLIST_PROGRESSION.md`
- **🚀 DevOps** → See `ml-backend/.gitignore` + Docker guides (coming soon)

## 🎓 Learning Path

1. Read: `_REFACTORING_MICROSERVICE_/00_COMMENCER_ICI/README.md`
2. Understand: `_REFACTORING_MICROSERVICE_/01_ARCHITECTURE/ARCHITECTURE_FINAL.md`
3. Plan: `_REFACTORING_MICROSERVICE_/02_PLANNING/`
4. Execute: `_REFACTORING_MICROSERVICE_/04_GUIDES_JOUR_PAR_JOUR/`

## 📦 Dependencies

See `ml-backend/requirements.txt` for Python packages.

For old project context, see `_OLD_ROOT_FILES/` for archived documentation.

## ✅ Checklist

Progress tracking: `_REFACTORING_MICROSERVICE_/02_PLANNING/CHECKLIST_PROGRESSION.md`

**Current:** 
- ✅ Phase 1: Structure created
- ⏳ Phase 2: Starting code migration
- ⏳ Phase 3: API endpoints
- ⏳ Phase 4: Production deployment

## 🤝 Contributing

1. Activate venv: `cd ml-backend && source venv/bin/activate`
2. Code follows DDD pattern (see architecture guide)
3. All code requires tests (pytest)
4. Push to branch matching phase

## 📞 Support

Check documentation first: `_REFACTORING_MICROSERVICE_/05_REFERENCE/INDEX_DOCS.md`

---

**Last Updated:** Phase 1 Complete  
**Status:** 🟢 Ready for Phase 2  
**Next:** Code Migration (Jour 2)
