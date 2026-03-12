# 📅 Planning - Roadmap & Timeline

This folder contains the detailed project plan and timeline.

## 📄 Files in This Folder

- **RESUME_EXECUTIF.md** - Executive summary:
  - Project overview (30-second pitch)
  - Timeline (26 days in 4 phases)
  - Success metrics & KPIs
  - Critical path dependencies

- **PLAN_D_ACTION_DETAILLE.md** - Week-by-week breakdown:
  - Phase 1: Structure & Environment (Days 1-5)
  - Phase 2: ML Pipeline & Tests (Days 6-12)
  - Phase 3: FastAPI & Inference (Days 13-18)
  - Phase 4: CI/CD & Deployment (Days 19-26)
  - Code examples for each phase

- **CHECKLIST_PROGRESSION.md** - Task checklist:
  - 100+ actionable items
  - Validation criteria for each item
  - Progress tracking by phase
  - Definition of done (DoD)

## 📊 Timeline at a Glance

| Phase | Duration | Main Goals |
|-------|----------|-----------|
| **1** | Days 1-5 | Create ml-backend folder structure, setup venv, install dependencies |
| **2** | Days 6-12 | Build data pipeline, ML models, unit tests (40% coverage) |
| **3** | Days 13-18 | Create FastAPI endpoints, inference API, integration tests |
| **4** | Days 19-26 | Setup CI/CD, Docker containers, documentation, deployment |

## 🎯 Success Metrics

By end of Phase 4, you should have:
- ✅ 40% test coverage minimum
- ✅ All code in src/ds_covid_backend/ (properly importable)
- ✅ FastAPI with at least 3 endpoints
- ✅ GitHub Actions CI/CD pipeline passing
- ✅ Docker image builds and runs
- ✅ Documentation complete

## 🚀 How to Use

1. **Planning sprint?** Read RESUME_EXECUTIF.md (executive overview)
2. **Daily stand-up?** Check CHECKLIST_PROGRESSION.md for current day's tasks
3. **Detailed guidance?** Reference PLAN_D_ACTION_DETAILLE.md
4. **Stuck on something?** Look at the validation criteria in the checklist

## 🔗 Cross-References

- **Want day-by-day details?** → See `04_GUIDES_JOUR_PAR_JOUR/`
- **Want architecture context?** → See `01_ARCHITECTURE/`
- **Ready to execute?** → Use `03_SCRIPTS/create_structure.ps1`
