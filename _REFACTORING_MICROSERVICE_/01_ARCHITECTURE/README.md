# 🏛️ Architecture - Technical Foundation

This folder contains the complete technical architecture for the refactored ML backend.

## 📄 Files in This Folder

- **ARCHITECTURE_FINAL.md** - Complete DDD-based architecture explaining:
  - Why we chose src/ folder structure
  - Domain-Driven Design layers (API, Domain, Application, Infrastructure)
  - Package structure and import patterns
  - Design decisions and trade-offs

- **ARCHITECTURE_MICROSERVICES.md** - High-level microservice design:
  - Service boundaries
  - API contracts
  - Data flow between services
  - Deployment topology

## 🎯 Key Architecture Decisions

### Pattern: Domain-Driven Design (DDD)
```
ml-backend/
├── src/ds_covid_backend/
│   ├── api/              # FastAPI endpoints
│   ├── domain/           # Business logic & models
│   ├── application/      # Use cases
│   ├── infrastructure/   # Data access, external integrations
│   └── config/           # Configuration management
├── tests/                # Unit & integration tests
└── pyproject.toml        # Dependencies
```

### Why This Structure?
- ✅ **Testable** - Easy to mock dependencies
- ✅ **Scalable** - Add new domains without breaking existing code
- ✅ **Maintainable** - Clear separation of concerns
- ✅ **Production-Ready** - Industry standard pattern

## 🔍 How to Use

1. **First time?** Read `ARCHITECTURE_FINAL.md` end-to-end (25 min read)
2. **Need quick overview?** Read ARCHITECTURE_MICROSERVICES.md (10 min)
3. **Building a new feature?** Reference the layer structure from ARCHITECTURE_FINAL.md

## 🤝 Cross-References

- **Want the step-by-step?** → See `04_GUIDES_JOUR_PAR_JOUR/`
- **Want the roadmap?** → See `02_PLANNING/`
- **Want to implement now?** → Use `03_SCRIPTS/create_structure.ps1`
