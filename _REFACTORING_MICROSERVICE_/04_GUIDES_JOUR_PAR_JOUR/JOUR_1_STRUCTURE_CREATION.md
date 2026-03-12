# 🎯 PLAN COMPLET - Création Structure + Nettoyage

## Situation Actuelle

```
DS_COVID/
├─ RESUME_EXECUTIF.md
├─ ARCHITECTURE_MICROSERVICES.md
├─ PLAN_D_ACTION_DETAILLE.md
├─ CHECKLIST_PROGRESSION.md
├─ INDEX_DOCS.md
├─ START_HERE.md
├─ ARCHITECTURE_FINAL.md ← NOUVELLE (avec src/)
├─ create_structure.sh ← Script bash
├─ create_structure.ps1 ← Script PowerShell
│
├─ src/                  ← À MIGRER
├─ notebooks/            ← À MIGRER
├─ page/                 ← À MIGRER
└─ ... [autres]
```

---

## ✅ ÉTAPE 1: Créer Structure (Aujourd'hui - 15 min)

### Option A: Windows (PowerShell)

```powershell
# 1. Ouvrir PowerShell en admin
# 2. Naviguer au repo root DS_COVID
cd "c:\Users\u1050780\OneDrive - Sanofi\Documents\DS_COVID"

# 3. Exécuter le script de création
powershell -ExecutionPolicy Bypass -File create_structure.ps1

# 4. Vérifier
ls ml-backend/src/ds_covid_backend/
```

### Option B: Linux/Mac (Bash)

```bash
# 1. Naviguer au repo root
cd ~/DS_COVID

# 2. Exécuter le script
bash create_structure.sh

# 3. Vérifier
tree -d -L 3 ml-backend/src/
```

### Option C: Manuel (Si scripts ne marchent pas)

```bash
# Créer directories manuellement
mkdir -p ml-backend/src/ds_covid_backend/{api,domain,application,infrastructure,config}
mkdir -p ml-backend/src/ds_covid_backend/api/{routes,schemas,errors}
mkdir -p ml-backend/tests/{unit,integration,fixtures}
mkdir -p ml-backend/notebooks
mkdir -p ml-backend/{logs,data,models}
mkdir -p migration_backup

# Créer __init__.py
touch ml-backend/src/ds_covid_backend/__init__.py
touch ml-backend/src/ds_covid_backend/api/__init__.py
# ... etc pour chaque dir
```

**Outcome:** 
```
✓ ml-backend/src/ds_covid_backend/ exists
✓ All __init__.py created
✓ .gitignore created
✓ .env.example created
✓ pyproject.toml created
```

---

## ✅ ÉTAPE 2: Archiver Documents (Aujourd'hui - 10 min)

Les documents **de guide** doivent être archivés pour ne pas polluer le repo:

### Documents à GARDER (versionner sur Git):
```
✓ docs/SPECIFICATION.md         (à créer)
✓ docs/SETUP.md                 (à créer)
✓ docs/API.md                   (à créer)
✓ docs/DEPLOYMENT.md            (à créer)
✓ ml-backend/README.md
✓ frontend/README.md
✓ infrastructure/*.yml
```

### Documents à ARCHIVER (temporaires, méthodologie):
```
À CRÉER dans dossier "docs_guides/":
├── START_HERE.md
├── RESUME_EXECUTIF.md
├── ARCHITECTURE_FINAL.md
├── PLAN_D_ACTION_DETAILLE.md
├── CHECKLIST_PROGRESSION.md
├── INDEX_DOCS.md
└── .gitkeep
```

**Commandes:**

```bash
# Créer dossier d'archive
mkdir -p docs_guides

# Archiver les guides (les déplacer, pas copier)
mv START_HERE.md docs_guides/
mv RESUME_EXECUTIF.md docs_guides/
mv ARCHITECTURE_FINAL.md docs_guides/
mv PLAN_D_ACTION_DETAILLE.md docs_guides/
mv CHECKLIST_PROGRESSION.md docs_guides/
mv INDEX_DOCS.md docs_guides/
mv ARCHITECTURE_MICROSERVICES.md docs_guides/

# Mettre en .gitignore
echo "docs_guides/" >> .gitignore

# Vérifier
ls docs_guides/
```

**Outcome:**
```
✓ docs_guides/ créé
✓ Tous les guides archivés
✓ .gitignore updated
```

---

## ✅ ÉTAPE 3: Créer 'migration_backup/' (Aujourd'hui - 5 min)

C'est le dossier pour **tout l'ancien code** (on va l'utiliser comme référence):

```bash
# Créer la structure de migration
mkdir -p migration_backup/{src_old,notebooks_old,pages_old,other_old}

# COPIER (pas déplacer) les anciens fichiers comme référence
cp -r src migration_backup/src_old/
cp -r notebooks migration_backup/notebooks_old/
cp -r page migration_backup/pages_old/

# Ajouter à .gitignore
echo "migration_backup/" >> ml-backend/.gitignore

# Vérifier
ls migration_backup/
```

**Important:** C'est en `.gitignore`, donc ça ne sera **pas versionné sur Git**. C'est juste pour vous, localement, comme backup au cas où.

**Outcome:**
```
✓ migration_backup/ créé
✓ Anciens fichiers copiés (sauvegarde locale)
✓ Ajouté à .gitignore
```

---

## ✅ ÉTAPE 4: Créer venv et Installer Dépendances (Aujourd'hui - 5-10 min)

```bash
# Naviguer au backend
cd ml-backend

# Créer venv
python -m venv venv

# Activer venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt

# Vérifier
python -c "import fastapi; print('✓ FastAPI OK')"
python -c "import tensorflow as tf; print('✓ TensorFlow OK')"
```

**Outcome:**
```
✓ venv créé et activé
✓ Dépendances installées
✓ Imports testing
```

---

## ✅ ÉTAPE 5: Git Initial Commit (Aujourd'hui - 5 min)

```bash
# Au root du projet
git add .

# Réviser ce qui est commité
git status

# Commit
git commit -m "feat: initial microservice architecture with src/ structure

- Created ml-backend/src/ds_covid_backend/ with DDD layers
  - api/: HTTP routes
  - domain/: Business entities
  - application/: Use cases
  - infrastructure/: TensorFlow, storage, logging
- Created tests/ structure (unit, integration, fixtures)
- Created migration_backup/ for old code (gitignored)
- Created docs_guides/ for methodology docs (gitignored)
- Added .gitignore, .env.example, pyproject.toml, requirements.txt
- Added create_structure.sh and create_structure.ps1 for automation

Refs: ARCHITECTURE_FINAL.md
"

# Vérifier
git log --oneline -1
```

**Outcome:**
```
✓ Initial commit fait
✓ Structure versionée sur Git
✓ migration_backup/ et docs_guides/ ignorés
```

---

## ☑️ Checklist - Jour 1 (Avant de continuer)

- [ ] **15 min:** Run create_structure.sh (ou .ps1)
- [ ] **10 min:** Créer docs_guides/ et archiver guides
- [ ] **5 min:** Créer migration_backup/ avec copies
- [ ] **10 min:** Créer venv et installer dépendances
- [ ] **5 min:** Git initial commit
- [ ] **Vérifier:** `ls ml-backend/src/ds_covid_backend/`
- [ ] **Vérifier:** `python -c "import fastapi"`

**Temps total:** ~45 minutes

---

## 📊 État Après Jour 1

```
DS_COVID/ (Git repo)
├─ .gitignore              ✓ updated
├─ ml-backend/
│  ├─ src/
│  │  └─ ds_covid_backend/ ✓ Structure DDD creates
│  ├─ tests/               ✓ ready
│  ├─ notebooks/           ✓ empty
│  ├─ venv/                ✓ activated
│  ├─ .env.example         ✓
│  ├─ pyproject.toml       ✓
│  ├─ requirements.txt     ✓
│  └─ README.md            ✓
│
├─ frontend/               (same as before)
├─ infrastructure/         (same as before)
│
├─ migration_backup/       ✓ gitignored (backup local)
│  ├─ src_old/
│  ├─ notebooks_old/
│  └─ pages_old/
│
├─ docs_guides/            ✓ gitignored (méthodologie)
│  ├─ START_HERE.md
│  ├─ RESUME_EXECUTIF.md
│  ├─ ARCHITECTURE_FINAL.md
│  ├─ PLAN_D_ACTION_DETAILLE.md
│  ├─ CHECKLIST_PROGRESSION.md
│  └─ INDEX_DOCS.md
│
└─ docs/                   (À CRÉER: SPECIFICATION.md, etc)
```

---

## 🚀 Prochaines Phases

### Phase 2 (Jour 2-3): Migrer Code Existant
1. Examiner migration_backup/src_old/
2. Copier code utile dans src/ds_covid_backend/
3. Adapter imports
4. Tests basiques

### Phase 3 (Jour 4-7): Construire Layers
1. Remplir `application/` (data_processor, predict_service)
2. Remplir `infrastructure/` (model_loader)
3. Remplir `api/` (routes, schemas)
4. Créer unit tests

### Phase 4 (Jour 8-14): FastAPI & Tests
1. Créer `src/ds_covid_backend/main.py` (FastAPI app)
2. Endpoints `/predict`, `/health`
3. Integration tests
4. Documentation API

### Phase 5 (Jour 15+): Deployment
1. Dockerfile
2. docker-compose.yml
3. GitHub Actions
4. Kubernetes (optionnel)

---

## 📚 Documentation Références

**Pour comprendre le pourquoi:**
- `docs_guides/ARCHITECTURE_FINAL.md` - Why `src/`?
- `docs_guides/RESUME_EXECUTIF.md` - Timeline global

**Pour savoir quoi faire ensuite:**
- `docs_guides/PLAN_D_ACTION_DETAILLE.md` - Phase par phase
- `docs_guides/CHECKLIST_PROGRESSION.md` - Tracking

**Pour visionner le code:**
- `ml-backend/README.md` - Quick start

---

## 💡 Pourquoi Cette Approche?

| Aspect | Avant | Après |
|--------|-------|-------|
| **Structure** | 😫 Chaos | ✅ DDD-clean |
| **`src/`** | ❌ Non | ✅ Python best-practice |
| **Imports** | ❌ Confus | ✅ `from src.ds_covid_backend.api import...` |
| **Docs** | 📚 Pollue repo | 📂 `docs_guides/` (gitignored) |
| **Backup** | ❌ Risqué | ✅ `migration_backup/` (local, safe) |
| **Scalability** | ❌ Non | ✅ Microservice-ready |

---

## ⚠️ Rappels Importants

1. **migration_backup/ est LOCAL ONLY**
   - Ne commit pas ✗
   - C'est juste pour vous (backup)
   - Supprimez-le quand vous n'en avez plus besoin

2. **docs_guides/ est DOCUMENTATION MÉTHODOLOGIE**
   - Ne commit pas ✗
   - C'est pour le refactoring (pas pour la distribution)
   - Supprimez-le une fois que c'est fini (ou gardez pour historique)

3. **docs/ est DOCUMENTATION PROJET**
   - Commit ✓
   - C'est pour les utilisateurs/développeurs
   - Examples: API.md, SETUP.md, etc.

4. **Activation venv CHAQUE FOIS**
   ```bash
   # Windows
   ml-backend\venv\Scripts\activate
   
   # Linux/Mac
   source ml-backend/venv/bin/activate
   ```

---

## 🎬 Start!

```bash
# Étape 1: Créer structure
powershell -ExecutionPolicy Bypass -File create_structure.ps1  # Windows
# ou
bash create_structure.sh  # Linux/Mac

# Étape 2-5: Suivre checklist ci-dessus

# Vérifier
ls -la ml-backend/src/ds_covid_backend/

# Go!
cd ml-backend
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate
pip install -r requirements.txt
```

**Status:** Ready to build! 🚀
