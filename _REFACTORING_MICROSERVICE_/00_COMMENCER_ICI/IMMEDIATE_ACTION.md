# 🎯 ACTION IMMÉDIATE - À Faire Maintenant

## ⏰ 45 Minutes = Tout est Prêt!

### MAINTENANT: Exécute les 5 Étapes

---

## 1️⃣ ÉTAPE 1: Créer Structure (10 min)

### Sur Windows (PowerShell):
```powershell
# Ouvrir PowerShell (ou Windows Terminal)
# Naviguer au dossier
cd "C:\Users\u1050780\OneDrive - Sanofi\Documents\DS_COVID"

# Exécuter le script
powershell -ExecutionPolicy Bypass -File create_structure.ps1

# Vérifier (tu devrais voir la structure):
ls ml-backend\src\ds_covid_backend\
```

**Output attendu:**
```
Mode                 Name
----                 ----
d-----          api
d-----          application
d-----          config
d-----          domain
d-----          infrastructure
```

### Sur Linux/Mac:
```bash
cd ~/DS_COVID
bash create_structure.sh
tree -d ml-backend/src/
```

✅ **Quand c'est fini:** Passe à l'étape 2

---

## 2️⃣ ÉTAPE 2: Créer docs_guides/ (5 min)

C'est pour archiver les guides de refactoring (pas du vrai code):

```bash
# Windows PowerShell
cd "C:\Users\u1050780\OneDrive - Sanofi\Documents\DS_COVID"

# Créer le dossier
mkdir docs_guides

# Déplacer les guides dedans
mv START_HERE.md docs_guides/
mv RESUME_EXECUTIF.md docs_guides/
mv ARCHITECTURE_FINAL.md docs_guides/
mv PLAN_D_ACTION_DETAILLE.md docs_guides/
mv CHECKLIST_PROGRESSION.md docs_guides/
mv INDEX_DOCS.md docs_guides/
mv ARCHITECTURE_MICROSERVICES.md docs_guides/

# Vérifier
ls docs_guides/
```

**Outcome:** `docs_guides/` exist avec tous les guides

✅ **Quand c'est fini:** Passe à l'étape 3

---

## 3️⃣ ÉTAPE 3: Créer migration_backup/ (5 min)

Pour **sauvegarder localement** les vieux fichiers (pas sur Git):

```bash
# Créer la structure
mkdir -p migration_backup/{src_old,notebooks_old,pages_old}

# Copier les vieux fichiers (pas déplacer!)
cp -r src migration_backup/src_old/
cp -r notebooks migration_backup/notebooks_old/
cp -r page migration_backup/pages_old/
```

**Outcome:** 
```
migration_backup/
├─ src_old/          (copie de src/)
├─ notebooks_old/    (copie de notebooks/)
└─ pages_old/        (copie de page/)
```

⚠️ **Important:** Ne supprime pas les vieux fichiers! On les utilise encore après.

✅ **Quand c'est fini:** Passe à l'étape 4

---

## 4️⃣ ÉTAPE 4: Setup venv & Dépendances (15-20 min)

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

# Attendre... (5-10 minutes selon la connexion)
# ⏳ Ça télécharge FastAPI, TensorFlow, etc.

# Vérifier installation
python -c "import fastapi; print('✓ FastAPI installed')"
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__}')"
```

**Outcome:**
```
✓ venv créé
✓ Dépendances installées
✓ FastAPI OK
✓ TensorFlow OK
```

✅ **Quand c'est fini:** Passe à l'étape 5

---

## 5️⃣ ÉTAPE 5: Git Commit Initial (5 min)

```bash
# Au root du repo DS_COVID
cd ..  # Revenir au root

# Ajouter tout
git add .

# Vérifier (déj assez pour voir ce qui se passe)
git status

# Commit
git commit -m "refactor: initialize microservice architecture with src/

- Created ml-backend/src/ds_covid_backend/ with DDD pattern
  - api/: HTTP routes and schemas
  - domain/: Business entities
  - application/: Use cases and services
  - infrastructure/: TensorFlow, storage, logging impl
- Created tests/ structure (unit, integration, fixtures)
- Created migration_backup/ for code reference (local only)
- Created docs_guides/ for methodology (local only)
- Added .gitignore, .env.example, pyproject.toml, requirements.txt
- Created automation scripts: create_structure.sh, create_structure.ps1

Refs: ARCHITECTURE_FINAL.md, JOUR_1_STRUCTURE_CREATION.md
"

# Vérifier
git log --oneline -1
```

✅ **Quand c'est fini:** Tu as fini l'étape 1!

---

## ✅ Checklist - Dis-moi Quand C'est Fait

```
Étape 1: Créer Structure
  [ ] Run create_structure.ps1 (ou .sh)
  [ ] Vérifier ml-backend/src/ds_covid_backend/ existe
  
Étape 2: Archiver Docs
  [ ] Créer docs_guides/
  [ ] Déplacer tous les guides dedans
  [ ] Vérifier ls docs_guides/
  
Étape 3: Créer Backup Local
  [ ] Créer migration_backup/
  [ ] Copier src/, notebooks/, page/
  [ ] Vérifier migration_backup/src_old/ existe
  
Étape 4: Setup venv
  [ ] Créer venv
  [ ] Activer venv
  [ ] pip install -r requirements.txt
  [ ] Vérifier imports (fastapi, tensorflow)
  
Étape 5: Git Commit
  [ ] git add .
  [ ] git commit -m "..."
  [ ] git log --oneline (vérifier)
  
TOUT FINI?
  [ ] OUI - Passe au Jour 2!
```

---

## 🎯 Après Étape 5 - Jour 2 Commence

Une fois que les 5 étapes sont **FAITES**, tu es **prêt pour Jour 2**:

### Jour 2: Migrer Code Existant

1. Lire: `JOUR_1_STRUCTURE_CREATION.md` → Section "Phase 2"
2. Examiner: `migration_backup/src_old/`
3. Migrer code utile → `ml-backend/src/ds_covid_backend/`
4. Adapter les imports
5. Créer tests basiques

---

## 📊 État Après Jour 1

```
✓ Structure créée
✓ venv prêt
✓ Git versionnage commencé
✓ Migration_backup en place (local, safe)
✓ Docs_guides archivé (gitignored)

→ Prêt pour Jour 2: Migration Code ✅
```

---

## 🆘 Si Quelque Chose Ne Marche Pas?

### Problème: PowerShell dit "Script file not found"
```powershell
# Vérifier que t'es au bon dossier
cd "C:\Users\u1050780\OneDrive - Sanofi\Documents\DS_COVID"

# Vérifier que create_structure.ps1 existe
ls create_structure.ps1

# Si pas trouvé, fais-le manuellement (Option C dans JOUR_1_STRUCTURE_CREATION.md)
```

### Problème: pip install échoue
```bash
# Essayer avec --upgrade
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Ou si problème de version Python
python --version  # Doit être 3.11 ou plus
```

### Problème: "venv not found" après activation
```bash
# Windows: Toi dans PowerShell? Sinon:
Set-ExecutionPolicy -ExecutionPolicy bypass -Scope CurrentUser

# Réessayer activation
venv\Scripts\activate
```

### Si vraiment bloqué:
1. Poste le message d'erreur
2. On va debug ensemble

---

## ✨ Tips & Tricks

### Commande Utile: Vérifier Structure
```bash
# Windows - Voir tous les dossiers
tree /F ml-backend\src\

# Linux/Mac
tree ml-backend/src/
# Si tree pas installé:
find ml-backend/src -type d
```

### Commande Utile: Vérifier Imports
```bash
# Dans le venv activé
python -c "from src.ds_covid_backend.api import routes"  # Doit pas crasher
```

### Commande Utile: Voir ce que Git va commiter
```bash
git diff --cached  # Show diffs avant commit
git status         # Show files to commit
```

---

## 🚀 MAINTENANT C'EST PARTI!

```
1. Exécute create_structure.ps1 (ou .sh)
2. Crée docs_guides/ et déplace guides
3. Crée migration_backup/ avec copies
4. pip install -r requirements.txt
5. git commit

Quand t'as fini → Dis-moi! On passe au Jour 2 🎉
```

**Bonne chance! 💪**

---

**Reminder:** T'es en train de refactoriser un projet ML en microservice. C'est une grosse refonte mais tu as tout le plan. Patte step by step. Tu vas le faire! 🚀
