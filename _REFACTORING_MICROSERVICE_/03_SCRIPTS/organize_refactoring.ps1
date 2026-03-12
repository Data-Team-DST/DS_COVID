# Script: Organize Refactoring Microservice Files
# Usage: powershell -ExecutionPolicy Bypass -File organize_refactoring.ps1

Write-Host "🚀 Reorganizing Refactoring Files..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

$rootPath = Get-Location
Write-Host "📍 Working in: $rootPath" -ForegroundColor Cyan

# 1. Create main directory structure
Write-Host ""
Write-Host "📁 Creating main directory..." -ForegroundColor Cyan
$mainDir = "_REFACTORING_MICROSERVICE_"
if (-not (Test-Path $mainDir)) {
    New-Item -ItemType Directory -Path $mainDir | Out-Null
    Write-Host "✓ Created $mainDir" -ForegroundColor Green
}

# 2. Create subdirectories
Write-Host "📂 Creating subdirectories..." -ForegroundColor Cyan
$subDirs = @(
    "00_COMMENCER_ICI",
    "01_ARCHITECTURE",
    "02_PLANNING",
    "03_SCRIPTS",
    "04_GUIDES_JOUR_PAR_JOUR",
    "05_REFERENCE"
)

foreach ($dir in $subDirs) {
    $path = Join-Path $mainDir $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        New-Item -ItemType File -Path (Join-Path $path ".gitkeep") | Out-Null
        Write-Host "✓ Created $dir/" -ForegroundColor Green
    }
}

# 3. Move files to appropriate directories
Write-Host ""
Write-Host "📋 Moving files..." -ForegroundColor Cyan

$fileMoves = @{
    "IMMEDIATE_ACTION.md" = "00_COMMENCER_ICI"
    "ARCHITECTURE_FINAL.md" = "01_ARCHITECTURE"
    "ARCHITECTURE_MICROSERVICES.md" = "01_ARCHITECTURE"
    "RESUME_EXECUTIF.md" = "02_PLANNING"
    "PLAN_D_ACTION_DETAILLE.md" = "02_PLANNING"
    "CHECKLIST_PROGRESSION.md" = "02_PLANNING"
    "create_structure.sh" = "03_SCRIPTS"
    "create_structure.ps1" = "03_SCRIPTS"
    "JOUR_1_STRUCTURE_CREATION.md" = "04_GUIDES_JOUR_PAR_JOUR"
    "INDEX_DOCS.md" = "05_REFERENCE"
    "START_HERE.md" = "05_REFERENCE"
}

foreach ($file in $fileMoves.Keys) {
    $sourcePath = $file
    $destDir = $fileMoves[$file]
    $destPath = Join-Path $mainDir $destDir $file
    
    if ((Test-Path $sourcePath) -and -not (Test-Path $destPath)) {
        Move-Item -Path $sourcePath -Destination $destPath -Force
        Write-Host "✓ Moved $file → $destDir/" -ForegroundColor Green
    } elseif (Test-Path $destPath) {
        Write-Host "⚠ $file already in $destDir/" -ForegroundColor Yellow
    }
}

# 4. Create README files for each subdirectory
Write-Host ""
Write-Host "📝 Creating README files..." -ForegroundColor Cyan

# 00_COMMENCER_ICI/README.md
$readme00 = @"
# 👋 Commencer Ici

Bienvenue dans le refactoring microservice de DS_COVID!

## 🎯 Par Où Commencer?

1. **Lire ce fichier** (5 min) ← Vous êtes ici
2. **Lire IMMEDIATE_ACTION.md** (45 min) - 5 étapes pour setup
3. **Explorer autres dossiers** - Guides détaillés

## 📚 Organisation des Guides

\`\`\`
_REFACTORING_MICROSERVICE_/
├── 00_COMMENCER_ICI/     ← Entry point (ce dossier!)
├── 01_ARCHITECTURE/      ← Comprendre la structure
├── 02_PLANNING/          ← Timeline et métriques
├── 03_SCRIPTS/           ← Scripts automatisation
├── 04_GUIDES_JOUR/       ← Détails jour par jour
└── 05_REFERENCE/         ← Lookup rapide
\`\`\`

## ⏱️ Timeline

- **Jour 1:** 45 minutes (IMMEDIATE_ACTION.md)
- **Jour 2-3:** Code migration
- **Jour 4-6:** API et tests
- **Jour 7+:** Deployment

## 👥 Pour Mon Rôle

- **Tech Lead:** Lire ARCHITECTURE/ puis PLANNING/RESUME_EXECUTIF.md
- **ML Engineer:** Lire GUIDES_JOUR_PAR_JOUR/JOUR_1
- **DevOps:** Lire SCRIPTS/ then GUIDES/
- **Product Manager:** Lire PLANNING/RESUME_EXECUTIF.md

## 🚀 Prochaine Étape

Lire **IMMEDIATE_ACTION.md** maintenant pour les 5 étapes (45 min)!
"@

Set-Content -Path (Join-Path $mainDir "00_COMMENCER_ICI" "README.md") -Value $readme00
Write-Host "✓ Created 00_COMMENCER_ICI/README.md" -ForegroundColor Green

# 03_SCRIPTS/README.md
$readme03 = @"
# Scripts Automatisation

Scripts pour créer la structure microservice automatiquement.

## Quelle Script Utiliser?

### Windows PowerShell
\`\`\`powershell
cd 03_SCRIPTS
powershell -ExecutionPolicy Bypass -File create_structure.ps1
\`\`\`

### Linux/Mac Bash
\`\`\`bash
cd 03_SCRIPTS
bash create_structure.sh
\`\`\`

## Que Font les Scripts?

✓ Créent \`ml-backend/src/ds_covid_backend/\` avec structure DDD
✓ Créent dossiers tests/
✓ Créent configuration files (.gitignore, .env.example, etc)
✓ Créent \`__init__.py\` everywhere

**Durée:** 2-3 minutes

Voir JOUR_1_STRUCTURE_CREATION.md pour détails complets.
"@

Set-Content -Path (Join-Path $mainDir "03_SCRIPTS" "README.md") -Value $readme03
Write-Host "✓ Created 03_SCRIPTS/README.md" -ForegroundColor Green

# 05_REFERENCE/README.md
$readme05 = @"
# Référence Rapide

## Cherche une Info?

| Question | Document |
|----------|----------|
| "Par où commencer?" | 00_COMMENCER_ICI/README.md |
| "En 45 min?" | 00_COMMENCER_ICI/IMMEDIATE_ACTION.md |
| "Comprendre la structure?" | ../01_ARCHITECTURE/ARCHITECTURE_FINAL.md |
| "Timeline et budget?" | ../02_PLANNING/RESUME_EXECUTIF.md |
| "Phase par phase?" | ../02_PLANNING/PLAN_D_ACTION_DETAILLE.md |
| "Tracker progrès?" | ../02_PLANNING/CHECKLIST_PROGRESSION.md |
| "Jour 1 détaillé?" | ../04_GUIDES_JOUR_PAR_JOUR/JOUR_1_STRUCTURE_CREATION.md |
| "Automatiser structure?" | ../03_SCRIPTS/README.md |

## Lecteurs Recommandés

**Tout le monde:** 
→ 00_COMMENCER_ICI/

**Tech Lead/Architect:** 
→ ../01_ARCHITECTURE/ 
→ ../02_PLANNING/RESUME_EXECUTIF.md

**Développeurs ML:** 
→ ../04_GUIDES_JOUR_PAR_JOUR/JOUR_1_STRUCTURE_CREATION.md

**DevOps/Infra:** 
→ ../03_SCRIPTS/ 
→ ../04_GUIDES_JOUR_PAR_JOUR/

Pour plus de détails: See INDEX_DOCS.md et START_HERE.md
"@

Set-Content -Path (Join-Path $mainDir "05_REFERENCE" "README.md") -Value $readme05
Write-Host "✓ Created 05_REFERENCE/README.md" -ForegroundColor Green

# 5. Create main README
$readmeMain = @"
# 📦 Refactoring Microservice - DS_COVID

**Tous les guides pour refactoriser en architecture microservice!**

## 🎯 Commencer?

👉 **Commence par:** [\`00_COMMENCER_ICI/README.md\`](00_COMMENCER_ICI/README.md)

## 📁 Structure

\`\`\`
_REFACTORING_MICROSERVICE_/
├── 00_COMMENCER_ICI/        ← Par ici! (entry point)
├── 01_ARCHITECTURE/         ← Comprendre la structure DDD + src/
├── 02_PLANNING/             ← Timeline, budget, métriques
├── 03_SCRIPTS/              ← Scripts d'automatisation
├── 04_GUIDES_JOUR_PAR_JOUR/ ← Détails jour 1, 2, 3... 
└── 05_REFERENCE/            ← Index et quick lookup
\`\`\`

## ⏱️ Timeline Global

| Phase | Durée | Contenu |
|-------|-------|---------|
| 🚀 **Jour 1** | 45 min | Structure + venv + Git (IMMEDIATE_ACTION.md) |
| 📦 **Jour 2-3** | 2-3h | Migration code existant |
| 🔌 **Jour 4-5** | 2-3h | FastAPI endpoints + tests |
| 🐳 **Jour 6** | 1-2h | Docker + CI/CD |

## 🚀 Quick Start (45 min)

1. Ouvre \`00_COMMENCER_ICI/IMMEDIATE_ACTION.md\`
2. Suis les 5 étapes
3. Tu as une structure microservice prête!

## 📚 Pour Chaque Rôle

| Rôle | Lire D'Abord |
|------|--------------|
| Tech Lead | 00_COMMENCER_ICI/ → 01_ARCHITECTURE/ → 02_PLANNING/ |
| ML Engineer | 00_COMMENCER_ICI/IMMEDIATE_ACTION.md → 04_GUIDES/ |
| DevOps | 03_SCRIPTS/README.md → 04_GUIDES/ |
| Manager | 02_PLANNING/RESUME_EXECUTIF.md |

## ✨ What's Inside?

✓ Architecture microservice avec \`src/\` structure  
✓ Scripts d'automatisation (Bash + PowerShell)  
✓ Guides détaillés jour par jour  
✓ Timeline et budget estimation  
✓ Checklist de progrès  
✓ Examples de code  

## 🗑️ Après Refactoring?

Une fois que c'est fini, tu peux supprimer ce dossier:
\`\`\`bash
rm -rf _REFACTORING_MICROSERVICE_/
\`\`\`

---

**Status:** Ready to start! 🚀

Commence par: [\`00_COMMENCER_ICI/README.md\`](00_COMMENCER_ICI/README.md)
"@

Set-Content -Path (Join-Path $mainDir "README.md") -Value $readmeMain
Write-Host "✓ Created _REFACTORING_MICROSERVICE_/README.md" -ForegroundColor Green

# 6. Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Reorganization Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

Write-Host "📊 Structure créée:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  _REFACTORING_MICROSERVICE_/" -ForegroundColor Cyan
Write-Host "  ├── 00_COMMENCER_ICI/       ← Lire en premier!" -ForegroundColor Green
Write-Host "  ├── 01_ARCHITECTURE/        ← Comprendre structure" -ForegroundColor Green
Write-Host "  ├── 02_PLANNING/            ← Timeline + budget" -ForegroundColor Green
Write-Host "  ├── 03_SCRIPTS/             ← Scripts automatisation" -ForegroundColor Green
Write-Host "  ├── 04_GUIDES_JOUR_PAR_JOUR/← Détails jour 1, 2..." -ForegroundColor Green
Write-Host "  ├── 05_REFERENCE/           ← Quick lookup" -ForegroundColor Green
Write-Host "  └── README.md               ← Index global" -ForegroundColor Green
Write-Host ""

Write-Host "📂 Fichiers déplacés:" -ForegroundColor Yellow
foreach ($file in $fileMoves.Keys) {
    Write-Host "  ✓ $file → $($fileMoves[$file])/" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Prochaine étape:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Ouvre: _REFACTORING_MICROSERVICE_/00_COMMENCER_ICI/README.md" -ForegroundColor White
Write-Host "  2. Puis: _REFACTORING_MICROSERVICE_/00_COMMENCER_ICI/IMMEDIATE_ACTION.md" -ForegroundColor White
Write-Host "  3. Suis les 5 étapes (45 min)" -ForegroundColor White
Write-Host ""

Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "  • Tout le refactoring est dans _REFACTORING_MICROSERVICE_/" -ForegroundColor Gray
Write-Host "  • Facile à supprimer après: rm -rf _REFACTORING_MICROSERVICE_/" -ForegroundColor Gray
Write-Host "  • Optimal pour collègues: structure claire!" -ForegroundColor Gray
Write-Host ""
