# 🤖 Scripts - Automation Tools

This folder contains the automation scripts for quick setup.

## 📄 Files in This Folder

- **create_structure.ps1** - PowerShell script (Windows):
  - Creates ml-backend folder structure automatically
  - Generates pyproject.toml, setup.py, requirements.txt
  - Creates sample module files
  - **Usage:** `powershell -ExecutionPolicy Bypass -File create_structure.ps1`

- **create_structure.sh** - Bash script (Linux/Mac):
  - Same functionality as PowerShell version
  - **Usage:** `bash create_structure.sh`

## ⚡ Quick Start with Scripts

### On Windows (PowerShell):
```powershell
cd C:\Users\{username}\Documents\DS_COVID
powershell -ExecutionPolicy Bypass -File _REFACTORING_MICROSERVICE_\03_SCRIPTS\create_structure.ps1
```

### On Linux/Mac (Bash):
```bash
cd ~/path/to/DS_COVID
bash _REFACTORING_MICROSERVICE_/03_SCRIPTS/create_structure.sh
```

## ✨ What the Scripts Do

1. **Creates folder structure:**
   ```
   ml-backend/
   ├── src/ds_covid_backend/
   │   ├── api/
   │   ├── domain/
   │   ├── application/
   │   ├── infrastructure/
   │   └── config/
   ├── tests/
   └── ...
   ```

2. **Generates configuration files:**
   - pyproject.toml (Python package metadata)
   - setup.py (installation script)
   - requirements.txt (dependencies)
   - .env.example (config template)

3. **Creates sample modules:**
   - __init__.py files in all packages
   - Sample data loader
   - Sample model factory
   - Sample API route

4. **Sets up path for imports:**
   - Uses src/ layout for proper Python packaging
   - Allows: `from ds_covid_backend.domain import ...`

## 📝 Next Steps After Running Script

1. Create Python virtual environment:
   ```powershell
   cd ml-backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # On Windows
   ```

2. Install dependencies:
   ```powershell
   pip install -e .
   ```

3. Verify installation:
   ```powershell
   pytest
   ```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"execution policy" error** | Use `-ExecutionPolicy Bypass` flag |
| **Folder already exists** | Script checks and skips existing folders |
| **Permission denied (Linux)** | Run `chmod +x create_structure.sh` first |

## 🔗 Cross-References

- **Want to understand the structure?** → See `01_ARCHITECTURE/`
- **Want step-by-step guidance?** → See `04_GUIDES_JOUR_PAR_JOUR/JOUR_1_STRUCTURE_CREATION.md`
- **Need quick reference?** → See `05_REFERENCE/`
