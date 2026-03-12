# ✅ Pre-Test Validation Checklist

**Phase 1 Completion Verification**

---

## 🔍 File Structure Validation

### Root Directory (Should be Clean)
- [ ] `start_services.ps1` exists
- [ ] `start_services.sh` exists
- [ ] `test_microservices.ps1` exists
- [ ] `test_microservices.sh` exists
- [ ] `streamlit_app.py` exists
- [ ] `MICROSERVICES_README.md` exists
- [ ] `CODE_INVENTORY.md` exists
- [ ] `setup.py` properly archived (NOT in root)
- [ ] Old `src/` folder properly archived (NOT in root)

**Expected Root Files (5-10 essential files only):**
```
✓ .gitignore
✓ LICENSE
✓ MANIFEST.in
✓ README.md
✓ pyproject.toml
✓ requirements.txt
+ MICROSERVICES_README.md
+ CODE_INVENTORY.md
+ start_services.ps1
+ start_services.sh
+ test_microservices.ps1
+ test_microservices.sh
+ streamlit_app.py
```

### Backend Structure
- [ ] `ml-backend/` exists
- [ ] `ml-backend/app.py` exists
- [ ] `ml-backend/requirements.txt` exists and includes:
  - [ ] FastAPI
  - [ ] uvicorn
  - [ ] streamlit
  - [ ] requests
  - [ ] pytest
  - [ ] pandas
  - [ ] numpy
  - [ ] scikit-learn
- [ ] `ml-backend/src/ds_covid_backend/` exists
- [ ] `ml-backend/src/ds_covid_backend/api/` exists
- [ ] `ml-backend/src/ds_covid_backend/domain/` exists
- [ ] `ml-backend/src/ds_covid_backend/application/` exists
- [ ] `ml-backend/src/ds_covid_backend/infrastructure/` exists
- [ ] `ml-backend/src/ds_covid_backend/config/` exists
- [ ] `ml-backend/tests/` exists
- [ ] `ml-backend/venv/` exists (Python environment)

### Documentation
- [ ] `_REFACTORING_MICROSERVICE_/` exists with 6 subdirectories
- [ ] `migration_backup/` exists (old code backup)
- [ ] `_OLD_ROOT_FILES/` exists (archived setup.py and src/)

---

## 🔧 Environment Validation

### Python Setup
- [ ] Python 3.11+ installed
  - Test: `python --version` → should be 3.11+
  
- [ ] Virtual environment exists in `ml-backend/venv/`
  - Test (Windows): `dir ml-backend\venv\Scripts\python.exe`
  - Test (Linux/Mac): `ls ml-backend/venv/bin/python`

### Dependencies Installation
- [ ] Dependencies installed in venv
  - Test: `pip list` while venv active
  - Should see: FastAPI, uvicorn, streamlit, pytest

- [ ] Can import key packages
  - Test: `python -c "import fastapi; print(fastapi.__version__)"`

---

## 🖥️ Port Availability

### Check Ports Are Free
- [ ] Port 8000 available (Backend)
  - Windows: `netstat -ano | findstr :8000` (should return empty)
  - Linux/Mac: `lsof -i :8000` (should return empty)

- [ ] Port 8501 available (Frontend)
  - Windows: `netstat -ano | findstr :8501` (should return empty)
  - Linux/Mac: `lsof -i :8501` (should return empty)

---

## 📝 Script Validation

### Windows PowerShell Scripts
- [ ] `start_services.ps1` is readable
  - Test: `Get-Content start_services.ps1 | head -20`
  
- [ ] `test_microservices.ps1` is readable
  - Test: `Get-Content test_microservices.ps1 | head -20`

- [ ] Execution policy allows scripts
  - Current: `Get-ExecutionPolicy`
  - Should be: RemoteSigned or Bypass
  - If not: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Linux/Mac Bash Scripts
- [ ] `start_services.sh` is readable and executable
  - Test: `ls -l start_services.sh` (should have x permission)
  - Test: `bash -n start_services.sh` (should have no syntax errors)

- [ ] `test_microservices.sh` is readable
  - Test: `bash -n test_microservices.sh` (no syntax errors)

---

## 🌐 Network Validation

### DNS/Localhost Resolution
- [ ] Can reach localhost
  - Windows: `ping localhost`
  - Linux/Mac: `ping -c 1 localhost`

- [ ] Can connect to ports
  - Windows: `Test-NetConnection -ComputerName localhost -Port 8000`
  - Linux/Mac: `nc -zv localhost 8000`

---

## 📊 Code Review

### backend/app.py
- [ ] FastAPI initialization
  - Test: `python ml-backend/app.py` (should start without errors)

### streamlit_app.py
- [ ] Streamlit installation ready
  - Test: `streamlit --version`

- [ ] Can import streamlit
  - Test: `python -c "import streamlit; print('OK')"`

---

## 🚀 Pre-Flight Checklist

Before launching services:

1. **Environment Ready?**
   - [ ] Python 3.11+
   - [ ] venv activated or in ml-backend/
   - [ ] All dependencies installed

2. **Ports Clear?**
   - [ ] Port 8000 free
   - [ ] Port 8501 free

3. **Scripts Ready?**
   - [ ] start_services.ps1/sh executable
   - [ ] test_microservices.ps1/sh executable
   - [ ] No syntax errors in scripts

4. **Files in Place?**
   - [ ] streamlit_app.py exists
   - [ ] app.py exists in ml-backend/
   - [ ] requirements.txt updated

5. **Git Ready?**
   - [ ] All changes committed (or ready to commit)
   - [ ] History preserved

---

## ✅ Verification Commands

Run these to validate everything:

```bash
# 1. Check Python version
python --version
# Expected: Python 3.11.x or higher

# 2. Check dependencies
pip list | grep -E "fastapi|streamlit|pytest"
# Expected: All should be listed

# 3. Check ports are free
netstat -ano | findstr ":8000 :8501"
# Expected: Empty (no output)

# 4. Check FastAPI runs
cd ml-backend && python -c "import app; print('FastAPI OK')"
# Expected: FastAPI OK

# 5. Check Streamlit runs
python -c "import streamlit; print('Streamlit OK')"
# Expected: Streamlit OK

# 6. Validate bash scripts
bash -n start_services.sh
bash -n test_microservices.sh
# Expected: No output (means no errors)

# 7. Validate PowerShell scripts
powershell -NoProfile -Command "& {. '.\start_services.ps1' -Validate}"
powershell -NoProfile -Command "& {. '.\test_microservices.ps1' -Validate}"
# Expected: Should parse without errors
```

---

## 🎯 Expected Behavior

### When Running `start_services.ps1` or `start_services.sh`:

1. **Immediately:**
   - Backend starts and logs appear
   - Frontend starts and logs appear
   - Both run concurrently (in background/parallel)

2. **After ~5-10 seconds:**
   - Backend should be accessible at http://localhost:8000
   - Frontend should be accessible at http://localhost:8501
   - No error messages in log

3. **In Browser:**
   - http://localhost:8000 → Returns welcome message
   - http://localhost:8000/health → Returns `{"status":"ok"}`
   - http://localhost:8501 → Shows Streamlit dashboard

### When Running Tests:

1. **Test 1 - Backend Health Check:**
   - Makes request to http://localhost:8000/health
   - Expected: Status 200, response `{"status":"ok"}`
   - Result: ✅ PASS

2. **Test 2 - Backend Root:**
   - Makes request to http://localhost:8000/
   - Expected: Status 200
   - Result: ✅ PASS

3. **Test 3 - Frontend:**
   - Makes request to http://localhost:8501
   - Expected: Status 200, HTML response
   - Result: ✅ PASS

**Final Result:** ✅ All tests passed

---

## ⚠️ Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `Port 8000 already in use` | Another service uses it | Kill existing process, wait 10s, retry |
| `streamlit not found` | Not installed | Run `pip install streamlit` |
| `FastAPI module not found` | Dependencies not installed | Run `pip install -r ml-backend/requirements.txt` |
| `Permission denied (scripts)` | Execution policy restricted | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` |
| `Connection refused` | Services not started | Verify services are running, wait 5s |
| `No such file or directory` | Running from wrong folder | Ensure working directory is root of repo |

---

## ✨ Phase 1 SUCCESS CRITERIA

### All boxes checked? Then Phase 1 is COMPLETE! 🎉

- ✅ Structure created (DDD in ml-backend/)
- ✅ Environment configured (venv, dependencies)
- ✅ Root cleaned (5-10 essential files)
- ✅ Scripts created (bash + PowerShell)
- ✅ Documentation written (6 directories)
- ✅ Tests working (3+ automated tests)

### Ready for Phase 2?

Once all checks pass:
1. Run `start_services.ps1` (or .sh)
2. Open http://localhost:8501
3. See dashboard with backend status
4. Run tests and verify everything passes

**Then:** Proceed to Jour 2 for code migration! 🚀

---

**Status:** 🟢 **Phase 1 Structure Ready**  
**Next:** 🔧 Activation Test (start services & verify)  
**Then:** 📦 Phase 2 Code Migration

*Last Updated: Today*
