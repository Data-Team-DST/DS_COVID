# ⚡ Quick Start Guide - FastAPI Backend

**Ready to test?** This takes 2 minutes.

---

## 🚀 Launch Backend (Windows PowerShell)

### Step 1: Open PowerShell and navigate to backend
```powershell
cd "C:\Users\u1050780\OneDrive - Sanofi\Documents\DS_COVID\ml-backend"
```

### Step 2: Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
Expected output: `(venv)` appears in prompt

### Step 3: Start the server
```powershell
python app.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

---

## ✅ Test It (New PowerShell Window)

```powershell
# Test 1: Health check
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# Test 2: Welcome page
curl http://localhost:8000/
# Expected: HTML response
```

---

## 🛑 Stop the Server

```powershell
# In the window where app.py is running:
Ctrl + C
```

---

## 📱 Access via Browser

- **API Root:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Health:** http://localhost:8000/health

---

## ⚙️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Activate venv first: `.\venv\Scripts\Activate.ps1` |
| `Port 8000 already in use` | Run `netstat -ano \| findstr :8000`, then kill process |
| `Python not found` | Use full path: `.\venv\Scripts\python.exe app.py` |
| Server won't start | Check logs for errors, verify `requirements.txt` installed |

---

## 📚 Next Steps

Once confirmed working:

1. **Review migration plan:** Open `CODE_INVENTORY.md`
2. **See architecture:** Open `MICROSERVICES_README.md`
3. **Continue Jour 2:** Start code migration to DDD layers

---

**Tip:** Bookmark this file for easy access! 🔖
