from fastapi import FastAPI

app = FastAPI(title="COVID-19 ML API", version="0.1.0")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "COVID-19 ML Backend API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
