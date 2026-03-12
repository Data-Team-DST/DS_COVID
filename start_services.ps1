# Start both FastAPI backend and Streamlit frontend
# Usage: powershell -ExecutionPolicy Bypass -File start_services.ps1

param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8501
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "STARTING MICROSERVICE ARCHITECTURE" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  http://localhost:$BackendPort" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:$FrontendPort" -ForegroundColor Yellow
Write-Host ""

$BackendJob = $null
$FrontendJob = $null

# Cleanup function
function Cleanup {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Stopping services..." -ForegroundColor Red
    Write-Host "==================================================" -ForegroundColor Cyan
    
    if ($BackendJob) {
        Stop-Job -Job $BackendJob -Force -ErrorAction SilentlyContinue
    }
    if ($FrontendJob) {
        Stop-Job -Job $FrontendJob -Force -ErrorAction SilentlyContinue
    }
    
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
}

# Register cleanup on exit
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup }

try {
    # Start Backend (FastAPI)
    Write-Host "Starting Backend (FastAPI on port $BackendPort)..." -ForegroundColor Cyan
    
    $BackendJob = Start-Job -ScriptBlock {
        param($Port, $ProjectRoot)
        Set-Location "$ProjectRoot\ml-backend"
        
        # Activate venv if it exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & .\venv\Scripts\Activate.ps1
        }
        
        # Run FastAPI app
        python app.py
    } -ArgumentList $BackendPort, $ProjectRoot -Name "FastAPI-Backend"
    
    Write-Host "Backend started (Job ID: $($BackendJob.Id))" -ForegroundColor Green
    Start-Sleep -Seconds 2
    
    # Start Frontend (Streamlit)
    Write-Host ""
    Write-Host "Starting Frontend (Streamlit on port $FrontendPort)..." -ForegroundColor Cyan
    
    $FrontendJob = Start-Job -ScriptBlock {
        param($Port, $ProjectRoot)
        Set-Location $ProjectRoot
        
        # Activate venv if it exists
        if (Test-Path "ml-backend\venv\Scripts\Activate.ps1") {
            & ml-backend\venv\Scripts\Activate.ps1
        }
        
        # Install streamlit if needed (quick check)
        python -c "import streamlit" 2>$null
        if ($LASTEXITCODE -ne 0) {
            pip install streamlit >$null 2>&1
        }
        
        # Run Streamlit app
        streamlit run streamlit_app.py --server.port $Port --logger.level=error
    } -ArgumentList $FrontendPort, $ProjectRoot -Name "Streamlit-Frontend"
    
    Write-Host "Frontend started (Job ID: $($FrontendJob.Id))" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "=================================================="  -ForegroundColor Cyan
    Write-Host "Both services started!" -ForegroundColor Green
    Write-Host "=================================================="  -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Access the services:" -ForegroundColor Yellow
    Write-Host "   Frontend: http://localhost:$FrontendPort" -ForegroundColor White
    Write-Host "   Backend:  http://localhost:$BackendPort" -ForegroundColor White
    Write-Host "   Health:   curl http://localhost:$BackendPort/health" -ForegroundColor White
    Write-Host ""
    Write-Host "Press Ctrl+C to stop services..." -ForegroundColor Cyan
    Write-Host ""
    
    # Keep the script running
    while ($true) {
        if ($BackendJob -and $BackendJob.State -eq "Failed") {
            Write-Host "Backend job failed" -ForegroundColor Red
            break
        }
        if ($FrontendJob -and $FrontendJob.State -eq "Failed") {
            Write-Host "Frontend job failed" -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 1
    }
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Cleanup
    exit 1
}
finally {
    Cleanup
}
