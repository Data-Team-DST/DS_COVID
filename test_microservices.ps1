# Test microservice architecture
# Verify that backend and frontend can communicate
# Usage: powershell -ExecutionPolicy Bypass -File test_microservices.ps1

param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8501,
    [int]$TimeoutSeconds = 5
)

Write-Host "=================================================="  -ForegroundColor Cyan
Write-Host "🧪 Microservice Architecture Test"                 -ForegroundColor Green
Write-Host "=================================================="  -ForegroundColor Cyan
Write-Host ""

$BackendUrl = "http://localhost:$BackendPort"
$FrontendUrl = "http://localhost:$FrontendPort"

$testCount = 0
$passCount = 0

# Helper function to test endpoint
function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Url,
        [int]$ExpectedCode = 200
    )
    
    $script:testCount++
    Write-Host -NoNewline "Test $testCount : $Name ... "
    
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $script:TimeoutSeconds -ErrorAction SilentlyContinue
        
        if ($response.StatusCode -eq $ExpectedCode) {
            Write-Host "✓ PASS" -ForegroundColor Green -NoNewline
            Write-Host " (HTTP $($response.StatusCode))" -ForegroundColor White
            $script:passCount++
        } else {
            Write-Host "✗ FAIL" -ForegroundColor Red -NoNewline
            Write-Host " (Got HTTP $($response.StatusCode), expected $ExpectedCode)" -ForegroundColor White
        }
    }
    catch {
        Write-Host "✗ FAIL" -ForegroundColor Red -NoNewline
        Write-Host " (Connection error)" -ForegroundColor White
    }
}

# Wait for services
Write-Host "Waiting for services to start..."
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "🔍 Running tests..."
Write-Host ""

# Backend tests
Write-Host "Backend Tests (Port $BackendPort):" -ForegroundColor Yellow
Test-Endpoint "Backend Health Check" "$BackendUrl/health" 200
Test-Endpoint "Backend Root Endpoint" "$BackendUrl/" 200

Write-Host ""
Write-Host "Frontend Tests (Port $FrontendPort):" -ForegroundColor Yellow
Test-Endpoint "Frontend Main Page" "$FrontendUrl/" 200

Write-Host ""
Write-Host "=================================================="  -ForegroundColor Cyan
Write-Host "Test Results: $passCount / $testCount PASSED"      -ForegroundColor Cyan
Write-Host "=================================================="  -ForegroundColor Cyan

if ($passCount -eq $testCount) {
    Write-Host ""
    Write-Host "✅ All tests passed!" -ForegroundColor Green
    Write-Host "Microservice architecture is working correctly." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Open browser: http://localhost:$FrontendPort" -ForegroundColor White
    Write-Host "  2. Backend API: http://localhost:$BackendPort" -ForegroundColor White
    Write-Host "  3. Health check: curl http://localhost:$BackendPort/health" -ForegroundColor White
}
else {
    Write-Host ""
    Write-Host "❌ Some tests failed." -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Ensure start_services.ps1 is running" -ForegroundColor White
    Write-Host "  2. Check that ports $BackendPort and $FrontendPort are not in use" -ForegroundColor White
    Write-Host "  3. Verify backend: curl http://localhost:$BackendPort/health" -ForegroundColor White
    Write-Host "  4. Verify frontend: curl http://localhost:$FrontendPort/" -ForegroundColor White
    Write-Host "  5. Check firewall settings" -ForegroundColor White
    exit 1
}
