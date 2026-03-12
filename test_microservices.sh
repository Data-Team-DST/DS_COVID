#!/usr/bin/env bash
# Test microservice architecture
# Verify that backend and frontend can communicate

set -e

BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:8501"
TIMEOUT=5

echo "=================================================="
echo "🧪 Microservice Architecture Test"
echo "=================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

test_count=0
pass_count=0

# Test function
run_test() {
    local name=$1
    local url=$2
    local expected_code=$3
    
    test_count=$((test_count + 1))
    echo -n "Test $test_count: $name ... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $response)"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}✗ FAIL${NC} (Got HTTP $response, expected $expected_code)"
    fi
}

# Give services time to start
echo "Waiting for services to start..."
sleep 3

echo ""
echo "🔍 Running tests..."
echo ""

# Backend tests
echo "Backend Tests (Port 8000):"
run_test "Backend Health Check" "$BACKEND_URL/health" "200"
run_test "Backend Root Endpoint" "$BACKEND_URL/" "200"

echo ""
echo "Frontend Tests (Port 8501):"
run_test "Frontend Main Page" "$FRONTEND_URL/" "200"

echo ""
echo "=================================================="
echo "Test Results: $pass_count / $test_count PASSED"
echo "=================================================="

if [ $pass_count -eq $test_count ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo "Microservice architecture is working correctly."
    exit 0
else
    echo -e "${RED}❌ Some tests failed.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Ensure start_services.sh is running"
    echo "2. Check that ports 8000 and 8501 are not in use"
    echo "3. Verify backend is started: curl http://localhost:8000/health"
    echo "4. Verify frontend is started: curl http://localhost:8501/"
    exit 1
fi
