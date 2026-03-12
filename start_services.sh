#!/usr/bin/env bash
# Start both FastAPI backend and Streamlit frontend

set -e

BACKEND_PORT=8000
FRONTEND_PORT=8501
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "🚀 COVID-19 Microservice Launcher"
echo "=================================================="
echo ""
echo "BACKEND:  http://localhost:${BACKEND_PORT}"
echo "FRONTEND: http://localhost:${FRONTEND_PORT}"
echo ""

# Cleanup function on exit
cleanup() {
    echo ""
    echo "=================================================="
    echo "🛑 Stopping services..."
    echo "=================================================="
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait 2>/dev/null || true
}

trap cleanup EXIT

# Start Backend (FastAPI)
echo "⚙️  Starting Backend (FastAPI on port ${BACKEND_PORT})..."
cd "${PROJECT_ROOT}/ml-backend"

# Activate venv if it exists
if [ -d venv ]; then
    source venv/bin/activate
fi

python app.py &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
sleep 2

# Start Frontend (Streamlit)
echo ""
echo "🎨 Starting Frontend (Streamlit on port ${FRONTEND_PORT})..."
cd "${PROJECT_ROOT}"

# Check if streamlit_app.py exists
if [ ! -f streamlit_app.py ]; then
    echo "⚠️  streamlit_app.py not found! Creating placeholder..."
    mkdir -p pages
    cat > streamlit_app.py << 'EOF'
import streamlit as st
import requests
import json

st.set_page_config(page_title="COVID-19 ML Dashboard", layout="wide")

st.title("🦠 COVID-19 ML Prediction Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("Backend Status")
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend API is running!")
            st.json(response.json())
        else:
            st.error("❌ Backend returned error")
    except Exception as e:
        st.error(f"❌ Cannot connect to backend: {str(e)}")

with col2:
    st.header("Microservice Architecture")
    st.info("""
    **Frontend:** Streamlit (Port 8501)
    **Backend:** FastAPI (Port 8000)
    
    Status: 🟢 Microservice architecture working!
    """)

st.divider()

# API Test Section
st.header("📊 API Health Check")
if st.button("Test Backend API"):
    try:
        response = requests.get("http://localhost:8000/health")
        st.success("Backend Response:")
        st.json(response.json())
    except Exception as e:
        st.error(f"Error: {str(e)}")
EOF
fi

streamlit run streamlit_app.py --server.port ${FRONTEND_PORT} &
FRONTEND_PID=$!
echo "✓ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "=================================================="
echo "✅ Both services started!"
echo "=================================================="
echo ""
echo "📍 Access:"
echo "   Frontend: http://localhost:${FRONTEND_PORT}"
echo "   Backend:  http://localhost:${BACKEND_PORT}"
echo "   Health:   curl http://localhost:${BACKEND_PORT}/health"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Keep script running
wait
