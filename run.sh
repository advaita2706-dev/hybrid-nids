#!/bin/bash
# ============================================
#  Hybrid NIDS — Network Intrusion Detection
#  Run script for Linux / macOS / WSL
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   Hybrid NIDS                            ║"
echo "  ║   Network Intrusion Detection System     ║"
echo "  ║   RF · Autoencoder · SHAP                ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# Check venv
if [ ! -f "venv/bin/activate" ]; then
    echo "  [!] Virtual environment not found."
    echo "      Creating one now..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check models
if [ ! -f "models/random_forest.pkl" ]; then
    echo "  [!] Models not found. Run training first:"
    echo "      python model/train_pipeline.py"
    exit 1
fi

# Start API server (serves dashboard at http://localhost:8000)
echo "  [*] Starting API server on port 8000..."
echo ""
echo "  ┌──────────────────────────────────────────┐"
echo "  │  Dashboard: http://localhost:8000         │"
echo "  │  API Docs:  http://localhost:8000/docs    │"
echo "  │  Press Ctrl+C to stop.                    │"
echo "  └──────────────────────────────────────────┘"
echo ""

# Open browser (best-effort)
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:8000 2>/dev/null &
elif command -v open &>/dev/null; then
    open http://localhost:8000 2>/dev/null &
fi

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
