#!/bin/bash
echo "===================================================="
echo "            httfacelond Auto-Updater            "
echo "===================================================="
echo ""
echo "[1/3] Pulling latest code changes from GitHub..."
git pull
echo ""
echo "[2/3] Checking virtual environment..."
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Updating dependencies..."
    pip install -r requirements.txt --upgrade
else
    echo "[Warning] Virtual environment (venv) not found."
    echo "If you want dependencies to update automatically, please create a 'venv' directory."
fi
echo ""
echo "[3/3] Update process complete!"
echo "===================================================="
