@echo off
title Hybrid NIDS - Network Intrusion Detection System
color 0B
echo.
echo   ====================================================
echo    _   _ ___ ____  ____
echo   ^| ^| ^| ^|_ _^|  _ \/ ___^|
echo   ^| ^|_^| ^| ^| ^| ^| ^| \___ \
echo   ^|  _  ^| ^| ^| ^|_^| ^|___) ^|
echo   ^|_^| ^|_^|___^|____/^|____/
echo.
echo    Hybrid Network Intrusion Detection System
echo    RF + Autoencoder + SHAP Explainability
echo   ====================================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo  [!] Virtual environment not found.
    echo      Run setup.bat first.
    echo.
    pause
    exit /b 1
)

:: Check if models exist
if not exist "models\random_forest.pkl" (
    echo  [!] Trained models not found in models/ folder.
    echo      Run the training pipeline first:
    echo      python model/train_pipeline.py
    echo.
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

echo.
echo  ====================================================
echo   Dashboard: http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   Close this window to stop the NIDS system.
echo  ====================================================
echo.

:: Open browser
start http://localhost:8000

:: Run API (foreground — closing window stops it)
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
