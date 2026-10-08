@echo off
title NIDS Setup - One-Time Installation
echo.
echo  ============================================
echo   Hybrid NIDS - First-Time Setup
echo   Network Intrusion Detection System
echo  ============================================
echo.
echo  This will create a Python virtual environment
echo  and install all dependencies. Run this ONCE.
echo.
pause

echo.
echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Python not found. Install 64-bit Python 3.12 or 3.13 from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [2/3] Activating environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Installing packages failed. Check your internet connection,
    echo and use 64-bit Python 3.12 or 3.13 from python.org.
    pause
    exit /b 1
)

echo.
echo  ============================================
echo   Setup Complete!
echo   Run "run.bat" to start the NIDS dashboard.
echo  ============================================
echo.
pause
