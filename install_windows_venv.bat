@echo off
:: Install script for Windows using Python venv
title httfacelond Windows (venv) Installer

echo ==========================================================
echo  Starting httfacelond Windows (venv) Installation
echo ==========================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python is not installed or not added to your system PATH!
    echo Please install Python 3.12 from python.org and try again.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [Error] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing Python dependencies...
python -m pip install --upgrade pip
pip install numpy==1.26.4
pip install -r requirements.txt

echo ==========================================================
echo  Downloading required AI models...
echo ==========================================================
python httfacelond\utils\downloader.py

echo ==========================================================
echo  Installation completed successfully!
echo ==========================================================
echo  To run the application:
echo  1. Run 'run_studio.bat' or activate venv manually: call venv\Scripts\activate
echo  2. Start application: python app.py
echo ==========================================================
pause
