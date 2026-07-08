@echo off
title httfacelond
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Starting application...
python app.py
pause
