@echo off
setlocal
cd /d "%~dp0"
title PLM Thesis Defense - Adaptive Disk Monitor

echo =========================================================
echo  PLM - DISK USAGE ANOMALY RESEARCH DASHBOARD
echo =========================================================
echo.
echo Preparing the local dashboard...

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python virtual environment...
    py -3.11 -m venv .venv >nul 2>&1
    if errorlevel 1 (
        py -m venv .venv >nul 2>&1
        if errorlevel 1 (
            python -m venv .venv
            if errorlevel 1 goto error
        )
    )
)

echo Installing or checking project requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo Starting presentation dashboard...
echo Open http://localhost:8501 if your browser does not open.
".venv\Scripts\python.exe" -m streamlit run dashboard.py
if errorlevel 1 goto error

goto done
:error
echo.
echo Launch failed. Confirm that Python 3.11 or newer is installed,
echo then check the error messages above. Installation needs internet.
:done
echo.
pause
