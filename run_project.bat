@echo off
setlocal

echo ===============================================
echo Smart India Road Accident Forecasting Dashboard
echo ===============================================

if not exist requirements.txt (
    echo ERROR: requirements.txt not found.
    echo Please run this file from the main project folder where app.py exists.
    pause
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Ensuring pip is available...
".venv\Scripts\python.exe" -m ensurepip --upgrade

echo Upgrading pip tools...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

echo Installing project requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Running data preprocessing...
".venv\Scripts\python.exe" -m src.data_preprocessing

echo Training / refreshing model...
".venv\Scripts\python.exe" -m src.train_model

echo Starting Streamlit dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py

pause
