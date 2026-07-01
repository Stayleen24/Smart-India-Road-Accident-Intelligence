#!/usr/bin/env bash
set -e

echo "==============================================="
echo "Smart India Road Accident Forecasting Dashboard"
echo "==============================================="

if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt not found. Run this script from the main project folder where app.py exists."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

echo "Ensuring pip is available..."
./.venv/bin/python -m ensurepip --upgrade || true

echo "Upgrading pip tools..."
./.venv/bin/python -m pip install --upgrade pip setuptools wheel

echo "Installing project requirements..."
./.venv/bin/python -m pip install -r requirements.txt

echo "Running data preprocessing..."
./.venv/bin/python -m src.data_preprocessing

echo "Training / refreshing model..."
./.venv/bin/python -m src.train_model

echo "Starting Streamlit dashboard..."
./.venv/bin/python -m streamlit run app.py
