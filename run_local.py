"""
run_local.py  —  Solution 1: run the full pipeline from PowerShell / VS Code terminal.

Usage:
    python run_local.py

All outputs (model, scalers, plots) are saved to ./outputs/
"""

import subprocess
import sys

# ── Auto-install dependencies ──────────────────────────────────────────────
REQUIRED = [
    'kaggle',
    'scikit-learn',
    'tensorflow',
    'matplotlib',
    'seaborn',
    'pandas',
    'numpy',
    'scipy',
]

print("Checking dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + REQUIRED)
print("✅ All dependencies ready.\n")

import os

# ── Set local paths BEFORE importing config ────────────────────────────────
os.environ['BATTERY_DATA_DIR']   = r'.\data\nasa_battery'
os.environ['BATTERY_MODEL']      = r'.\outputs\battery_cnn_lstm_model.keras'
os.environ['BATTERY_BEST']       = r'.\outputs\best_battery_model.keras'
os.environ['BATTERY_SCALERS']    = r'.\outputs\scalers.pkl'
os.environ['BATTERY_EV_MODEL']   = r'.\outputs\ev_finetuned_model.keras'
os.environ['BATTERY_EV_SCALERS'] = r'.\outputs\ev_scalers.pkl'
os.environ['PLOT_EDA']           = r'.\outputs\eda_plot.png'
os.environ['PLOT_HISTORY']       = r'.\outputs\training_history.png'
os.environ['PLOT_EVAL']          = r'.\outputs\evaluation_results.png'
os.environ['PLOT_SOLAR']         = r'.\outputs\solar_battery_prediction.png'

# Ensure outputs folder exists
os.makedirs(r'.\outputs', exist_ok=True)

# ── Now import pipeline modules (they read paths from os.environ via config) ─
sys.path.insert(0, os.path.dirname(__file__))

from config import print_config, DATA_DIR, MODEL_PATH, SCALERS_PATH
from config import FEATURE_COLS, SEQUENCE_LEN
from data_loader import download_dataset, load_all_batteries
from feature_engineering import plot_eda, prepare_data
from model import build_cnn_lstm_model, train_model, plot_training_history, evaluate_model
from inference import save_model_and_scalers

# ── 1. Config ──────────────────────────────────────────────────────────────
print_config()

# ── 2. Download dataset (skipped if already present) ──────────────────────
# UPDATED: Checks for both .mat and .csv files
if not os.path.exists(DATA_DIR) or not any(
    (f.endswith('.mat') or f.endswith('.csv')) for _, _, files in os.walk(DATA_DIR) for f in files
):
    download_dataset()
else:
    print(f"Dataset already present at {DATA_DIR} — skipping download.")

# ── 3. Load all batteries ──────────────────────────────────────────────────
df_all = load_all_batteries()

# ── 4. EDA ────────────────────────────────────────────────────────────────
plot_eda(df_all)

# ── 5. Sequences + scaling ─────────────────────────────────────────────────
data = prepare_data(df_all)

X_train, X_val, X_test             = data['X_train'], data['X_val'], data['X_test']
y_soh_train, y_soh_val, y_soh_test = data['y_soh_train'], data['y_soh_val'], data['y_soh_test']
y_rul_train, y_rul_val, y_rul_test = data['y_rul_train'], data['y_rul_val'], data['y_rul_test']
scaler_X, scaler_soh, scaler_rul   = data['scaler_X'], data['scaler_soh'], data['scaler_rul']

# ── 6. Build model ────────────────────────────────────────────────────────
model = build_cnn_lstm_model(SEQUENCE_LEN, len(FEATURE_COLS))
model.summary()

# ── 7. Train ──────────────────────────────────────────────────────────────
history = train_model(model, X_train, y_soh_train, y_rul_train,
                              X_val,   y_soh_val,   y_rul_val)

# ── 8. Training history plots ─────────────────────────────────────────────
plot_training_history(history)

# ── 9. Evaluate ───────────────────────────────────────────────────────────
evaluate_model(model, X_test, y_soh_test, y_rul_test, scaler_soh, scaler_rul)

# ── 10. Save ──────────────────────────────────────────────────────────────
save_model_and_scalers(model, scaler_X, scaler_soh, scaler_rul)

print(f"\n✅ Pipeline complete. All outputs saved to .\\outputs\\")