"""
config.py
All shared constants extracted from the original notebook.
Every other module imports from here — change values here only.
"""

import os
import warnings
import numpy as np
import tensorflow as tf

# ── Reproducibility ────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')

# ── Paths — override for local runs ───────────────────────────────────────
# Colab defaults; run_local.py overwrites these before importing other modules
DATA_DIR         = os.environ.get('BATTERY_DATA_DIR',  '/content/nasa_battery')
MODEL_PATH       = os.environ.get('BATTERY_MODEL',     '/content/battery_cnn_lstm_model.keras')
BEST_MODEL_PATH  = os.environ.get('BATTERY_BEST',      '/content/best_battery_model.keras')
SCALERS_PATH     = os.environ.get('BATTERY_SCALERS',   '/content/scalers.pkl')
EV_MODEL_PATH    = os.environ.get('BATTERY_EV_MODEL',  '/content/ev_finetuned_model.keras')
EV_SCALERS_PATH  = os.environ.get('BATTERY_EV_SCALERS','/content/ev_scalers.pkl')

PLOT_EDA      = os.environ.get('PLOT_EDA',      '/content/eda_plot.png')
PLOT_HISTORY  = os.environ.get('PLOT_HISTORY',  '/content/training_history.png')
PLOT_EVAL     = os.environ.get('PLOT_EVAL',     '/content/evaluation_results.png')
PLOT_SOLAR    = os.environ.get('PLOT_SOLAR',    '/content/solar_battery_prediction.png')

# ── Feature / target config ────────────────────────────────────────────────
FEATURE_COLS = [
    'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max', 'voltage_delta',
    'current_mean', 'current_std', 'current_delta',
    'temp_mean', 'temp_std', 'temp_max', 'temp_delta',
    'cur_load_mean', 'cur_load_std',
    'vol_load_mean', 'vol_load_std',
    'energy_discharged', 'peak_power', 'discharge_time', 'capacity',
]
TARGET_SOH    = 'SOH'
TARGET_RUL    = 'RUL'
SEQUENCE_LEN  = 20
EOL_THRESHOLD = 80.0

# ── Model hyperparameters ──────────────────────────────────────────────────
LSTM_UNITS   = 128
CNN_FILTERS  = 64
DROPOUT      = 0.3
EPOCHS       = 200
BATCH_SIZE   = 64
LEARNING_RATE = 1e-3

def print_config():
    print("=" * 50)
    print("  Battery SOH & RUL Pipeline — Configuration")
    print("=" * 50)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU        : {tf.config.list_physical_devices('GPU')}")
    print(f"  Seq len    : {SEQUENCE_LEN} cycles")
    print(f"  EOL thresh : {EOL_THRESHOLD}%")
    print(f"  Features   : {len(FEATURE_COLS)}")
    print(f"  Data dir   : {DATA_DIR}")
    print("=" * 50)
