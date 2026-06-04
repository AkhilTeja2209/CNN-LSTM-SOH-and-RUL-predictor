# 🔋 Battery SOH & RUL Prediction Pipeline

An end-to-end deep learning pipeline to predict the State of Health (SOH) and Remaining Useful Life (RUL) of Lithium-ion batteries using a hybrid **CNN-BiLSTM architecture with Multi-Head Self-Attention**.

This repository is pre-trained on the **NASA Battery Dataset** and includes modules for local solar battery inference and transfer-learning fine-tuning for EV battery datasets.

## 🧠 Model Architecture
- **Input:** 20-cycle sliding window of 20 engineered features (voltage, current, temperature, load).
- **Spatial Encoder:** 3-block 1D CNN to extract local feature correlations.
- **Temporal Encoder:** Bidirectional LSTM to capture long-term degradation trends.
- **Attention:** Multi-Head Self-Attention to focus on critical degradation phases.
- **Output:** Dual regression heads for SOH (%) and RUL (remaining cycles).

## 🚀 How to Run

This project is modular and can be run in two ways depending on your setup.

### Solution 1: Local Terminal Execution (Recommended)
Run the complete pipeline locally. The script handles dependencies, downloads the dataset via the Kaggle API, processes the raw files, trains the model, and saves all artifacts.

1. Ensure your Kaggle API token (`kaggle.json`) is in your user directory (e.g., `~/.kaggle/`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python run_local.py
   ```
*All models (`.keras`), scalers (`.pkl`), and plots will be saved to the `./outputs/` folder.*

### Solution 2: Google Colab / Jupyter Execution
For interactive execution or utilizing free cloud GPUs:

1. Open `run_colab.ipynb` in Jupyter or Google Colab.
2. Upload the accompanying `.py` module files (`config.py`, `data_loader.py`, etc.) when prompted in the notebook.
3. Run the cells sequentially. The notebook imports the Python modules to keep the interface clean while executing the pipeline.

## ☀️ Custom Inference (Solar Batteries)
You can use the pre-trained model to evaluate your own battery data. Collect a minimum of 20 discharge cycles with voltage, current, and temperature, and run the `predict_solar_battery` function found in `inference.py`.

## 🚗 EV Transfer Learning
The `transfer_learning.py` module allows you to freeze the core CNN/LSTM layers and fine-tune the dense heads on new EV battery datasets (e.g., CALCE, MATR) with minimal compute.

## ⚠️ NOTE: 
The `run_local.py` module and `run_colab.ipynb` module are still work in progress and DO NOT include custom inference and EV transfer learning yet.

## 📊 Dataset
Powered by the [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset). The custom data loader automatically parses both the original `.mat` structures and the unpacked Kaggle `cleaned_dataset` manifests.
