"""
inference.py
Original notebook cells 23 (save) + 25 (predict_solar_battery) + 26 (usage).
Zero changes to logic.
"""

from imports import *
from config import (
    FEATURE_COLS, SEQUENCE_LEN, EOL_THRESHOLD,
    MODEL_PATH, SCALERS_PATH, PLOT_SOLAR,
)


# ── Cell 23: Save ─────────────────────────────────────────────────────────

def save_model_and_scalers(model, scaler_X, scaler_soh, scaler_rul):
    model.save(MODEL_PATH)
    with open(SCALERS_PATH, 'wb') as f:
        pickle.dump({
            'scaler_X'     : scaler_X,
            'scaler_soh'   : scaler_soh,
            'scaler_rul'   : scaler_rul,
            'feature_cols' : FEATURE_COLS,
            'seq_len'      : SEQUENCE_LEN,
            'eol_threshold': EOL_THRESHOLD,
        }, f)
    print(f"✅ Model saved  : {MODEL_PATH}")
    print(f"✅ Scalers saved: {SCALERS_PATH}")


def load_model_and_scalers():
    model = load_model(MODEL_PATH)
    with open(SCALERS_PATH, 'rb') as f:
        scalers_dict = pickle.load(f)
    print(f"✅ Model loaded  : {MODEL_PATH}")
    print(f"✅ Scalers loaded: {SCALERS_PATH}")
    return model, scalers_dict


# ── Cell 25: predict_solar_battery ────────────────────────────────────────

def predict_solar_battery(csv_path, model, scalers_dict, plot=True):
    """
    Predict SOH and RUL for a locally-built solar battery.
    Original notebook cell 25 — zero changes.
    """
    scaler_X     = scalers_dict['scaler_X']
    scaler_soh   = scalers_dict['scaler_soh']
    scaler_rul   = scalers_dict['scaler_rul']
    feature_cols = scalers_dict['feature_cols']
    seq_len      = scalers_dict['seq_len']
    eol          = scalers_dict['eol_threshold']

    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip()
    df_dis = df_raw[df_raw['type'] == 'discharge'].copy()

    records = []
    for cyc_idx, grp in df_dis.groupby('cycle_index'):
        def safe_stats(arr):
            arr = np.array(arr.dropna())
            if len(arr) == 0: return [np.nan] * 6
            return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
                    np.median(arr), arr[-1] - arr[0]]

        capacity = grp['Capacity'].mean()
        rec = {'cycle_index': cyc_idx, 'capacity': capacity}
        col_map = {
            'Voltage_measured'    : 'voltage',
            'Current_measured'    : 'current',
            'Temperature_measured': 'temp',
            'Current_load'        : 'cur_load',
            'Voltage_load'        : 'vol_load',
        }
        for col_name, prefix in col_map.items():
            if col_name in grp.columns:
                stats = safe_stats(grp[col_name])
                for stat_name, val in zip(['mean', 'std', 'min', 'max', 'med', 'delta'], stats):
                    rec[f'{prefix}_{stat_name}'] = val
            else:
                for stat_name in ['mean', 'std', 'min', 'max', 'med', 'delta']:
                    rec[f'{prefix}_{stat_name}'] = 0.0

        rec['discharge_time'] = (float(grp['Time'].iloc[-1] - grp['Time'].iloc[0])
                                  if 'Time' in grp.columns else len(grp))

        if 'Voltage_measured' in grp.columns and 'Current_measured' in grp.columns:
            power = np.abs(grp['Voltage_measured'] * grp['Current_measured'])
            rec['energy_discharged'] = float(np.trapz(power))
            rec['peak_power']        = float(power.max())
        else:
            rec['energy_discharged'] = 0.0
            rec['peak_power']        = 0.0

        records.append(rec)

    df_cyc = pd.DataFrame(records).dropna(subset=['capacity']).reset_index(drop=True)

    if len(df_cyc) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} discharge cycles. Got {len(df_cyc)}.")

    nominal = df_cyc['capacity'].iloc[0]
    df_cyc['SOH'] = (df_cyc['capacity'] / nominal * 100).clip(0, 100)

    for col in feature_cols:
        if col not in df_cyc.columns:
            df_cyc[col] = 0.0

    df_cyc[feature_cols] = scaler_X.transform(df_cyc[feature_cols].fillna(0))
    data  = df_cyc[feature_cols].values
    X_seq = np.array([data[i - seq_len:i] for i in range(seq_len, len(data))], dtype=np.float32)

    preds        = model.predict(X_seq, verbose=0)
    soh_pred_raw = scaler_soh.inverse_transform(preds['soh']).flatten()
    rul_pred_raw = scaler_rul.inverse_transform(preds['rul']).flatten()

    cycle_nums = df_cyc['cycle_index'].values[seq_len:]
    soh_true   = df_cyc['SOH'].values[seq_len:]

    last_soh = soh_pred_raw[-1]
    last_rul = max(0, int(round(rul_pred_raw[-1])))
    status   = "🟢 HEALTHY" if last_soh >= eol else "🔴 DEGRADED"

    print("\n" + "=" * 50)
    print("  SOLAR BATTERY HEALTH REPORT")
    print("=" * 50)
    print(f"  Latest Cycle       : {int(cycle_nums[-1])}")
    print(f"  Predicted SOH      : {last_soh:.2f}%")
    print(f"  Predicted RUL      : {last_rul} cycles")
    print(f"  Status             : {status}")
    print(f"  Nominal Capacity   : {nominal:.4f} Ah")
    print("=" * 50)

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('🌞 Solar Battery — SOH & RUL Prediction', fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(cycle_nums, soh_true,    label='Measured SOH',  color='steelblue', lw=1.5)
        ax.plot(cycle_nums, soh_pred_raw, label='Predicted SOH', color='orange',   lw=2, linestyle='--')
        ax.axhline(eol, color='red', lw=1.5, linestyle=':', label=f'EOL ({eol}%)')
        ax.fill_between(cycle_nums, soh_pred_raw, eol, where=soh_pred_raw < eol,
                        alpha=0.2, color='red', label='Degraded Zone')
        ax.set_xlabel('Cycle Index'); ax.set_ylabel('SOH (%)')
        ax.set_title('State of Health Over Cycles'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(cycle_nums, rul_pred_raw, label='Predicted RUL', color='green', lw=2)
        ax.axhline(20, color='orange', lw=1.5, linestyle='--', label='Urgency Threshold (20 cycles)')
        ax.set_xlabel('Cycle Index'); ax.set_ylabel('RUL (cycles)')
        ax.set_title('Remaining Useful Life'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        residuals = soh_true - soh_pred_raw
        ax.plot(cycle_nums, residuals, color='purple', lw=1.2, alpha=0.7)
        ax.axhline(0, color='black', lw=1.5, linestyle='--')
        ax.fill_between(cycle_nums, residuals, 0, alpha=0.15, color='purple')
        ax.set_xlabel('Cycle Index'); ax.set_ylabel('SOH Residual')
        ax.set_title('SOH Prediction Error Over Time'); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')
        mae = mean_absolute_error(soh_true, soh_pred_raw)
        rmse = np.sqrt(mean_squared_error(soh_true, soh_pred_raw))
        r2   = r2_score(soh_true, soh_pred_raw)
        soh_cls_true = (soh_true    >= eol).astype(int)
        soh_cls_pred = (soh_pred_raw >= eol).astype(int)
        acc = accuracy_score(soh_cls_true, soh_cls_pred)
        f1  = f1_score(soh_cls_true, soh_cls_pred, average='weighted', zero_division=0)
        summary_text = (
            f"  Prediction Summary\n"
            f"  {'─'*30}\n"
            f"  Current SOH     : {last_soh:.2f}%\n"
            f"  Current RUL     : {last_rul} cycles\n"
            f"  Status          : {status}\n\n"
            f"  SOH MAE         : {mae:.3f}%\n"
            f"  SOH RMSE        : {rmse:.3f}%\n"
            f"  SOH R²          : {r2:.3f}\n"
            f"  Accuracy        : {acc*100:.1f}%\n"
            f"  F1 Score        : {f1:.3f}\n"
        )
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(PLOT_SOLAR, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nPlot saved: {PLOT_SOLAR}")

    return soh_pred_raw, rul_pred_raw, cycle_nums
