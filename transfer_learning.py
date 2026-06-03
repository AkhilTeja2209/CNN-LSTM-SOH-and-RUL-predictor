"""
transfer_learning.py
Original notebook cell 28 — EV battery fine-tuning.
Zero changes to logic.
"""

from imports import *
from config import SEQUENCE_LEN, EV_MODEL_PATH, EV_SCALERS_PATH


def load_ev_battery_csv(filepath, cycle_col='cycle', capacity_col='Capacity',
                        voltage_col='Voltage', current_col='Current',
                        temp_col='Temperature', time_col='Time'):
    """Original notebook cell 28 — zero changes."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    col_rename = {
        cycle_col   : 'cycle_index',
        capacity_col: 'Capacity',
        voltage_col : 'Voltage_measured',
        current_col : 'Current_measured',
        temp_col    : 'Temperature_measured',
        time_col    : 'Time',
    }
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

    for col in ['Current_load', 'Voltage_load']:
        if col not in df.columns:
            df[col] = 0.0

    df['type'] = 'discharge'
    tmp_path = '/tmp/ev_battery_standardized.csv'
    df.to_csv(tmp_path, index=False)
    return tmp_path


def retrain_on_ev_battery(ev_csv_path, pretrained_model, scalers_dict,
                          freeze_cnn=True, freeze_lstm=True,
                          epochs=100, batch_size=32, lr=5e-4,
                          col_mapping=None):
    """Original notebook cell 28 — zero changes."""
    feature_cols = scalers_dict['feature_cols']
    seq_len      = scalers_dict['seq_len']

    kwargs   = col_mapping if col_mapping else {}
    std_path = load_ev_battery_csv(ev_csv_path, **kwargs)

    df_raw = pd.read_csv(std_path)
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
            stats = safe_stats(grp[col_name]) if col_name in grp.columns else [0] * 6
            for stat_name, val in zip(['mean', 'std', 'min', 'max', 'med', 'delta'], stats):
                rec[f'{prefix}_{stat_name}'] = val

        rec['discharge_time'] = (float(grp['Time'].iloc[-1] - grp['Time'].iloc[0])
                                  if 'Time' in grp.columns else len(grp))
        rec['energy_discharged'] = float(np.trapz(np.abs(
            grp.get('Voltage_measured', pd.Series([0])) *
            grp.get('Current_measured', pd.Series([0])))))
        rec['peak_power'] = float(np.max(np.abs(
            grp.get('Voltage_measured', pd.Series([0])) *
            grp.get('Current_measured', pd.Series([0])))))
        records.append(rec)

    df_ev = pd.DataFrame(records).dropna(subset=['capacity']).reset_index(drop=True)
    nominal = df_ev['capacity'].iloc[0]
    df_ev['SOH'] = (df_ev['capacity'] / nominal * 100).clip(0, 100)

    eol_idx = df_ev.index[df_ev['SOH'] < 80].tolist()
    if eol_idx:
        df_ev['RUL'] = (eol_idx[0] - df_ev.index).clip(lower=0)
    else:
        df_ev['RUL'] = (len(df_ev) - df_ev.index).clip(lower=0)

    for col in feature_cols:
        if col not in df_ev.columns:
            df_ev[col] = 0.0

    ev_scaler_X   = MinMaxScaler()
    ev_scaler_soh = MinMaxScaler()
    ev_scaler_rul = MinMaxScaler()

    df_ev[feature_cols] = ev_scaler_X.fit_transform(df_ev[feature_cols].fillna(0))
    df_ev[['SOH']]      = ev_scaler_soh.fit_transform(df_ev[['SOH']])
    df_ev[['RUL']]      = ev_scaler_rul.fit_transform(df_ev[['RUL']])

    data  = df_ev[feature_cols].values
    soh_v = df_ev['SOH'].values
    rul_v = df_ev['RUL'].values
    X_ev, y_soh_ev, y_rul_ev = [], [], []
    for i in range(seq_len, len(data)):
        X_ev.append(data[i - seq_len:i])
        y_soh_ev.append(soh_v[i])
        y_rul_ev.append(rul_v[i])

    X_ev     = np.array(X_ev, dtype=np.float32)
    y_soh_ev = np.array(y_soh_ev, dtype=np.float32)
    y_rul_ev = np.array(y_rul_ev, dtype=np.float32)

    X_tr, X_vl, y_s_tr, y_s_vl, y_r_tr, y_r_vl = train_test_split(
        X_ev, y_soh_ev, y_rul_ev, test_size=0.2, random_state=42
    )

    print(f"EV dataset: {len(df_ev)} cycles → {len(X_ev)} sequences")
    print(f"Train: {len(X_tr)}, Val: {len(X_vl)}")

    ft_model = tf.keras.models.clone_model(pretrained_model)
    ft_model.set_weights(pretrained_model.get_weights())

    for layer in ft_model.layers:
        name = layer.name.lower()
        if freeze_cnn and 'conv' in name:
            layer.trainable = False
        elif freeze_lstm and ('lstm' in name or 'bidirectional' in name):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_params = sum([np.prod(v.shape) for v in ft_model.trainable_variables])
    total_params     = sum([np.prod(v.shape) for v in ft_model.variables])
    print(f"\nTransfer learning config:")
    print(f"  Trainable params : {trainable_params:,} / {total_params:,} "
          f"({trainable_params / total_params * 100:.1f}%)")

    huber = tf.keras.losses.Huber(delta=0.1)
    ft_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={'soh': huber, 'rul': huber},
        loss_weights={'soh': 1.0, 'rul': 0.8},
        metrics={'soh': ['mae'], 'rul': ['mae']},
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1),
        ModelCheckpoint(EV_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    ev_history = ft_model.fit(
        X_tr, {'soh': y_s_tr, 'rul': y_r_tr},
        validation_data=(X_vl, {'soh': y_s_vl, 'rul': y_r_vl}),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )

    print("\n✅ EV battery fine-tuning complete.")
    print(f"   Model saved: {EV_MODEL_PATH}")

    ev_scalers = {
        'scaler_X'     : ev_scaler_X,
        'scaler_soh'   : ev_scaler_soh,
        'scaler_rul'   : ev_scaler_rul,
        'feature_cols' : feature_cols,
        'seq_len'      : seq_len,
        'eol_threshold': 80.0,
    }
    with open(EV_SCALERS_PATH, 'wb') as f:
        pickle.dump(ev_scalers, f)
    print(f"   Scalers saved: {EV_SCALERS_PATH}")

    return ft_model, ev_history, ev_scalers
