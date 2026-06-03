"""
feature_engineering.py
Original notebook cell 10 (EDA plots) + cell 12 (sequences + scaling).
Zero changes to logic.
"""

from imports import *
from config import (
    FEATURE_COLS, TARGET_SOH, TARGET_RUL,
    SEQUENCE_LEN, EOL_THRESHOLD, PLOT_EDA,
)


# ── Cell 10: EDA ───────────────────────────────────────────────────────────

def plot_eda(df_all):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('NASA Battery Dataset — EDA', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10.colors

    ax = axes[0, 0]
    for i, (bat_id, grp) in enumerate(df_all.groupby('battery_id')):
        ax.plot(grp.index - grp.index[0], grp['SOH'], label=bat_id, color=colors[i % 10], lw=1.5)
    ax.axhline(80, color='red', linestyle='--', lw=1.5, label='EOL (80%)')
    ax.set_xlabel('Relative Cycle'); ax.set_ylabel('SOH (%)')
    ax.set_title('State of Health Degradation'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(df_all['RUL'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('RUL (cycles)'); ax.set_ylabel('Count')
    ax.set_title('RUL Distribution'); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for i, (bat_id, grp) in enumerate(df_all.groupby('battery_id')):
        ax.scatter(grp['cycle_index'], grp['capacity'], label=bat_id, s=10,
                   color=colors[i % 10], alpha=0.7)
    ax.set_xlabel('Cycle Index'); ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Capacity Fade'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sc = ax.scatter(df_all['temp_mean'], df_all['SOH'], c=df_all['cycle_index'],
                    cmap='coolwarm', s=15, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Cycle Index')
    ax.set_xlabel('Mean Temperature (°C)'); ax.set_ylabel('SOH (%)')
    ax.set_title('Temp vs SOH'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sc = ax.scatter(df_all['voltage_mean'], df_all['SOH'], c=df_all['RUL'],
                    cmap='viridis', s=15, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='RUL')
    ax.set_xlabel('Mean Voltage (V)'); ax.set_ylabel('SOH (%)')
    ax.set_title('Voltage vs SOH (colored by RUL)'); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    feature_cols = [c for c in df_all.columns
                    if c not in ['battery_id', 'cycle_index', 'SOH', 'RUL', 'nominal_capacity']]
    corr = df_all[feature_cols + ['SOH', 'RUL']].corr()[['SOH', 'RUL']].drop(['SOH', 'RUL'])
    sns.heatmap(corr.T, ax=ax, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
    ax.set_title('Feature Correlation with SOH & RUL')

    plt.tight_layout()
    plt.savefig(PLOT_EDA, dpi=150, bbox_inches='tight')
    plt.show()
    print("EDA plot saved.")


# ── Cell 12: Sequences + Scaling ──────────────────────────────────────────

def build_sequences(df_group, feature_cols, seq_len=SEQUENCE_LEN):
    """
    Build sliding-window sequences from a single battery's cycle data.
    Returns X (3D), y_soh (1D), y_rul (1D).
    Original notebook cell 12 — zero changes.
    """
    data = df_group[feature_cols].values
    soh  = df_group[TARGET_SOH].values
    rul  = df_group[TARGET_RUL].values

    X, y_soh, y_rul = [], [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y_soh.append(soh[i])
        y_rul.append(rul[i])

    return (np.array(X, dtype=np.float32),
            np.array(y_soh, dtype=np.float32),
            np.array(y_rul, dtype=np.float32))


def prepare_data(df_all):
    """
    Battery-aware split → scale → build sequences → val split.
    Original notebook cell 12 — zero changes.

    Returns dict of all arrays + fitted scalers.
    """
    battery_ids = df_all['battery_id'].unique()
    np.random.shuffle(battery_ids)
    n_test = max(1, int(len(battery_ids) * 0.2))
    test_batteries  = battery_ids[:n_test]
    train_batteries = battery_ids[n_test:]

    print(f"Train batteries: {list(train_batteries)}")
    print(f"Test  batteries: {list(test_batteries)}")

    df_train_raw = df_all[df_all['battery_id'].isin(train_batteries)].copy()
    df_test_raw  = df_all[df_all['battery_id'].isin(test_batteries)].copy()

    scaler_X   = MinMaxScaler()
    scaler_soh = MinMaxScaler()
    scaler_rul = MinMaxScaler()

    df_train_raw[FEATURE_COLS] = scaler_X.fit_transform(df_train_raw[FEATURE_COLS].fillna(0))
    df_test_raw[FEATURE_COLS]  = scaler_X.transform(df_test_raw[FEATURE_COLS].fillna(0))

    df_train_raw[[TARGET_SOH]] = scaler_soh.fit_transform(df_train_raw[[TARGET_SOH]])
    df_test_raw[[TARGET_SOH]]  = scaler_soh.transform(df_test_raw[[TARGET_SOH]])

    df_train_raw[[TARGET_RUL]] = scaler_rul.fit_transform(df_train_raw[[TARGET_RUL]])
    df_test_raw[[TARGET_RUL]]  = scaler_rul.transform(df_test_raw[[TARGET_RUL]])

    X_train_list, y_soh_train_list, y_rul_train_list = [], [], []
    for bat_id in train_batteries:
        grp = df_train_raw[df_train_raw['battery_id'] == bat_id].sort_values('cycle_index')
        if len(grp) > SEQUENCE_LEN + 1:
            Xs, ys, yr = build_sequences(grp, FEATURE_COLS, SEQUENCE_LEN)
            X_train_list.append(Xs); y_soh_train_list.append(ys); y_rul_train_list.append(yr)

    X_test_list, y_soh_test_list, y_rul_test_list = [], [], []
    for bat_id in test_batteries:
        grp = df_test_raw[df_test_raw['battery_id'] == bat_id].sort_values('cycle_index')
        if len(grp) > SEQUENCE_LEN + 1:
            Xs, ys, yr = build_sequences(grp, FEATURE_COLS, SEQUENCE_LEN)
            X_test_list.append(Xs); y_soh_test_list.append(ys); y_rul_test_list.append(yr)

    X_train     = np.concatenate(X_train_list)
    y_soh_train = np.concatenate(y_soh_train_list)
    y_rul_train = np.concatenate(y_rul_train_list)
    X_test      = np.concatenate(X_test_list)
    y_soh_test  = np.concatenate(y_soh_test_list)
    y_rul_test  = np.concatenate(y_rul_test_list)

    X_train, X_val, y_soh_train, y_soh_val, y_rul_train, y_rul_val = train_test_split(
        X_train, y_soh_train, y_rul_train, test_size=0.15, random_state=42
    )

    print(f"\nX_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Features: {len(FEATURE_COLS)}, Sequence length: {SEQUENCE_LEN}")

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_soh_train=y_soh_train, y_soh_val=y_soh_val, y_soh_test=y_soh_test,
        y_rul_train=y_rul_train, y_rul_val=y_rul_val, y_rul_test=y_rul_test,
        scaler_X=scaler_X, scaler_soh=scaler_soh, scaler_rul=scaler_rul,
        train_batteries=train_batteries, test_batteries=test_batteries,
    )
