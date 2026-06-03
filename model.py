"""
model.py
Original notebook cells 14 (architecture) + 16 (training) +
18 (history plots) + 20 + 21 (evaluation).
Zero changes to logic.
"""

from imports import *
from config import (
    FEATURE_COLS, SEQUENCE_LEN, EOL_THRESHOLD,
    LSTM_UNITS, CNN_FILTERS, DROPOUT,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    BEST_MODEL_PATH, PLOT_HISTORY, PLOT_EVAL,
)


# ── Cell 14: Architecture ─────────────────────────────────────────────────

def build_cnn_lstm_model(seq_len, n_features, lstm_units=LSTM_UNITS,
                         cnn_filters=CNN_FILTERS, dropout=DROPOUT):
    inp = Input(shape=(seq_len, n_features), name='cycle_sequence')

    x = Conv1D(cnn_filters, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(1e-4), name='conv1')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv1D(cnn_filters * 2, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(1e-4), name='conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv1D(cnn_filters * 4, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(1e-4), name='conv3')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True,
                           dropout=dropout, recurrent_dropout=0.1), name='bilstm1')(x)
    x = Bidirectional(LSTM(lstm_units // 2, return_sequences=True,
                           dropout=dropout, recurrent_dropout=0.1), name='bilstm2')(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=lstm_units // 4, name='attention')(x, x)
    x    = Add()([x, attn])
    x    = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    shared = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(dropout)(shared)
    shared = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(shared)
    shared = Dropout(dropout)(shared)

    soh_branch = Dense(64, activation='relu', name='soh_dense')(shared)
    soh_branch = Dropout(dropout / 2)(soh_branch)
    soh_out    = Dense(1, activation='sigmoid', name='soh')(soh_branch)

    rul_branch = Dense(64, activation='relu', name='rul_dense')(shared)
    rul_branch = Dropout(dropout / 2)(rul_branch)
    rul_out    = Dense(1, activation='sigmoid', name='rul')(rul_branch)

    return Model(inputs=inp, outputs={'soh': soh_out, 'rul': rul_out})


# ── Cell 16: Training ─────────────────────────────────────────────────────

def train_model(model, X_train, y_soh_train, y_rul_train,
                X_val, y_soh_val, y_rul_val):
    huber = tf.keras.losses.Huber(delta=0.1)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={'soh': huber, 'rul': huber},
        loss_weights={'soh': 1.0, 'rul': 0.8},
        metrics={'soh': ['mae', 'mse'], 'rul': ['mae', 'mse']},
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    history = model.fit(
        X_train,
        {'soh': y_soh_train, 'rul': y_rul_train},
        validation_data=(X_val, {'soh': y_soh_val, 'rul': y_rul_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    print("\n✅ Training complete.")
    return history


# ── Cell 18: History plots ────────────────────────────────────────────────

def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['loss'],         label='Train Loss',    color='steelblue')
    axes[0].plot(history.history['val_loss'],      label='Val Loss',      color='orange')
    axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['soh_mae'],       label='Train SOH MAE', color='steelblue')
    axes[1].plot(history.history['val_soh_mae'],   label='Val SOH MAE',   color='orange')
    axes[1].set_title('SOH MAE'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history.history['rul_mae'],       label='Train RUL MAE', color='steelblue')
    axes[2].plot(history.history['val_rul_mae'],   label='Val RUL MAE',   color='orange')
    axes[2].set_title('RUL MAE'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_HISTORY, dpi=150, bbox_inches='tight')
    plt.show()


# ── Cells 20 + 21: Evaluation ─────────────────────────────────────────────

def regression_metrics(true, pred, name):
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2   = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
    print(f"\n{'='*40}")
    print(f"  {name} Regression Metrics")
    print(f"{'='*40}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return mae, rmse, r2, mape


def evaluate_model(model, X_test, y_soh_test, y_rul_test, scaler_soh, scaler_rul):
    preds = model.predict(X_test, batch_size=64, verbose=0)
    soh_pred_scaled = preds['soh'].flatten()
    rul_pred_scaled = preds['rul'].flatten()

    soh_pred = scaler_soh.inverse_transform(soh_pred_scaled.reshape(-1, 1)).flatten()
    soh_true = scaler_soh.inverse_transform(y_soh_test.reshape(-1, 1)).flatten()
    rul_pred = scaler_rul.inverse_transform(rul_pred_scaled.reshape(-1, 1)).flatten()
    rul_true = scaler_rul.inverse_transform(y_rul_test.reshape(-1, 1)).flatten()

    soh_metrics = regression_metrics(soh_true, soh_pred, 'SOH')
    rul_metrics = regression_metrics(rul_true, rul_pred, 'RUL')

    print(f"\n{'='*40}")
    print("  SOH Classification (Healthy ≥ 80% vs Degraded < 80%)")
    print(f"{'='*40}")
    soh_true_cls = (soh_true >= EOL_THRESHOLD).astype(int)
    soh_pred_cls = (soh_pred >= EOL_THRESHOLD).astype(int)
    acc_soh = accuracy_score(soh_true_cls, soh_pred_cls)
    f1_soh  = f1_score(soh_true_cls, soh_pred_cls, average='weighted')
    print(f"  Accuracy : {acc_soh:.4f} ({acc_soh*100:.2f}%)")
    print(f"  F1 Score : {f1_soh:.4f}")
    print(f"\n{classification_report(soh_true_cls, soh_pred_cls, target_names=['Degraded','Healthy'])}")

    print(f"\n{'='*40}")
    print("  RUL Classification (Urgent RUL < 20 cycles)")
    print(f"{'='*40}")
    rul_true_cls = (rul_true >= 20).astype(int)
    rul_pred_cls = (rul_pred >= 20).astype(int)
    acc_rul = accuracy_score(rul_true_cls, rul_pred_cls)
    f1_rul  = f1_score(rul_true_cls, rul_pred_cls, average='weighted')
    print(f"  Accuracy : {acc_rul:.4f} ({acc_rul*100:.2f}%)")
    print(f"  F1 Score : {f1_rul:.4f}")
    print(f"\n{classification_report(rul_true_cls, rul_pred_cls, target_names=['Urgent','Safe'])}")

    # ── Full results figure (original cell 21) ─────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Model Evaluation Results — NASA Battery CNN-LSTM', fontsize=16, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(soh_true, soh_pred, s=15, alpha=0.5, color='steelblue')
    lim = [min(soh_true.min(), soh_pred.min()) - 2, max(soh_true.max(), soh_pred.max()) + 2]
    ax1.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True SOH (%)'); ax1.set_ylabel('Predicted SOH (%)')
    ax1.set_title(f'SOH: True vs Predicted\nR²={soh_metrics[2]:.3f}, RMSE={soh_metrics[1]:.3f}')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(rul_true, rul_pred, s=15, alpha=0.5, color='darkorange')
    lim2 = [min(rul_true.min(), rul_pred.min()) - 2, max(rul_true.max(), rul_pred.max()) + 2]
    ax2.plot(lim2, lim2, 'r--', lw=2, label='Perfect prediction')
    ax2.set_xlabel('True RUL (cycles)'); ax2.set_ylabel('Predicted RUL (cycles)')
    ax2.set_title(f'RUL: True vs Predicted\nR²={rul_metrics[2]:.3f}, RMSE={rul_metrics[1]:.3f}')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    metrics_labels = ['SOH MAE', 'SOH RMSE', 'RUL MAE', 'RUL RMSE']
    metrics_vals   = [soh_metrics[0], soh_metrics[1], rul_metrics[0], rul_metrics[1]]
    bars = ax3.bar(metrics_labels, metrics_vals,
                   color=['steelblue', 'steelblue', 'darkorange', 'darkorange'], alpha=0.8)
    for bar, val in zip(bars, metrics_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax3.set_title('Error Metrics'); ax3.set_ylabel('Error')
    ax3.grid(True, alpha=0.3, axis='y'); ax3.tick_params(axis='x', rotation=20)

    ax4 = fig.add_subplot(gs[1, :])
    sort_idx = np.argsort(soh_true)
    ax4.plot(np.arange(len(soh_true)), soh_true[sort_idx], label='True SOH', color='steelblue', lw=1.5)
    ax4.plot(np.arange(len(soh_pred)), soh_pred[sort_idx], label='Predicted SOH',
             color='orange', lw=1.5, linestyle='--')
    ax4.fill_between(np.arange(len(soh_true)), soh_true[sort_idx], soh_pred[sort_idx],
                     alpha=0.15, color='red')
    ax4.axhline(EOL_THRESHOLD, color='red', lw=1.5, linestyle=':', label='EOL Threshold (80%)')
    ax4.set_xlabel('Sorted Sample Index'); ax4.set_ylabel('SOH (%)')
    ax4.set_title('SOH Prediction vs Ground Truth (Test Set)')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    cm_soh = confusion_matrix(soh_true_cls, soh_pred_cls)
    ConfusionMatrixDisplay(cm_soh, display_labels=['Degraded', 'Healthy']).plot(
        ax=ax5, colorbar=False, cmap='Blues')
    ax5.set_title(f'SOH Confusion Matrix\nAcc={acc_soh*100:.1f}%, F1={f1_soh:.3f}')

    ax6 = fig.add_subplot(gs[2, 1])
    cm_rul = confusion_matrix(rul_true_cls, rul_pred_cls)
    ConfusionMatrixDisplay(cm_rul, display_labels=['Urgent', 'Safe']).plot(
        ax=ax6, colorbar=False, cmap='Oranges')
    ax6.set_title(f'RUL Urgency Confusion Matrix\nAcc={acc_rul*100:.1f}%, F1={f1_rul:.3f}')

    ax7 = fig.add_subplot(gs[2, 2])
    soh_residuals = soh_true - soh_pred
    ax7.hist(soh_residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax7.axvline(0, color='red', lw=2, linestyle='--')
    ax7.set_xlabel('SOH Residual (True − Predicted)'); ax7.set_ylabel('Count')
    ax7.set_title(f'SOH Residual Distribution\nMean={soh_residuals.mean():.3f}, Std={soh_residuals.std():.3f}')
    ax7.grid(True, alpha=0.3)

    plt.savefig(PLOT_EVAL, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Evaluation plots saved to {PLOT_EVAL}")

    return dict(soh_metrics=soh_metrics, rul_metrics=rul_metrics,
                acc_soh=acc_soh, f1_soh=f1_soh, acc_rul=acc_rul, f1_rul=f1_rul,
                soh_true=soh_true, soh_pred=soh_pred,
                rul_true=rul_true, rul_pred=rul_pred)
