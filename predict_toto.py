"""
ToTo Lottery Deep Learning Prediction Model
Uses advanced neural network architectures to predict next draw numbers
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ToTo Deep Learning Prediction Model")
print("=" * 80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"toto_model_best_{RUN_ID}.keras")

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        print(f"[GPU] GPU acceleration enabled for faster training!")
    except RuntimeError as e:
        print(f"[GPU] Error configuring GPU: {e}")
else:
    print("[CPU] No GPU detected - training will use CPU (slower)")
print()

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("Step 1: Loading ToTo Historical Data...")
df = pd.read_csv('ToTo-12_Feb_2026.csv')
print(f"[OK] Loaded {len(df)} historical draws")
print(f"[OK] Date range: {df['Date'].iloc[-1]} to {df['Date'].iloc[0]}")
print(f"[OK] Columns: {list(df.columns)}")
print()

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("Step 2: Feature Engineering...")

# Select features as specified by user
feature_columns = [
    'Win_1', 'Win_2', 'Win_3', 'Win_4', 'Win_5', 'Win_6', 'Addl No.',
    'Sum', 'Average', 'Low/High', 'Odd/Even',
    '1-10', '11-20', '21-30', '31-40', '41-50'
]

# Extract numerical data from Low/High and Odd/Even columns
df['Low_Count'] = df['Low/High'].apply(lambda x: int(str(x).split(' / ')[0]) if pd.notna(x) else 3)
df['High_Count'] = df['Low/High'].apply(lambda x: int(str(x).split(' / ')[1]) if pd.notna(x) else 3)
df['Odd_Count'] = df['Odd/Even'].apply(lambda x: int(str(x).split(' / ')[0]) if pd.notna(x) else 3)
df['Even_Count'] = df['Odd/Even'].apply(lambda x: int(str(x).split(' / ')[1]) if pd.notna(x) else 3)

# Date-based seasonality features
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['DayOfWeek'] = df['Date'].dt.dayofweek.fillna(0).astype(int)
df['Month'] = df['Date'].dt.month.fillna(1).astype(int)
df['DOW_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
df['DOW_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

# Reverse the dataframe to chronological order (oldest to newest)
df = df.iloc[::-1].reset_index(drop=True)

# Lag features (previous draw metrics)
lag_base_cols = [
    'Sum', 'Average', 'Low_Count', 'High_Count', 'Odd_Count', 'Even_Count',
    '1-10', '11-20', '21-30', '31-40', '41-50'
]
for col in lag_base_cols:
    prev_col = f"Prev_{col}"
    df[prev_col] = df[col].shift(1)
    df[prev_col] = df[prev_col].fillna(df[col])

# Rolling frequency features for numbers 1-49 over the last N draws
FREQ_WINDOW = 20
win_cols = ['Win_1', 'Win_2', 'Win_3', 'Win_4', 'Win_5', 'Win_6']
wins_matrix = df[win_cols].values.astype(int)
freq_features = np.zeros((len(df), 49), dtype=float)
for i in range(len(df)):
    start = max(0, i - FREQ_WINDOW)
    window_vals = wins_matrix[start:i].reshape(-1)
    if window_vals.size == 0:
        continue
    counts = np.bincount(window_vals, minlength=50)[1:50]
    freq_features[i] = counts / window_vals.size

for num in range(1, 50):
    df[f"Freq_{num}"] = freq_features[:, num - 1]

# Time-since-last-seen features for numbers 1-49 (based on past draws)
last_seen = np.full(50, -1, dtype=int)
since_last = np.zeros((len(df), 49), dtype=float)
for i in range(len(df)):
    for num in range(1, 50):
        if last_seen[num] == -1:
            since_last[i, num - 1] = i + 1
        else:
            since_last[i, num - 1] = i - last_seen[num]
    for num in wins_matrix[i]:
        last_seen[num] = i

for num in range(1, 50):
    df[f"Since_{num}"] = since_last[:, num - 1]

# Create feature matrix
features_numeric = [
    'Win_1', 'Win_2', 'Win_3', 'Win_4', 'Win_5', 'Win_6', 'Addl No.',
    'Sum', 'Average', 'Low_Count', 'High_Count', 'Odd_Count', 'Even_Count',
    '1-10', '11-20', '21-30', '31-40', '41-50',
    'DOW_sin', 'DOW_cos', 'Month_sin', 'Month_cos',
    'Prev_Sum', 'Prev_Average', 'Prev_Low_Count', 'Prev_High_Count',
    'Prev_Odd_Count', 'Prev_Even_Count', 'Prev_1-10', 'Prev_11-20',
    'Prev_21-30', 'Prev_31-40', 'Prev_41-50'
]

freq_columns = [f"Freq_{num}" for num in range(1, 50)]
features_numeric.extend(freq_columns)

since_columns = [f"Since_{num}" for num in range(1, 50)]
features_numeric.extend(since_columns)

X = df[features_numeric].values

# Targets: multi-label for winning numbers + one-hot for additional number
win_multi = np.zeros((len(df), 49), dtype=int)
for i, row in enumerate(df[win_cols].values.astype(int)):
    win_multi[i, row - 1] = 1
addl_onehot = np.zeros((len(df), 49), dtype=int)
addl_indices = df['Addl No.'].values.astype(int) - 1
addl_onehot[np.arange(len(df)), addl_indices] = 1

print(f"[OK] Feature matrix shape: {X.shape}")
print(f"[OK] Target matrix shape: wins={win_multi.shape}, addl={addl_onehot.shape}")
print(f"[OK] Features: {features_numeric}")
print()

# ============================================================================
# 3. DATA PREP + OPTIONAL HYPERPARAMETER SWEEP
# ============================================================================

# Hyperparameter sweep config
RUN_HYPERPARAM_SWEEP = True
SWEEP_MODE = "full"  # "quick" or "full"
SWEEP_SEQUENCE_LENGTHS = [7, 10, 15, 20, 30]
SWEEP_L2_WEIGHTS = [0.0, 1e-5]
SWEEP_DROPOUT_RATES = [0.2, 0.3]
SWEEP_EPOCHS = 60
SWEEP_PATIENCE = 10
SWEEP_WALK_FORWARD_SPLITS = 5
SWEEP_VAL_FRACTION = 0.1
SWEEP_TRAIN_START = 0.6
SWEEP_SAMPLE_FRACTION = 1.0  # use only the most recent fraction of sweep data
SWEEP_BATCH_SIZE = 32
FINAL_HOLDOUT_FRACTION = 0.1

if SWEEP_MODE == "quick":
    SWEEP_SEQUENCE_LENGTHS = [15, 30]
    SWEEP_L2_WEIGHTS = [0.0]
    SWEEP_DROPOUT_RATES = [0.2, 0.3]
    SWEEP_EPOCHS = 35
    SWEEP_PATIENCE = 6
    SWEEP_WALK_FORWARD_SPLITS = 3
    SWEEP_VAL_FRACTION = 0.08
    SWEEP_TRAIN_START = 0.7
    SWEEP_SAMPLE_FRACTION = 0.6
    SWEEP_BATCH_SIZE = 48

FEATURE_SCALER = "standard"  # "standard" or "minmax"
RECURRENT_DROPOUT = 0.0

# Defaults if sweep disabled
SEQUENCE_LENGTH = 15  # Look back at last 15 draws
L2_WEIGHT = 1e-5
DROPOUT_RATE = 0.3

TRAIN_BATCH_SIZE = 32

def create_sequences_multi(X, y_list, seq_length):
    """Create sequences for LSTM input with multiple targets."""
    X_seq = []
    y_seq_list = [[] for _ in y_list]
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        for idx, y in enumerate(y_list):
            y_seq_list[idx].append(y[i+seq_length])
    return np.array(X_seq), [np.array(lst) for lst in y_seq_list]

def get_feature_scaler():
    if FEATURE_SCALER == "standard":
        return StandardScaler()
    return MinMaxScaler()

def scale_with_indices(X_sequences, y_list, train_idx, val_idx):
    X_train = X_sequences[train_idx]
    X_val = X_sequences[val_idx]
    y_train_list = [y[train_idx] for y in y_list]
    y_val_list = [y[val_idx] for y in y_list]

    scaler_X = get_feature_scaler()
    n_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    scaler_X.fit(X_train_reshaped)
    X_train_scaled = scaler_X.transform(X_train_reshaped).reshape(n_train, n_timesteps, n_features)

    n_val = X_val.shape[0]
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(n_val, n_timesteps, n_features)
    return X_train_scaled, X_val_scaled, y_train_list, y_val_list, scaler_X

def scale_train_val(X_sequences, y_list, split_idx):
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, len(X_sequences))
    return scale_with_indices(X_sequences, y_list, train_idx, val_idx)

def transform_sequence(scaler_X, sequence):
    n_timesteps, n_features = sequence.shape
    reshaped = sequence.reshape(-1, n_features)
    scaled = scaler_X.transform(reshaped)
    return scaled.reshape(1, n_timesteps, n_features)

def transform_sequences(scaler_X, sequences):
    n_samples, n_timesteps, n_features = sequences.shape
    reshaped = sequences.reshape(-1, n_features)
    scaled = scaler_X.transform(reshaped)
    return scaled.reshape(n_samples, n_timesteps, n_features)

def create_attention_layer(inputs):
    """Self-attention mechanism"""
    attention = layers.Dense(inputs.shape[-1], activation='tanh')(inputs)
    attention = layers.Dense(1, activation='softmax')(attention)
    weighted = layers.Multiply()([inputs, attention])
    return weighted

def build_model(seq_length, n_features, l2_weight, dropout_rate):
    dense_dropout = min(0.5, dropout_rate + 0.1)

    input_layer = keras.Input(shape=(seq_length, n_features), name='sequence_input')

    lstm1 = layers.Bidirectional(
        layers.LSTM(
            256,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=RECURRENT_DROPOUT,
            kernel_regularizer=regularizers.l2(l2_weight),
            recurrent_regularizer=regularizers.l2(l2_weight),
        ),
        name='bidirectional_lstm_1'
    )(input_layer)

    lstm2 = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=RECURRENT_DROPOUT,
            kernel_regularizer=regularizers.l2(l2_weight),
            recurrent_regularizer=regularizers.l2(l2_weight),
        ),
        name='bidirectional_lstm_2'
    )(lstm1)

    attention_output = create_attention_layer(lstm2)
    pooled = layers.GlobalAveragePooling1D()(attention_output)

    dense1 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_weight))(pooled)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(dense_dropout)(dense1)

    dense2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_weight))(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(dense_dropout)(dense2)

    dense3 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_weight))(dense2)
    dense3 = layers.Dropout(dropout_rate)(dense3)

    win_output = layers.Dense(49, activation='sigmoid', name='win_output')(dense3)
    addl_output = layers.Dense(49, activation='softmax', name='addl_output')(dense3)

    return models.Model(
        inputs=input_layer,
        outputs=[win_output, addl_output],
        name='ToTo_Predictor'
    )

def get_walk_forward_splits(n_samples):
    val_size = max(1, int(n_samples * SWEEP_VAL_FRACTION))
    train_start = max(val_size, int(n_samples * SWEEP_TRAIN_START))
    splits = []
    for i in range(SWEEP_WALK_FORWARD_SPLITS):
        train_end = train_start + i * val_size
        val_start = train_end
        val_end = min(n_samples, val_start + val_size)
        if val_end <= val_start or train_end <= 0:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
    return splits

def run_hyperparam_sweep(X, y_list, holdout_fraction):
    results = []
    total = len(SWEEP_SEQUENCE_LENGTHS) * len(SWEEP_L2_WEIGHTS) * len(SWEEP_DROPOUT_RATES)
    sweep_idx = 0

    for seq_len in SWEEP_SEQUENCE_LENGTHS:
        X_sequences, y_sequences_list = create_sequences_multi(X, y_list, seq_len)
        sweep_samples = max(1, int(len(X_sequences) * (1 - holdout_fraction)))
        X_sequences = X_sequences[:sweep_samples]
        y_sequences_list = [y_seq[:sweep_samples] for y_seq in y_sequences_list]

        if SWEEP_SAMPLE_FRACTION < 1.0:
            sample_len = max(1, int(len(X_sequences) * SWEEP_SAMPLE_FRACTION))
            start_idx = max(0, len(X_sequences) - sample_len)
            X_sequences = X_sequences[start_idx:]
            y_sequences_list = [y_seq[start_idx:] for y_seq in y_sequences_list]
        n_features = X_sequences.shape[-1]
        splits = get_walk_forward_splits(len(X_sequences))

        for l2_weight in SWEEP_L2_WEIGHTS:
            for dropout_rate in SWEEP_DROPOUT_RATES:
                sweep_idx += 1
                print(f"[SWEEP {sweep_idx}/{total}] seq_len={seq_len}, l2={l2_weight}, dropout={dropout_rate}")

                fold_losses = []
                for train_idx, val_idx in splits:
                    X_train, X_val, y_train_list, y_val_list, _ = scale_with_indices(
                        X_sequences, y_sequences_list, train_idx, val_idx
                    )

                    tf.keras.backend.clear_session()
                    model = build_model(seq_len, n_features, l2_weight, dropout_rate)
                    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
                    model.compile(
                        optimizer=optimizer,
                        loss={
                            'win_output': 'binary_crossentropy',
                            'addl_output': 'categorical_crossentropy',
                        },
                        loss_weights={'win_output': 1.0, 'addl_output': 0.5},
                        metrics={
                            'win_output': [keras.metrics.BinaryAccuracy(name='win_acc')],
                            'addl_output': [keras.metrics.CategoricalAccuracy(name='addl_acc')],
                        },
                    )

                    sweep_callbacks = [
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=SWEEP_PATIENCE,
                            restore_best_weights=True,
                            verbose=0
                        ),
                        keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=max(5, SWEEP_PATIENCE // 2),
                            min_lr=0.00001,
                            verbose=0
                        ),
                    ]

                    history = model.fit(
                        X_train,
                        {'win_output': y_train_list[0], 'addl_output': y_train_list[1]},
                        validation_data=(
                            X_val,
                            {'win_output': y_val_list[0], 'addl_output': y_val_list[1]},
                        ),
                        epochs=SWEEP_EPOCHS,
                        batch_size=SWEEP_BATCH_SIZE,
                        callbacks=sweep_callbacks,
                        verbose=0,
                        shuffle=False
                    )

                    fold_losses.append(float(np.min(history.history['val_loss'])))

                best_val = float(np.mean(fold_losses)) if fold_losses else float('inf')
                results.append({
                    'seq_len': seq_len,
                    'l2': l2_weight,
                    'dropout': dropout_rate,
                    'val_loss': best_val,
                })
                print(f"[SWEEP] mean_val_loss={best_val:.6f}")

    best = min(results, key=lambda r: r['val_loss'])
    return best, results

step = 3
if RUN_HYPERPARAM_SWEEP:
    print(f"Step {step}: Hyperparameter Sweep (sequence length, L2, dropout)...")
    best_params, _ = run_hyperparam_sweep(X, [win_multi, addl_onehot], FINAL_HOLDOUT_FRACTION)
    SEQUENCE_LENGTH = best_params['seq_len']
    L2_WEIGHT = best_params['l2']
    DROPOUT_RATE = best_params['dropout']
    print(f"[SWEEP] Best params: sequence_length={SEQUENCE_LENGTH}, l2={L2_WEIGHT}, dropout={DROPOUT_RATE}")
    print(f"[SWEEP] Mode: {SWEEP_MODE}, sample_fraction={SWEEP_SAMPLE_FRACTION}, folds={SWEEP_WALK_FORWARD_SPLITS}")
    print()
    step += 1

print(f"Step {step}: Creating Sequential Windows for LSTM...")
X_sequences, y_sequences_list = create_sequences_multi(X, [win_multi, addl_onehot], SEQUENCE_LENGTH)
print(f"[OK] Created {len(X_sequences)} sequences")
print(f"[OK] Sequence shape: {X_sequences.shape}")
print(f"[OK] Each sequence contains {SEQUENCE_LENGTH} historical draws")
print()
step += 1

print(f"Step {step}: Normalizing Features...")
holdout_idx = max(1, int(len(X_sequences) * (1 - FINAL_HOLDOUT_FRACTION)))
X_sequences_train = X_sequences[:holdout_idx]
y_sequences_train_list = [y_seq[:holdout_idx] for y_seq in y_sequences_list]
X_sequences_holdout = X_sequences[holdout_idx:]
y_sequences_holdout_list = [y_seq[holdout_idx:] for y_seq in y_sequences_list]

split_idx = int(len(X_sequences_train) * 0.8)
X_train, X_val, y_train_list, y_val_list, scaler_X = scale_train_val(
    X_sequences_train, y_sequences_train_list, split_idx
)
n_samples, n_timesteps, n_features = X_sequences.shape
print(f"[OK] Features scaled using {FEATURE_SCALER}")
print()
step += 1

print(f"Step {step}: Splitting Data...")
print(f"[OK] Training samples: {len(X_train)}")
print(f"[OK] Validation samples: {len(X_val)}")
print(f"[OK] Holdout samples: {len(X_sequences_holdout)}")
print()
step += 1

print(f"Step {step}: Building Deep Learning Model Architecture...")
print()
model = build_model(SEQUENCE_LENGTH, n_features, L2_WEIGHT, DROPOUT_RATE)
print(model.summary())
print()
step += 1

# ============================================================================
# 7. COMPILE MODEL
# ============================================================================

print(f"Step {step}: Compiling Model...")

# Use fixed learning rate - ReduceLROnPlateau callback will adjust it during training.
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss={
        'win_output': 'binary_crossentropy',
        'addl_output': 'categorical_crossentropy',
    },
    loss_weights={'win_output': 1.0, 'addl_output': 0.5},
    metrics={
        'win_output': [keras.metrics.BinaryAccuracy(name='win_acc')],
        'addl_output': [keras.metrics.CategoricalAccuracy(name='addl_acc')],
    },
)

print("[OK] Model compiled successfully")
print()
step += 1

# ============================================================================
# 8. CALLBACKS FOR TRAINING
# ============================================================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=0.00001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_CHECKPOINT_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================================
# 9. TRAIN MODEL
# ============================================================================

print(f"Step {step}: Training Model...")
print("=" * 80)
print(f"[INFO] Checkpoint path: {MODEL_CHECKPOINT_PATH}")

EPOCHS = 300

history = model.fit(
    X_train,
    {'win_output': y_train_list[0], 'addl_output': y_train_list[1]},
    validation_data=(
        X_val,
        {'win_output': y_val_list[0], 'addl_output': y_val_list[1]},
    ),
    epochs=EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    callbacks=callbacks,
    shuffle=False,
    verbose=1
)

print()
print("=" * 80)
print("[OK] Training completed!")
print()
step += 1

# ============================================================================
# 10. EVALUATE MODEL
# ============================================================================

print(f"Step {step}: Evaluating Model Performance...")

train_metrics = model.evaluate(
    X_train,
    {'win_output': y_train_list[0], 'addl_output': y_train_list[1]},
    verbose=0,
    return_dict=True
)
val_metrics = model.evaluate(
    X_val,
    {'win_output': y_val_list[0], 'addl_output': y_val_list[1]},
    verbose=0,
    return_dict=True
)

holdout_metrics = None
if len(X_sequences_holdout) > 0:
    X_holdout = transform_sequences(scaler_X, X_sequences_holdout)
    holdout_metrics = model.evaluate(
        X_holdout,
        {'win_output': y_sequences_holdout_list[0], 'addl_output': y_sequences_holdout_list[1]},
        verbose=0,
        return_dict=True
    )

print(f"Training Loss: {train_metrics['loss']:.6f}")
print(f"Training Win Acc: {train_metrics.get('win_output_win_acc', 0.0):.6f}")
print(f"Training Addl Acc: {train_metrics.get('addl_output_addl_acc', 0.0):.6f}")
print(f"Validation Loss: {val_metrics['loss']:.6f}")
print(f"Validation Win Acc: {val_metrics.get('win_output_win_acc', 0.0):.6f}")
print(f"Validation Addl Acc: {val_metrics.get('addl_output_addl_acc', 0.0):.6f}")
if holdout_metrics:
    print(f"Holdout Loss: {holdout_metrics['loss']:.6f}")
    print(f"Holdout Win Acc: {holdout_metrics.get('win_output_win_acc', 0.0):.6f}")
    print(f"Holdout Addl Acc: {holdout_metrics.get('addl_output_addl_acc', 0.0):.6f}")
print()
step += 1

# ============================================================================
# 11. MAKE PREDICTIONS FOR NEXT DRAW
# ============================================================================

print(f"Step {step}: Generating Predictions for Next Draw...")
print("=" * 80)

# Use the most recent sequence for prediction (scale using training scaler)
last_sequence_raw = X_sequences[-1:]
last_sequence = transform_sequence(scaler_X, last_sequence_raw[0])

# Make prediction
win_probs, addl_probs = model.predict(last_sequence, verbose=0)
win_probs = win_probs[0]
addl_probs = addl_probs[0]

# Pick top-6 winning numbers by probability
win_indices = np.argsort(win_probs)[-6:][::-1]
win_numbers = sorted((win_indices + 1).tolist())

# Pick the most likely additional number not in winning set
addl_indices = np.argsort(addl_probs)[::-1]
additional_number = None
for idx in addl_indices:
    num = int(idx + 1)
    if num not in win_numbers:
        additional_number = num
        break
if additional_number is None:
    additional_number = int(addl_indices[0] + 1)

unique_win = win_numbers

print()
print(">>> PREDICTED NEXT DRAW NUMBERS <<<")
print("=" * 80)
print(f"Win Numbers: {unique_win}")
print(f"Additional Number: {additional_number}")
print()
print(f"Full Prediction: {unique_win[0]}, {unique_win[1]}, {unique_win[2]}, {unique_win[3]}, {unique_win[4]}, {unique_win[5]} + {additional_number}")
print("=" * 80)
print()
step += 1

# Calculate some statistics
recent_draws = df[['Win_1', 'Win_2', 'Win_3', 'Win_4', 'Win_5', 'Win_6']].iloc[-20:].values.flatten()
avg_recent = np.mean(recent_draws)
predicted_avg = np.mean(unique_win)

print("Prediction Analysis:")
print(f"  • Predicted Sum: {sum(unique_win)}")
print(f"  • Predicted Average: {predicted_avg:.2f}")
print(f"  • Recent 20 Draws Average: {avg_recent:.2f}")
print(f"  • Low Numbers (1-24): {sum(1 for x in unique_win if x <= 24)}")
print(f"  • High Numbers (25-49): {sum(1 for x in unique_win if x > 24)}")
print(f"  • Odd Numbers: {sum(1 for x in unique_win if x % 2 == 1)}")
print(f"  • Even Numbers: {sum(1 for x in unique_win if x % 2 == 0)}")
print()

# ============================================================================
# 12. SAVE RESULTS TO MARKDOWN REPORT
# ============================================================================

print(f"Step {step}: Generating Prediction Report...")

report_date = datetime.now().strftime('%d_%b_%Y')
report_filename = f'Predict_ToTo_{report_date}.md'

# Use UTF-8 to avoid Windows cp1252 encoding errors for symbols like arrows.
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write(f"# ToTo Lottery Prediction Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Model:** Deep Learning LSTM with Attention Mechanism\n\n")
    
    f.write(f"---\n\n")
    f.write(f"## Predicted Next Draw\n\n")
    f.write(f"### Winning Numbers\n")
    f.write(f"```\n")
    f.write(f"{unique_win[0]:2d}  {unique_win[1]:2d}  {unique_win[2]:2d}  {unique_win[3]:2d}  {unique_win[4]:2d}  {unique_win[5]:2d}\n")
    f.write(f"```\n\n")
    f.write(f"### Additional Number\n")
    f.write(f"```\n")
    f.write(f"{additional_number:2d}\n")
    f.write(f"```\n\n")
    
    f.write(f"---\n\n")
    f.write(f"## Statistical Analysis\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Sum | {sum(unique_win)} |\n")
    f.write(f"| Average | {predicted_avg:.2f} |\n")
    f.write(f"| Low/High Ratio | {sum(1 for x in unique_win if x <= 24)} / {sum(1 for x in unique_win if x > 24)} |\n")
    f.write(f"| Odd/Even Ratio | {sum(1 for x in unique_win if x % 2 == 1)} / {sum(1 for x in unique_win if x % 2 == 0)} |\n")
    f.write(f"| Range 1-10 | {sum(1 for x in unique_win if 1 <= x <= 10)} |\n")
    f.write(f"| Range 11-20 | {sum(1 for x in unique_win if 11 <= x <= 20)} |\n")
    f.write(f"| Range 21-30 | {sum(1 for x in unique_win if 21 <= x <= 30)} |\n")
    f.write(f"| Range 31-40 | {sum(1 for x in unique_win if 31 <= x <= 40)} |\n")
    f.write(f"| Range 41-50 | {sum(1 for x in unique_win if 41 <= x <= 49)} |\n\n")
    
    f.write(f"---\n\n")
    f.write(f"## Model Performance\n\n")
    f.write(f"| Dataset | Loss | Win Acc | Addl Acc |\n")
    f.write(f"|---------|------|---------|----------|\n")
    f.write(f"| Training | {train_metrics['loss']:.6f} | {train_metrics.get('win_output_win_acc', 0.0):.6f} | {train_metrics.get('addl_output_addl_acc', 0.0):.6f} |\n")
    f.write(f"| Validation | {val_metrics['loss']:.6f} | {val_metrics.get('win_output_win_acc', 0.0):.6f} | {val_metrics.get('addl_output_addl_acc', 0.0):.6f} |\n")
    if holdout_metrics:
        f.write(f"| Holdout | {holdout_metrics['loss']:.6f} | {holdout_metrics.get('win_output_win_acc', 0.0):.6f} | {holdout_metrics.get('addl_output_addl_acc', 0.0):.6f} |\n")
    f.write(f"\n")

    f.write(f"**Loss:** Binary Cross-Entropy (wins) + Categorical Cross-Entropy (addl)\n\n")
    
    f.write(f"**Architecture:**\n")
    f.write(f"- Bidirectional LSTM (256 + 128 units)\n")
    f.write(f"- Self-Attention Mechanism\n")
    f.write(f"- Dense Layers (256 → 128 → 64)\n")
    f.write(f"- Multi-head Outputs (wins multi-label + addl softmax)\n")
    f.write(f"- Dropout Regularization (tuned)\n")
    f.write(f"- Batch Normalization\n\n")

    f.write(f"**Checkpoint:** {MODEL_CHECKPOINT_PATH}\n\n")

    f.write(f"**Selected Hyperparameters:**\n")
    f.write(f"- Sequence Length: {SEQUENCE_LENGTH}\n")
    f.write(f"- Dropout Rate: {DROPOUT_RATE}\n")
    f.write(f"- L2 Weight: {L2_WEIGHT}\n\n")
    f.write(f"- Hyperparameter Selection: Walk-forward ({SWEEP_WALK_FORWARD_SPLITS} folds)\n")
    f.write(f"- Final Holdout Fraction: {FINAL_HOLDOUT_FRACTION:.2f}\n\n")

    f.write(f"**Feature Engineering:**\n")
    f.write(f"- Feature Scaler: {FEATURE_SCALER}\n")
    f.write(f"- Rolling Frequency Window: {FREQ_WINDOW} draws\n")
    f.write(f"- Lagged Metrics: Enabled\n")
    f.write(f"- Seasonality Features: Day-of-week & Month (sin/cos)\n")
    f.write(f"- Time-since-last-seen Features: Enabled\n")
    f.write(f"- Target: Win multi-label + additional one-hot\n\n")
    
    f.write(f"**Training Details:**\n")
    f.write(f"- Historical Draws Analyzed: {len(df)}\n")
    f.write(f"- Training Sequences: {len(X_train)}\n")
    f.write(f"- Validation Sequences: {len(X_val)}\n")
    f.write(f"- Holdout Sequences: {len(X_sequences_holdout)}\n")
    f.write(f"- Sequence Length: {SEQUENCE_LENGTH} draws\n")
    f.write(f"- Features Used: {len(features_numeric)}\n")
    f.write(f"- Epochs Trained: {len(history.history['loss'])}\n\n")
    
    f.write(f"---\n\n")
    f.write(f"## Disclaimer\n\n")
    f.write(f"> **Important:** This prediction is generated using advanced deep learning algorithms\n")
    f.write(f"> analyzing historical patterns. However, lottery draws are designed to be random,\n")
    f.write(f"> and past performance does not guarantee future results. This model identifies\n")
    f.write(f"> statistical patterns but cannot predict truly random events with certainty.\n")
    f.write(f"> \n")
    f.write(f"> Use this prediction for entertainment and educational purposes only.\n")

print(f"[OK] Report saved to: {report_filename}")
print()

print("=" * 80)
print("=== ToTo Prediction Model Completed Successfully! ===")
print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
