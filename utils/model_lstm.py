"""
LSTM for multivariate time series: predict next-day T_max (regression).
Uses normalized data and sequence of past days.
"""
from typing import Optional, List
import numpy as np
import pandas as pd

keras = None
layers = None
HAS_KERAS = False
try:
    import tensorflow as tf
    keras = tf.keras
    layers = tf.keras.layers
    HAS_KERAS = True
except ImportError:
    try:
        import keras
        layers = keras.layers
        HAS_KERAS = True
    except ImportError:
        pass


# Default feature columns for sequence input (normalized dataset)
LSTM_FEATURE_COLS = [
    "T_max", "T_min", "rain_mm", "RH_mean", "cloud", "MSLP",
    "DTR", "rain_freq", "MSLP_delta",
]


def build_sequences(
    df: pd.DataFrame,
    target_col: str = "T_max",
    feature_cols: Optional[List[str]] = None,
    seq_len: int = 30,
) -> tuple:
    """
    Build sequences: X[batch, seq_len, n_features], y[batch] (next-day target).
    Returns (X, y, dates) where dates are the date of the target (next day).
    """
    if feature_cols is None:
        feature_cols = [c for c in LSTM_FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    vals = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    dates = df["date"].values

    X_list, y_list, date_list = [], [], []
    for i in range(seq_len, len(df)):
        X_list.append(vals[i - seq_len : i])
        y_list.append(target[i])
        date_list.append(dates[i])
    return np.array(X_list), np.array(y_list), np.array(date_list)


def get_lstm_model(
    seq_len: int,
    n_features: int,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    classification: bool = False,
) -> "keras.Model":
    """Build LSTM model for regression or binary classification (e.g. rain yes/no)."""
    if not HAS_KERAS:
        raise ImportError("TensorFlow is required for LSTM. Install with: pip install tensorflow")
    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=(seq_len, n_features)),
        layers.Dropout(dropout),
        layers.LSTM(units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid" if classification else None),
    ])
    if classification:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    units: int = 64,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    verbose: int = 1,
    classification: bool = False,
) -> "keras.Model":
    """
    Train LSTM for regression or binary classification. Uses validation data for early stopping if provided.
    """
    if not HAS_KERAS:
        raise ImportError("TensorFlow is required for LSTM. Install with: pip install tensorflow")
    _, seq_len, n_features = X_train.shape
    model = get_lstm_model(seq_len, n_features, units=units, dropout=dropout, classification=classification)
    callbacks = []
    if X_val is not None and y_val is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if (X_val is not None and y_val is not None) else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model


def predict_lstm(model: "keras.Model", X: np.ndarray, classification: bool = False) -> np.ndarray:
    """Predict next-day value (regression) or class 0/1 (classification). Returns 1D array."""
    pred = model.predict(X, verbose=0)
    if classification:
        return (pred.ravel() >= 0.5).astype(np.int32)
    return pred.ravel()
