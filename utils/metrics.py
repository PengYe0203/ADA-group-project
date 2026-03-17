"""
Error metrics for regression and classification model evaluation.
"""
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error (%). Avoids div by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict of MSE, RMSE, MAE, MAPE."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy (fraction correct)."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Binary F1 score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    if tp + fp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict of Accuracy and F1 (binary)."""
    return {
        "Accuracy": accuracy(y_true, y_pred),
        "F1": f1_binary(y_true, y_pred),
    }
