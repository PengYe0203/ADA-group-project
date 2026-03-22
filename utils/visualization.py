"""
Plotting helpers for model comparison and prediction visualization.
"""
from typing import Optional, List, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec as mgs


def plot_predictions_vs_actual(
    dates: Union[np.ndarray, pd.DatetimeIndex],
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predictions vs Actual",
    ylabel: str = "Value",
    figsize: tuple = (12, 4),
    path: Optional[str] = None,
) -> None:
    """Time series plot: actual and predicted."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, actual, label="Actual", alpha=0.8)
    ax.plot(dates, predicted, label="Predicted", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def _binary_confusion_counts(
    y_true, y_pred
) -> Tuple[np.ndarray, int, int, int, int]:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] for labels in {0, 1}."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    return cm, tn, fp, fn, tp


def plot_binary_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion matrix (test set)",
    subtitle: Optional[str] = None,
    figsize: tuple = (7.0, 6.5),
    path: Optional[str] = None,
) -> None:
    """
    Confusion matrix for binary classification: rows = actual, columns = predicted.
    Each cell shows count and row-wise fraction (share of that actual class).
    """
    cm, tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred)
    n = tn + fp + fn + tp
    acc = (tn + tp) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_pct = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(cm.max(), 1.0)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted dry (0)", "Predicted rain (1)"])
    ax.set_yticklabels(["Actual dry (0)", "Actual rain (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for (i, j), v in np.ndenumerate(cm):
        cnt = int(v)
        pct = row_pct[i, j]
        txt = f"{cnt}\n({pct:.1%} of row)"
        ax.text(
            j, i, txt, ha="center", va="center",
            color="white" if v > vmax / 2 else "black",
            fontsize=13,
        )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08, label="Count")

    metrics_line = (
        f"Accuracy = {acc:.3f}   |   F1 (rain) = {f1:.3f}   |   "
        f"Precision (rain) = {prec:.3f}   |   Recall (rain) = {rec:.3f}"
    )
    fig.text(0.5, 0.02, metrics_line, ha="center", fontsize=9)
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rain_classification_diagnostic(
    dates: Union[np.ndarray, pd.DatetimeIndex],
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None,
    title: str = "Rain classification — test set",
    score_label: str = "P(rain)",
    zoom_last_days: int = 120,
    figsize: tuple = (14, 11),
    path: Optional[str] = None,
) -> None:
    """
    Readability-focused plots for binary rain vs a line chart of 0/1:
    confusion matrix, optional predicted probability, 2-row actual/predicted
    heat strip, agreement strip, and a zoomed step plot for the last days.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    dates = pd.to_datetime(np.asarray(dates).ravel())
    n = len(y_true)
    if len(y_pred) != n or len(dates) != n:
        raise ValueError("dates, y_true, and y_pred must have the same length")
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float).ravel()
        if len(y_score) != n:
            raise ValueError("y_score length must match y_true")

    cm, tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred)

    acc = float(np.mean(y_true == y_pred))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 1.0, 0.35, 1.25],
        width_ratios=[1.0, 1.0],
        hspace=0.35,
        wspace=0.25,
    )

    ax_cm = fig.add_subplot(gs[0, 0])
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred dry (0)", "Pred rain (1)"])
    ax_cm.set_yticklabels(["Actual dry (0)", "Actual rain (1)"])
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, f"{int(v)}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax_cm.set_title("Confusion matrix (counts)")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")
    ax_txt.set_title("Test-set summary")
    summary = (
        f"Accuracy: {acc:.3f}\n"
        f"F1 (rain=1): {f1:.3f}\n"
        f"Precision (rain): {prec:.3f}\n"
        f"Recall (rain): {rec:.3f}\n\n"
        f"TN={tn}  FP={fp}\n"
        f"FN={fn}  TP={tp}\n\n"
        f"Green strip = day matches; red = mismatch."
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes, va="top", fontsize=11, family="monospace")

    ax_prob = fig.add_subplot(gs[1, :])
    if y_score is not None:
        ax_prob.plot(dates, y_score, color="C0", linewidth=0.8, alpha=0.85, label=score_label)
        ax_prob.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="0.5 threshold")
        ax_prob.set_ylabel("Probability")
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.legend(loc="upper right")
        ax_prob.set_title("Predicted probability over time (full test window)")
    else:
        ax_prob.text(0.5, 0.5, "No probability scores — only hard 0/1 predictions", ha="center", va="center", transform=ax_prob.transAxes)
        ax_prob.set_xticks([])
        ax_prob.set_yticks([])
    ax_prob.grid(True, alpha=0.3)
    ax_prob.tick_params(axis="x", rotation=45)

    ax_strip = fig.add_subplot(gs[2, :])
    agree = (y_true == y_pred).astype(float)[np.newaxis, :]
    ax_strip.imshow(agree, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax_strip.set_yticks([0])
    ax_strip.set_yticklabels(["Match"])
    ax_strip.set_xticks([])
    ax_strip.set_title("Per-day agreement (green = correct, red = wrong)")
    pos = int(max(0, n - zoom_last_days))
    ax_strip.axvline(x=pos - 0.5, color="k", linestyle=":", linewidth=1, alpha=0.6)

    gs_zoom = mgs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3, :], hspace=0.12)
    d_z = dates[pos:]
    a_z = y_true[pos:].astype(float)
    p_z = y_pred[pos:].astype(float)
    ax_za = fig.add_subplot(gs_zoom[0, 0])
    ax_zp = fig.add_subplot(gs_zoom[1, 0], sharex=ax_za)
    ax_za.step(d_z, a_z, where="post", color="C0", linewidth=1.2)
    ax_za.set_ylabel("Actual")
    ax_za.set_yticks([0.0, 1.0])
    ax_za.set_yticklabels(["0 dry", "1 rain"])
    ax_za.set_ylim(-0.15, 1.15)
    ax_za.grid(True, alpha=0.3)
    ax_za.set_title(f"Zoom: last {len(d_z)} days (step plots, stacked)")
    ax_za.tick_params(axis="x", labelbottom=False)

    ax_zp.step(d_z, p_z, where="post", color="C1", linewidth=1.2)
    ax_zp.set_ylabel("Predicted")
    ax_zp.set_yticks([0.0, 1.0])
    ax_zp.set_yticklabels(["0 dry", "1 rain"])
    ax_zp.set_ylim(-0.15, 1.15)
    ax_zp.grid(True, alpha=0.3)
    ax_zp.tick_params(axis="x", rotation=45)

    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    metrics_dict: dict,
    metric_names: Optional[List[str]] = None,
    title: str = "Model Comparison",
    path: Optional[str] = None,
) -> None:
    """
    metrics_dict: e.g. {"Logistic": {"Accuracy": 0.8, "F1": 0.7}, "LSTM": {"RMSE": 1.2}, ...}
    """
    if not metrics_dict:
        return
    models = list(metrics_dict.keys())
    if metric_names is None:
        metric_names = sorted(set(k for m in metrics_dict for k in metrics_dict[m]))
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [metrics_dict[model].get(m, np.nan) for m in metric_names]
        rects = ax.bar(x + i * width - (len(models) - 1) * width / 2, vals, width, label=model)
        # Label each bar with its value
        labels = [f"{v:.2f}" if np.isfinite(v) else "—" for v in vals]
        ax.bar_label(rects, labels=labels, fontsize=8, padding=2)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_future_forecast(
    hist_dates: Union[np.ndarray, pd.DatetimeIndex],
    hist_values: np.ndarray,
    future_dates: Union[np.ndarray, pd.DatetimeIndex],
    future_pred: np.ndarray,
    title: str = "Future Forecast",
    ylabel: str = "Value",
    figsize: tuple = (12, 5),
    path: Optional[str] = None,
) -> None:
    """Plot history and future forecast."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hist_dates, hist_values, label="Historical", color="C0", alpha=0.8)
    ax.plot(future_dates, future_pred, label="Forecast", color="C1", linestyle="--", marker="o", markersize=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
