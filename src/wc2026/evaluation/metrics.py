"""Evaluation metrics for W/D/L probabilistic predictions.

All functions take:
  y_true : array-like of {'H','D','A'} or {0,1,2}
  y_proba: DataFrame or array with columns [p_home, p_draw, p_away]

Returns: float (lower = better for log-loss/Brier/RPS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_TO_IDX = {"H": 0, "D": 1, "A": 2, 0: 0, 1: 1, 2: 2}


def _to_idx(y_true) -> np.ndarray:
    if isinstance(y_true, pd.Series):
        y_true = y_true.tolist()
    return np.array([OUTCOME_TO_IDX[v] for v in y_true], dtype=int)


def _to_array(y_proba) -> np.ndarray:
    if isinstance(y_proba, pd.DataFrame):
        return y_proba[["p_home", "p_draw", "p_away"]].to_numpy()
    return np.asarray(y_proba)


def log_loss(y_true, y_proba, eps: float = 1e-12) -> float:
    """Multi-class log loss (cross-entropy)."""
    idx = _to_idx(y_true)
    proba = np.clip(_to_array(y_proba), eps, 1.0 - eps)
    return float(-np.log(proba[np.arange(len(idx)), idx]).mean())


def brier_score(y_true, y_proba) -> float:
    """Multi-class Brier score (mean squared error of one-hot encoding).

    Range [0, 2]; lower is better.
    """
    idx = _to_idx(y_true)
    proba = _to_array(y_proba)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(idx)), idx] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def rps(y_true, y_proba) -> float:
    """Ranked Probability Score for ordered outcomes (H < D < A or A < D < H).

    Note: W/D/L is ORDERED naturally — Home win and Away win are at opposite
    ends of the "result" continuum, with Draw in between. This makes RPS the
    canonical metric in football prediction.

    Range [0, 1]; lower is better.
    """
    idx = _to_idx(y_true)
    proba = _to_array(y_proba)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(idx)), idx] = 1.0
    cum_proba = proba.cumsum(axis=1)
    cum_onehot = onehot.cumsum(axis=1)
    # RPS = mean over samples of (1/(K-1)) * sum_{k=1}^{K-1} (Cp_k - Co_k)^2
    k = proba.shape[1]
    return float(((cum_proba[:, : k - 1] - cum_onehot[:, : k - 1]) ** 2).sum(axis=1).mean() / (k - 1))


def accuracy(y_true, y_proba) -> float:
    """Top-1 accuracy."""
    idx = _to_idx(y_true)
    proba = _to_array(y_proba)
    return float((proba.argmax(axis=1) == idx).mean())


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """ECE across all (sample, class) pairs flattened."""
    idx = _to_idx(y_true)
    proba = _to_array(y_proba)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(idx)), idx] = 1.0
    p_flat = proba.reshape(-1)
    y_flat = onehot.reshape(-1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p_flat)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (p_flat >= lo) & (p_flat < hi if hi < 1 else p_flat <= hi)
        if not m.any():
            continue
        conf = p_flat[m].mean()
        acc = y_flat[m].mean()
        ece += m.sum() / n * abs(conf - acc)
    return float(ece)


def metrics_summary(y_true, y_proba) -> dict[str, float]:
    """Return all metrics in a dict."""
    return {
        "log_loss": log_loss(y_true, y_proba),
        "brier": brier_score(y_true, y_proba),
        "rps": rps(y_true, y_proba),
        "accuracy": accuracy(y_true, y_proba),
        "ece": expected_calibration_error(y_true, y_proba),
    }
