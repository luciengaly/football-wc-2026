"""Evaluation metrics for score-distribution predictions.

A model that predicts a joint score distribution P(h, a) can be evaluated on
multiple complementary metrics:

  - exact_score_accuracy : fraction of matches where the model's mode equals
                           the actual score.
  - top_k_accuracy       : fraction where the actual score is in the model's
                           top-K most probable scores.
  - score_log_loss       : -mean log P(actual score) — penalises confident
                           wrong score predictions.
  - btts_brier           : Brier score on "both teams to score" (binary).
  - over_under_brier     : Brier score on a goals threshold (default 2.5).
  - goal_diff_rps        : RPS over an ordered goal-difference outcome space
                           {<= -2, -1, 0, +1, >= +2} — captures whether the
                           model gets the *direction and magnitude* roughly
                           right.

Each function takes the list of joint score matrices (one per match), aligned
with `y_home_goals` / `y_away_goals` arrays. Matrices need not sum to 1
exactly; they are renormalised internally.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _stack_and_normalise(joints: Iterable[np.ndarray]) -> np.ndarray:
    """Stack into shape (n, max_h+1, max_a+1) with each plate summing to 1."""
    arrays = [j / j.sum() for j in joints]
    return np.stack(arrays, axis=0)


def exact_score_accuracy(joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray) -> float:
    """Mode of P(h, a) == (h*, a*) ?"""
    correct = 0
    n = 0
    for j, h, a in zip(joints, y_h, y_a, strict=True):
        if pd.isna(h) or pd.isna(a):
            continue
        mh, ma = np.unravel_index(j.argmax(), j.shape)
        if int(mh) == int(h) and int(ma) == int(a):
            correct += 1
        n += 1
    return correct / n if n else 0.0


def top_k_accuracy(
    joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray, k: int = 3
) -> float:
    """Is the actual score in the top-K most probable scores?"""
    correct = 0
    n = 0
    for j, h, a in zip(joints, y_h, y_a, strict=True):
        if pd.isna(h) or pd.isna(a):
            continue
        flat = j.flatten()
        top_idx = np.argpartition(-flat, k)[:k]
        rows = top_idx // j.shape[1]
        cols = top_idx % j.shape[1]
        if any(int(r) == int(h) and int(c) == int(a) for r, c in zip(rows, cols, strict=True)):
            correct += 1
        n += 1
    return correct / n if n else 0.0


def score_log_loss(
    joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray, eps: float = 1e-12
) -> float:
    """-mean log P(actual score), truncated max_goals beyond the joint shape."""
    losses = []
    for j, h, a in zip(joints, y_h, y_a, strict=True):
        if pd.isna(h) or pd.isna(a):
            continue
        h, a = int(h), int(a)
        h = min(h, j.shape[0] - 1)
        a = min(a, j.shape[1] - 1)
        p = max(float(j[h, a]), eps)
        losses.append(-np.log(p))
    return float(np.mean(losses)) if losses else float("nan")


def btts_brier(joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray) -> float:
    """Brier on Both-Teams-To-Score (binary: both scored at least 1)."""
    sq = []
    for j, h, a in zip(joints, y_h, y_a, strict=True):
        if pd.isna(h) or pd.isna(a):
            continue
        # P(BTTS) = sum over (h_>=1, a_>=1)
        p_btts = float(j[1:, 1:].sum())
        y = 1.0 if (int(h) > 0 and int(a) > 0) else 0.0
        sq.append((p_btts - y) ** 2)
    return float(np.mean(sq)) if sq else float("nan")


def over_under_brier(
    joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray, threshold: float = 2.5
) -> float:
    """Brier on a goals threshold (default 2.5)."""
    sq = []
    for j, h, a in zip(joints, y_h, y_a, strict=True):
        if pd.isna(h) or pd.isna(a):
            continue
        h_idx, a_idx = np.indices(j.shape)
        p_over = float(j[(h_idx + a_idx) > threshold].sum())
        y = 1.0 if (int(h) + int(a)) > threshold else 0.0
        sq.append((p_over - y) ** 2)
    return float(np.mean(sq)) if sq else float("nan")


def goal_diff_rps(joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray) -> float:
    """RPS over goal-difference buckets {<= -2, -1, 0, +1, >= +2}."""
    # Cumulative buckets
    cum_sq = []
    for j, h, a in zip(joints, y_h, y_a, strict=True):
        if pd.isna(h) or pd.isna(a):
            continue
        h_idx, a_idx = np.indices(j.shape)
        diff = h_idx - a_idx  # home - away

        p_le_m2 = float(j[diff <= -2].sum())
        p_m1 = float(j[diff == -1].sum())
        p_0 = float(j[diff == 0].sum())
        p_p1 = float(j[diff == 1].sum())
        p_ge_p2 = float(j[diff >= 2].sum())

        probs = np.array([p_le_m2, p_m1, p_0, p_p1, p_ge_p2])
        # Build one-hot for actual
        d = int(h) - int(a)
        bucket = 0 if d <= -2 else (1 if d == -1 else (2 if d == 0 else (3 if d == 1 else 4)))
        onehot = np.zeros(5)
        onehot[bucket] = 1.0

        cum_p = probs.cumsum()
        cum_o = onehot.cumsum()
        # RPS = (1/(K-1)) * sum_{k=1}^{K-1} (cum_p[k] - cum_o[k])^2
        cum_sq.append(((cum_p[:-1] - cum_o[:-1]) ** 2).sum() / (len(probs) - 1))
    return float(np.mean(cum_sq)) if cum_sq else float("nan")


def score_metrics_summary(
    joints: Iterable[np.ndarray], y_h: np.ndarray, y_a: np.ndarray
) -> dict[str, float]:
    """Bundle all score-level metrics into one dict."""
    joints_list = list(joints)  # materialise so we can iterate multiple times
    return {
        "score_acc": exact_score_accuracy(joints_list, y_h, y_a),
        "top3_acc": top_k_accuracy(joints_list, y_h, y_a, k=3),
        "score_log_loss": score_log_loss(joints_list, y_h, y_a),
        "btts_brier": btts_brier(joints_list, y_h, y_a),
        "ou25_brier": over_under_brier(joints_list, y_h, y_a, threshold=2.5),
        "gd_rps": goal_diff_rps(joints_list, y_h, y_a),
    }
