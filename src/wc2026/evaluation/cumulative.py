"""Cumulative live-performance metrics over the WC 2026 timeline.

For each day d in the tournament, compute the metric on all matches played
on or before d. Lets us see the trajectory of each model as more matches
get played: are we converging? is one model pulling away?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc2026.evaluation.metrics import metrics_summary

METRIC_COLS = ["log_loss", "brier", "rps", "accuracy", "ece"]


def cumulative_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative metrics per model at each match-day cutoff.

    Args:
        predictions: DataFrame with columns
            date, model, outcome, p_home, p_draw, p_away
            (outcome ∈ {'H','D','A'} for played matches)

    Returns:
        DataFrame with columns:
            date, model, n_matches, log_loss, brier, rps, accuracy, ece
    """
    df = predictions.dropna(subset=["outcome"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "model", "n_matches", *METRIC_COLS])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    rows = []
    cutoffs = sorted(df["date"].unique())
    for cutoff in cutoffs:
        for model_name in sorted(df["model"].unique()):
            sub = df[(df["model"] == model_name) & (df["date"] <= cutoff)]
            if len(sub) < 1:
                continue
            m = metrics_summary(sub["outcome"].tolist(),
                                sub[["p_home", "p_draw", "p_away"]])
            rows.append({
                "date": pd.Timestamp(cutoff),
                "model": model_name,
                "n_matches": len(sub),
                **m,
            })
    return pd.DataFrame(rows)


def cumulative_score_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Cumulative exact-score accuracy + MAE per model at each cutoff.

    Only meaningful for predictions that include score_mode_h / score_mode_a
    / e_h / e_a (i.e. Poisson-family models).
    """
    df = predictions.dropna(subset=["score_mode_h", "score_mode_a", "true_home", "true_away"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "model", "n_matches", "exact_acc", "mae_h", "mae_a"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["correct"] = (
        df["score_mode_h"].astype(int).eq(df["true_home"].astype(int))
        & df["score_mode_a"].astype(int).eq(df["true_away"].astype(int))
    )

    rows = []
    cutoffs = sorted(df["date"].unique())
    for cutoff in cutoffs:
        for model_name in sorted(df["model"].unique()):
            sub = df[(df["model"] == model_name) & (df["date"] <= cutoff)]
            if len(sub) < 1:
                continue
            rows.append({
                "date": pd.Timestamp(cutoff),
                "model": model_name,
                "n_matches": len(sub),
                "exact_acc": float(sub["correct"].mean()),
                "mae_h": float(
                    np.abs(sub["e_h"].astype(float) - sub["true_home"].astype(float)).mean()
                ),
                "mae_a": float(
                    np.abs(sub["e_a"].astype(float) - sub["true_away"].astype(float)).mean()
                ),
            })
    return pd.DataFrame(rows)
