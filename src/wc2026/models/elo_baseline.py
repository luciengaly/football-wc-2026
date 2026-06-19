"""M1 — Elo baseline model.

Conversion from Elo expected-score E_H to W/D/L probabilities:

  E_H = P(H wins) + 0.5 * P(D)            (Elo identity)
  P(D) = ν * 4 * E_H * (1 - E_H)          (peaks at E_H=0.5, where draws are most likely)
  P(H) = E_H - P(D)/2
  P(A) = (1 - E_H) - P(D)/2

The draw parameter ν is fit on training data to minimize log-loss.

This is the simplest principled baseline: it only uses Elo ratings, no
feature engineering, no machine learning. Models M2+ must beat it.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from wc2026.ingestion.elo import HOME_ADVANTAGE, compute_elo
from wc2026.models.base import Model, encode_outcome


def _wdl_from_elo(
    elo_h: np.ndarray,
    elo_a: np.ndarray,
    neutral: np.ndarray,
    draw_param: float,
    home_adv: float = HOME_ADVANTAGE,
) -> np.ndarray:
    """Vectorized W/D/L probability conversion.

    Returns array of shape (n, 3) with columns [p_home, p_draw, p_away].
    """
    h_adv = np.where(neutral, 0.0, home_adv)
    e_h = 1.0 / (1.0 + np.power(10.0, (elo_a - elo_h - h_adv) / 400.0))
    p_d = draw_param * 4.0 * e_h * (1.0 - e_h)
    p_h = e_h - p_d / 2.0
    p_a = (1.0 - e_h) - p_d / 2.0
    return np.stack([p_h, p_d, p_a], axis=1)


def _log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-12) -> float:
    y_proba = np.clip(y_proba, eps, 1.0 - eps)
    return -np.log(y_proba[np.arange(len(y_true)), y_true]).mean()


class EloBaseline(Model):
    """M1 — Elo + Davidson-style W/D/L conversion."""

    name = "M1_elo"

    def __init__(self, home_advantage: float = HOME_ADVANTAGE, draw_param: float | None = None):
        self.home_advantage = home_advantage
        self.draw_param = draw_param  # if None, fit on training data
        self.ratings_: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> Self:
        # 1) Compute Elo trajectory & final ratings
        traj, final = compute_elo(matches.sort_values("date").reset_index(drop=True))
        self.ratings_ = final

        # 2) Fit draw_param on the SAME training set (using pre-match ratings → no leakage)
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        if self.draw_param is None and len(played) > 200:
            # Join pre-match ratings from the trajectory
            traj_played = traj[traj["k"].notna()].reset_index(drop=True)
            # Align on (date, home_team, away_team)
            key_cols = ["date", "home_team", "away_team"]
            merged = played.merge(
                traj_played[[*key_cols, "elo_home_pre", "elo_away_pre"]],
                on=key_cols,
                how="inner",
            )
            y = encode_outcome(merged["home_score"], merged["away_score"])
            y_idx = y.map({"H": 0, "D": 1, "A": 2}).to_numpy()
            elo_h = merged["elo_home_pre"].to_numpy()
            elo_a = merged["elo_away_pre"].to_numpy()
            neutral = merged["neutral"].to_numpy()

            def objective(nu: float) -> float:
                proba = _wdl_from_elo(elo_h, elo_a, neutral, nu, self.home_advantage)
                # Ensure proba > 0 even at the boundary
                proba = np.clip(proba, 1e-12, None)
                proba = proba / proba.sum(axis=1, keepdims=True)
                return _log_loss(y_idx, proba)

            res = minimize_scalar(objective, bounds=(0.01, 0.99), method="bounded")
            self.draw_param = float(res.x)
        elif self.draw_param is None:
            self.draw_param = 0.30  # sensible default

        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if not self.ratings_:
            raise RuntimeError("Model not fitted")
        nu = self.draw_param if self.draw_param is not None else 0.30

        elo_h = matches["home_team"].map(lambda t: self.ratings_.get(t, 1500.0)).to_numpy()
        elo_a = matches["away_team"].map(lambda t: self.ratings_.get(t, 1500.0)).to_numpy()
        neutral = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()

        proba = _wdl_from_elo(elo_h, elo_a, neutral, nu, self.home_advantage)
        # Renormalize defensively (in case of clipping at extreme Elo differences)
        proba = np.clip(proba, 0.0, None)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return pd.DataFrame(
            proba, columns=["p_home", "p_draw", "p_away"], index=matches.index
        )
