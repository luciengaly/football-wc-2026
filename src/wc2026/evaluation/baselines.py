"""Trivial baselines for sanity-checking model performance.

If a real model can't beat these, it's broken or the data isn't informative.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from wc2026.models.base import Model, encode_outcome


class UniformBaseline(Model):
    """Always predicts P(H)=P(D)=P(A)=1/3. Log-loss = log(3) ≈ 1.0986."""

    name = "uniform"

    def fit(self, matches: pd.DataFrame) -> Self:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        n = len(matches)
        return pd.DataFrame(
            np.full((n, 3), 1.0 / 3.0),
            columns=["p_home", "p_draw", "p_away"],
            index=matches.index,
        )


class MarginalBaseline(Model):
    """Predicts the empirical W/D/L frequencies of the training set."""

    name = "marginal"

    def __init__(self):
        self.proba_: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(self, matches: pd.DataFrame) -> Self:
        played = matches.dropna(subset=["home_score", "away_score"])
        y = encode_outcome(played["home_score"], played["away_score"])
        counts = y.value_counts(normalize=True)
        self.proba_ = np.array([counts.get("H", 0.0), counts.get("D", 0.0), counts.get("A", 0.0)])
        if self.proba_.sum() == 0:
            self.proba_ = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            self.proba_ = self.proba_ / self.proba_.sum()
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        n = len(matches)
        return pd.DataFrame(
            np.tile(self.proba_, (n, 1)),
            columns=["p_home", "p_draw", "p_away"],
            index=matches.index,
        )
