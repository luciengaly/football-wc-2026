"""Common interface for all prediction models.

Every model implements:
    fit(matches_df)      -> self
    predict_proba(matches_df) -> DataFrame[p_home, p_draw, p_away]

Optional:
    predict_score_dist(matches_df) -> DataFrame of joint score probabilities

All probabilities sum to 1 row-wise. Models are stateless across calls — they
keep what they need (ratings, fitted params) as instance attributes after fit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd


class Model(ABC):
    """Abstract base class for all W/D/L prediction models."""

    name: str = "base"

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> Self:
        """Fit the model on training matches.

        Args:
            matches: DataFrame with at least the columns
                date, home_team, away_team, home_score, away_score,
                tournament, neutral.

        Returns:
            self (fitted).
        """

    @abstractmethod
    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return P(home wins), P(draw), P(away wins) for each input match.

        Args:
            matches: DataFrame with at least columns
                home_team, away_team, neutral (optional, default False).
                Scores are ignored.

        Returns:
            DataFrame with columns [p_home, p_draw, p_away], same length and
            index as input.
        """

    def predict_outcome(self, matches: pd.DataFrame) -> pd.Series:
        """Argmax over W/D/L."""
        proba = self.predict_proba(matches)
        outcomes = np.array(["H", "D", "A"])
        return pd.Series(
            outcomes[proba.to_numpy().argmax(axis=1)],
            index=proba.index,
            name="outcome",
        )


def encode_outcome(home_score: pd.Series, away_score: pd.Series) -> pd.Series:
    """Map (home_score, away_score) -> 'H' / 'D' / 'A' (or NaN if not played)."""
    out = pd.Series(pd.NA, index=home_score.index, dtype="object")
    mask = home_score.notna() & away_score.notna()
    h = home_score[mask]
    a = away_score[mask]
    out.loc[mask] = np.where(h > a, "H", np.where(h < a, "A", "D"))
    return out
