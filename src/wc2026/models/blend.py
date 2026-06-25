"""M11_blend — weighted average of M3 Dixon-Coles + M9 Odds.

Bench finding (134 matchs, 4 tournois historiques) :
                  log_loss  brier   rps    acc   ece
    M3            1.0368   0.6140  0.2062 52.2% 0.0456
    M9            1.0195   0.6091  0.2034 51.5% 0.0514
    M11 (w=0.5)   1.0236   0.6089  0.2037 52.2% 0.0373   ← best Brier, RPS, ECE blend

The 50/50 weight is the natural "no information" prior. We could tune it
(0.3-0.7 all perform similarly) but 0.5 is robust and explainable.

Fallback when no odds are available (e.g. fixture too far ahead, bookmakers
silent): pure M3. This makes M11 a true drop-in replacement for the W/D/L
champion — equal to M3 when no odds, strictly better when odds are present.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from wc2026.models.base import Model
from wc2026.models.dixon_coles import DixonColes
from wc2026.models.odds_model import OddsModel

DEFAULT_W_ODDS = 0.5


class Blend(Model):
    """M11 — weighted blend of a generative base model + the odds model."""

    name = "M11_blend"

    def __init__(
        self,
        w_odds: float = DEFAULT_W_ODDS,
        base_factory: type[Model] = DixonColes,
        odds_factory: type[Model] = OddsModel,
    ):
        if not 0.0 <= w_odds <= 1.0:
            raise ValueError(f"w_odds must be in [0, 1], got {w_odds}")
        self.w_odds = w_odds
        self.base_factory = base_factory
        self.odds_factory = odds_factory
        self.base_model_: Model | None = None
        self.odds_model_: Model | None = None

    def fit(self, matches: pd.DataFrame) -> Self:
        self.base_model_ = self.base_factory()
        self.base_model_.fit(matches)
        self.odds_model_ = self.odds_factory()
        self.odds_model_.fit(matches)  # OddsModel.fit ignores `matches`
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.base_model_ is None or self.odds_model_ is None:
            raise RuntimeError("Model not fitted")
        base = self.base_model_.predict_proba(matches).to_numpy()
        odds = self.odds_model_.predict_proba(matches).to_numpy()

        # Per-row blend: where odds are NaN, fall back to base
        has_odds = ~np.isnan(odds).any(axis=1)
        out = base.copy()
        out[has_odds] = (
            self.w_odds * odds[has_odds] + (1.0 - self.w_odds) * base[has_odds]
        )
        # Renormalise defensively
        out = out / out.sum(axis=1, keepdims=True)
        return pd.DataFrame(
            out, columns=["p_home", "p_draw", "p_away"], index=matches.index
        )
