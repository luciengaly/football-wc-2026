"""M13_blend3 — equal-weight blend of three orthogonal signals.

Components (intentionally decorrelated — this is what S7's failed blends lacked):
  * M3 Dixon-Coles  : match-result history (Elo-style team strengths)
  * M9 Odds         : betting market consensus
  * M12i MarketValue: Transfermarkt squad value, injury-adjusted (available
                      squad only — beats full-value M12 on all 5 metrics, ADR-027)

Per-row equal weighting over whatever components are AVAILABLE:
  * all three present            → (1/3, 1/3, 1/3)
  * M3 + M12 (no odds)           → (1/2, 1/2)
  * M3 only (no odds, no value)  → M3
M3 is always available, so there is always at least one component.

Bench (134 matches with all three present, historical):
                  log_loss  brier   rps    acc    ece
    M3            1.0368   0.6140  0.2062 52.2% 0.0456
    M9            1.0195   0.6091  0.2034 51.5% 0.0514
    M12           1.0238   0.6080  0.2042 52.2% 0.0450
    M11 (M3+M9)   1.0236   0.6089  0.2037 52.2% 0.0373
    M13 (3-way)   1.0159   0.6041  0.2018 51.5% 0.0191   ← best on 4/5 metrics

Equal weights chosen deliberately: a tuned simplex would overfit 134 matches,
and the surface is flat around the centre. Equal weighting is the robust,
explainable choice.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from wc2026.models.base import Model
from wc2026.models.dixon_coles import DixonColes
from wc2026.models.market_value_model import MarketValueModel
from wc2026.models.odds_model import OddsModel


class Blend3(Model):
    """M13 — equal-weight mean of available {M3, M9, M12} predictions."""

    name = "M13_blend3"

    def __init__(self, store=None):
        self.base_ = DixonColes()
        self.odds_ = OddsModel()
        # Injury-aware value (M12i) beats full-value M12 on all 5 metrics
        # standalone and improves the blend's log-loss/Brier/RPS. See ADR-027.
        self.value_ = MarketValueModel(store=store, use_availability=True)

    def fit(self, matches: pd.DataFrame) -> Self:
        self.base_.fit(matches)
        self.odds_.fit(matches)   # ignores `matches`, loads latest odds
        self.value_.fit(matches)
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        base = self.base_.predict_proba(matches).to_numpy()       # always present
        odds = self.odds_.predict_proba(matches).to_numpy()       # may be NaN rows
        value = self.value_.predict_proba(matches).to_numpy()     # may be NaN rows

        n = len(matches)
        out = np.zeros((n, 3))
        for i in range(n):
            components = [base[i]]
            if not np.isnan(odds[i]).any():
                components.append(odds[i])
            if not np.isnan(value[i]).any():
                components.append(value[i])
            stacked = np.vstack(components)
            mix = stacked.mean(axis=0)
            out[i] = mix / mix.sum()
        return pd.DataFrame(out, columns=["p_home", "p_draw", "p_away"], index=matches.index)
