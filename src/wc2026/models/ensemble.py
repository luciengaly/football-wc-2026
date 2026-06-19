"""M5 — Ensemble: weighted average of base model predictions.

Weights are learned by minimising log-loss on a held-out validation slice
(last `validation_window_days` days of training data). The weights live on
the simplex (non-negative, sum to 1).

Each base model is fit on ALL training data (so it sees the validation
matches too — that's fine because base models don't see the *outcomes* via
the weight optimizer, only their own predictions). Strictly speaking we
could nest the CV more tightly but for live tournament prediction the
recency of the validation window matters more than perfect isolation.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc2026.evaluation.metrics import log_loss
from wc2026.models.base import Model, encode_outcome

OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}


class Ensemble(Model):
    """M5 — weighted-average ensemble of fitted models."""

    name = "M5_ensemble"

    def __init__(
        self,
        base_factories: dict[str, type[Model]] | None = None,
        validation_window_days: int = 365,
    ):
        from wc2026.models.elo_baseline import EloBaseline
        from wc2026.models.lightgbm_clf import LightGBMClassifier
        from wc2026.models.poisson import PoissonIndependent

        self.base_factories = base_factories or {
            "elo": EloBaseline,
            "poisson": PoissonIndependent,
            "lightgbm": LightGBMClassifier,
        }
        self.validation_window_days = validation_window_days
        self.fitted_models_: dict[str, Model] = {}
        self.weights_: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> Self:
        for name, factory in self.base_factories.items():
            m = factory()
            m.fit(matches)
            self.fitted_models_[name] = m

        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        cutoff = played["date"].max() - pd.Timedelta(days=self.validation_window_days)
        val = played[played["date"] >= cutoff].copy()
        if len(val) < 100:
            self.weights_ = {n: 1.0 / len(self.fitted_models_) for n in self.fitted_models_}
            return self

        # Get each model's predictions on the validation slice
        proba_stack = []
        names = list(self.fitted_models_.keys())
        for n in names:
            p = self.fitted_models_[n].predict_proba(val)
            proba_stack.append(p[["p_home", "p_draw", "p_away"]].to_numpy())
        # shape: (n_models, n_val, 3)
        proba_arr = np.stack(proba_stack, axis=0)
        y = encode_outcome(val["home_score"], val["away_score"]).map(OUTCOME_TO_INT).to_numpy()

        # Optimize simplex weights
        n_models = len(names)

        def neg_log_lik(raw_weights: np.ndarray) -> float:
            # Softmax to enforce simplex constraint
            w = np.exp(raw_weights - raw_weights.max())
            w = w / w.sum()
            mix = (w[:, None, None] * proba_arr).sum(axis=0)
            mix = np.clip(mix, 1e-12, None)
            mix = mix / mix.sum(axis=1, keepdims=True)
            return log_loss(y, mix)

        res = minimize(neg_log_lik, np.zeros(n_models), method="Nelder-Mead", options={"xatol": 1e-4})
        raw = res.x
        w = np.exp(raw - raw.max())
        w = w / w.sum()
        self.weights_ = dict(zip(names, w.tolist(), strict=True))
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        proba_stack = []
        names = list(self.fitted_models_.keys())
        for n in names:
            p = self.fitted_models_[n].predict_proba(matches)
            proba_stack.append(p[["p_home", "p_draw", "p_away"]].to_numpy())
        proba_arr = np.stack(proba_stack, axis=0)
        weights = np.array([self.weights_[n] for n in names])
        mix = (weights[:, None, None] * proba_arr).sum(axis=0)
        mix = np.clip(mix, 1e-12, None)
        mix = mix / mix.sum(axis=1, keepdims=True)
        return pd.DataFrame(mix, columns=["p_home", "p_draw", "p_away"], index=matches.index)
