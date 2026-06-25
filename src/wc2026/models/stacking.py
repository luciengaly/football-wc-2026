"""M6_stack — Stacking ensemble with multinomial logistic regression meta-learner.

Why stacking over a simple weighted average:
  - Linear average treats each base model as additive. Stacking can exploit
    non-linear combinations (e.g. "when M2 and M3 disagree, trust M1").
  - With well-correlated base models, the simplex weights collapse near
    {1, 0, ..., 0} and we get the single best model. Stacking is more
    expressive even with small samples.

Why logistic regression and not LightGBM as meta-learner:
  - The meta-training set is small (~365 days × 5-15 matches/day).
  - 12 features (3 probas × 4 base models). A gradient-boosted tree would
    overfit. Multinomial logistic with L2 has the right capacity.

Strict out-of-sample workflow:
  1. Split training set: pre-cutoff (base train) vs post-cutoff (meta train).
  2. Fit base models on the pre-cutoff slice only.
  3. Have base models predict on the meta slice (out-of-sample for them).
  4. Train meta-learner on (base predictions, outcome) from the meta slice.
  5. Re-fit base models on the FULL training set (so live preds use all data).
  6. At predict time: pass live base predictions through the meta-learner.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wc2026.models.base import Model, encode_outcome
from wc2026.models.dixon_coles import DixonColes
from wc2026.models.elo_baseline import EloBaseline
from wc2026.models.lightgbm_clf import LightGBMClassifier
from wc2026.models.poisson import PoissonIndependent

OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}
DEFAULT_VAL_WINDOW_DAYS = 365


def default_base_factories() -> dict[str, type[Model]]:
    return {
        "elo": EloBaseline,
        "poisson": PoissonIndependent,
        "dixon_coles": DixonColes,
        "lightgbm": LightGBMClassifier,
    }


class StackingEnsemble(Model):
    """M6_stack — meta-learner over base model probabilities."""

    name = "M6_stack"

    def __init__(
        self,
        base_factories: dict[str, type[Model]] | None = None,
        validation_window_days: int = DEFAULT_VAL_WINDOW_DAYS,
        l2: float = 1.0,
    ):
        self.base_factories = base_factories or default_base_factories()
        self.validation_window_days = validation_window_days
        self.l2 = l2
        self.fitted_models_: dict[str, Model] = {}
        self.meta_: LogisticRegression | None = None
        self.feature_names_: list[str] = []

    def _stack_features(self, matches: pd.DataFrame) -> np.ndarray:
        """Stack base model probas horizontally: [p_h_M1, p_d_M1, p_a_M1, p_h_M2, ...]."""
        chunks = []
        for name in self.feature_names_:
            p = self.fitted_models_[name].predict_proba(matches)
            chunks.append(p[["p_home", "p_draw", "p_away"]].to_numpy())
        return np.hstack(chunks)

    def fit(self, matches: pd.DataFrame) -> Self:
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        cutoff = played["date"].max() - pd.Timedelta(days=self.validation_window_days)
        base_train = matches[matches["date"] < cutoff].copy()
        meta_set = played[played["date"] >= cutoff].copy()

        if len(meta_set) < 100 or len(base_train) < 1000:
            raise ValueError(
                f"Not enough data: base_train={len(base_train)}, meta_set={len(meta_set)}. "
                "Need at least 1000 base + 100 meta matches."
            )

        # --- Step 1: fit base models on pre-cutoff ------------------------
        self.fitted_models_ = {}
        for name, factory in self.base_factories.items():
            m = factory()
            m.fit(base_train)
            self.fitted_models_[name] = m
        self.feature_names_ = list(self.base_factories.keys())

        # --- Step 2: build meta-features (base preds on meta_set) ---------
        X_meta = self._stack_features(meta_set)
        y_meta = (
            encode_outcome(meta_set["home_score"], meta_set["away_score"])
            .map(OUTCOME_TO_INT)
            .astype(int)
            .to_numpy()
        )

        # --- Step 3: train meta-learner -----------------------------------
        # Recency weights on the meta training set
        latest = meta_set["date"].max()
        days_ago = (latest - meta_set["date"]).dt.days.to_numpy()
        weights = np.exp(-np.log(2.0) / 180.0 * days_ago)  # 6-month half-life

        # sklearn >=1.7 always uses multinomial loss for multi-class targets
        self.meta_ = LogisticRegression(
            penalty="l2",
            C=1.0 / self.l2,
            solver="lbfgs",
            max_iter=2000,
        )
        self.meta_.fit(X_meta, y_meta, sample_weight=weights)

        # --- Step 4: re-fit base models on FULL data ----------------------
        # Live predictions benefit from the most recent matches too.
        for name, factory in self.base_factories.items():
            m = factory()
            m.fit(matches)
            self.fitted_models_[name] = m

        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.meta_ is None:
            raise RuntimeError("Model not fitted")
        X = self._stack_features(matches)
        proba = self.meta_.predict_proba(X)
        # sklearn keeps the order of self.meta_.classes_ — should be [0, 1, 2]
        out = np.zeros_like(proba)
        for i, c in enumerate(self.meta_.classes_):
            out[:, int(c)] = proba[:, i]
        return pd.DataFrame(
            out, columns=["p_home", "p_draw", "p_away"], index=matches.index
        )
