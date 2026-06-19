"""M4 — LightGBM 3-class classifier on engineered features.

The model trains on:
  X = engineered features at match time (no leakage)
  y = outcome label {'H', 'D', 'A'} → {0, 1, 2}

Strategy:
  - Build features for ALL training matches using the build_features pipeline.
  - Apply a recency-weighted training: more recent matches get higher weight
    (exponential half-life of 3 years).
  - Use multi-class logistic objective with class probabilities.
  - Predict on target matches by computing the same features at as_of time.

Note: this model only uses features. It does NOT use Elo as a parameter,
but Elo enters as a (very predictive) feature.
"""

from __future__ import annotations

from typing import Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from wc2026.features.build import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    build_features,
    encode_categoricals,
)
from wc2026.models.base import Model, encode_outcome

OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}
HALF_LIFE_DAYS = 1095  # 3 years


class LightGBMClassifier(Model):
    """M4 — LightGBM multi-class classifier on rich features."""

    name = "M4_lightgbm"

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_data_in_leaf: int = 80,
        reg_lambda: float = 1.0,
        half_life_days: int = HALF_LIFE_DAYS,
    ):
        self.params = dict(
            objective="multiclass",
            num_class=3,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            reg_lambda=reg_lambda,
            metric="multi_logloss",
            verbose=-1,
            force_col_wise=True,
        )
        self.n_estimators = n_estimators
        self.half_life_days = half_life_days
        self.booster_: lgb.Booster | None = None
        self.training_history_: pd.DataFrame | None = None
        # Cache of features for the training set (used as history when predicting)
        self._history: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> Self:
        sorted_matches = matches.sort_values("date").reset_index(drop=True)
        self._history = sorted_matches

        # Build features for the full training set (chronological, no leakage)
        feats = build_features(sorted_matches)
        played_mask = sorted_matches["home_score"].notna() & sorted_matches["away_score"].notna()
        feats = feats[played_mask].reset_index(drop=True)
        outcomes = encode_outcome(
            sorted_matches.loc[played_mask, "home_score"].reset_index(drop=True),
            sorted_matches.loc[played_mask, "away_score"].reset_index(drop=True),
        )
        y = outcomes.map(OUTCOME_TO_INT).astype(int).to_numpy()

        # Recency weights
        latest = feats["date"].max()
        days_ago = (latest - feats["date"]).dt.days.to_numpy()
        decay = np.log(2.0) / self.half_life_days
        weights = np.exp(-decay * days_ago)

        X = encode_categoricals(feats[ALL_FEATURES])

        train_set = lgb.Dataset(
            X,
            label=y,
            weight=weights,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )
        self.booster_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_estimators,
        )
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.booster_ is None or self._history is None:
            raise RuntimeError("Model not fitted")

        # Stitch target matches into the history to compute features with full
        # rolling state. We need the history to include matches BEFORE the
        # earliest target match.
        targets = matches.copy()
        if "tournament" not in targets.columns:
            targets["tournament"] = "FIFA World Cup"
        combined = pd.concat(
            [self._history, targets[self._history.columns.intersection(targets.columns).tolist()
                                    + [c for c in targets.columns if c not in self._history.columns]]],
            ignore_index=True,
            sort=False,
        )
        combined = combined.sort_values("date").reset_index(drop=True)

        feats = build_features(combined)
        # Select feature rows corresponding to the target matches
        target_keys = list(zip(targets["date"], targets["home_team"], targets["away_team"], strict=True))
        feats_indexed = feats.set_index(["date", "home_team", "away_team"])
        target_feats = feats_indexed.loc[target_keys].reset_index()

        X = encode_categoricals(target_feats[ALL_FEATURES])
        proba = self.booster_.predict(X)
        return pd.DataFrame(
            proba,
            columns=["p_home", "p_draw", "p_away"],
            index=matches.index,
        )
