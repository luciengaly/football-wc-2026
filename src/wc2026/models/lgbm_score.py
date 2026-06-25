"""M8 — LightGBM 49-class classifier on the exact score.

Classes are pairs (h, a) with 0 ≤ h ≤ MAX_GOALS, 0 ≤ a ≤ MAX_GOALS,
i.e. 7×7 = 49 classes by default (any actual score > 6 gets clipped).

Key design choices:

1. **Class encoding**: index = h * (MAX_GOALS + 1) + a. We can recover (h, a)
   from the index trivially with divmod.

2. **Class weights**: many score classes (5-0, 4-3, 6-2…) are rare. We use
   class weights inversely proportional to frequency to avoid the model
   collapsing onto the dominant 1-0 / 0-0 / 1-1 cells.

3. **Features**: same engineered features as M4 (Elo, form, rest,
   confederation, importance) **plus** λ_H and λ_A predicted by a fitted
   M2 Poisson. The Poisson lambdas act as a strong "prior" — M8 learns
   *corrections* on top of the generative baseline (e.g. when a high-form
   team faces a low-form opponent, M2's 1.5-1.0 might become 2-1).

4. **Output**: P(h, a) for each (h, a) up to MAX_GOALS. Aggregated to W/D/L
   via the standard tril/diag/triu split.

5. **Time-decay weights**: 3-year half-life (same as M4) layered on top of
   class weights.
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
from wc2026.models.base import Model
from wc2026.models.poisson import PoissonIndependent

MAX_GOALS = 6
N_CLASSES = (MAX_GOALS + 1) ** 2
HALF_LIFE_DAYS = 1095  # 3 years


def _to_class(h: int, a: int) -> int:
    h = min(int(h), MAX_GOALS)
    a = min(int(a), MAX_GOALS)
    return h * (MAX_GOALS + 1) + a


def _from_class(c: int) -> tuple[int, int]:
    return divmod(c, MAX_GOALS + 1)


class LGBMScoreClassifier(Model):
    """M8 — exact-score classifier with M2 lambdas as features."""

    name = "M8_lgbm_score"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.04,
        num_leaves: int = 31,
        min_data_in_leaf: int = 60,
        reg_lambda: float = 1.0,
        half_life_days: int = HALF_LIFE_DAYS,
    ):
        self.params = dict(
            objective="multiclass",
            num_class=N_CLASSES,
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
        self.poisson_: PoissonIndependent | None = None
        self._history: pd.DataFrame | None = None
        self.feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    # Feature building (engineered + M2 lambdas)
    # ------------------------------------------------------------------
    def _add_lambda_features(
        self, feats: pd.DataFrame, matches: pd.DataFrame
    ) -> pd.DataFrame:
        """Append λ_H and λ_A from the fitted M2 Poisson as numeric features."""
        if self.poisson_ is None:
            raise RuntimeError("Poisson model not fitted")

        # Need the neutral column; matches typically has it, otherwise infer
        if "neutral" in matches.columns:
            neutrals = matches["neutral"].to_numpy()
        else:
            neutrals = np.ones(len(matches), dtype=bool)

        lams = []
        for i, m in enumerate(matches.itertuples(index=False)):
            lh, la = self.poisson_._lambdas(m.home_team, m.away_team, bool(neutrals[i]))
            lams.append((lh, la))
        lams_arr = np.array(lams)
        feats = feats.copy()
        feats["lam_h_m2"] = lams_arr[:, 0]
        feats["lam_a_m2"] = lams_arr[:, 1]
        feats["lam_diff_m2"] = lams_arr[:, 0] - lams_arr[:, 1]
        return feats

    def _features_for(self, target_matches: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Compute features for target_matches, using history as rolling state."""
        if "tournament" not in target_matches.columns:
            target_matches = target_matches.copy()
            target_matches["tournament"] = "FIFA World Cup"
        combined = pd.concat(
            [
                history,
                target_matches[
                    list(history.columns.intersection(target_matches.columns))
                    + [c for c in target_matches.columns if c not in history.columns]
                ],
            ],
            ignore_index=True,
            sort=False,
        )
        combined = combined.sort_values("date").reset_index(drop=True)
        feats = build_features(combined)
        # Index by (date, home, away) to select target rows
        feats_indexed = feats.set_index(["date", "home_team", "away_team"])
        keys = list(zip(
            target_matches["date"], target_matches["home_team"], target_matches["away_team"], strict=True
        ))
        target_feats = feats_indexed.loc[keys].reset_index()
        target_feats = self._add_lambda_features(target_feats, target_matches)
        return target_feats

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> Self:
        sorted_matches = matches.sort_values("date").reset_index(drop=True)
        self._history = sorted_matches

        # Fit underlying M2 Poisson for the lambda features
        self.poisson_ = PoissonIndependent().fit(sorted_matches)

        # Build features for all training matches
        feats = build_features(sorted_matches)
        played_mask = (
            sorted_matches["home_score"].notna() & sorted_matches["away_score"].notna()
        )
        feats = feats[played_mask].reset_index(drop=True)
        played = sorted_matches.loc[played_mask].reset_index(drop=True)
        feats = self._add_lambda_features(feats, played)

        # Class labels (clip to MAX_GOALS)
        h_goals = played["home_score"].astype(int).clip(upper=MAX_GOALS).to_numpy()
        a_goals = played["away_score"].astype(int).clip(upper=MAX_GOALS).to_numpy()
        y = h_goals * (MAX_GOALS + 1) + a_goals

        # NO class weights: an earlier attempt with inverse-frequency weights
        # blew up the rare cells (e.g. 5-1 predicted at 36% for upset matches).
        # Let LightGBM learn the natural class prior; if it collapses onto the
        # frequent 1-0 / 0-0 cells, that's fine — that's the empirical base
        # rate, and M8 should learn to *shift* it based on the M2 lambdas.

        # Recency weights only
        latest = feats["date"].max()
        days_ago = (latest - feats["date"]).dt.days.to_numpy()
        decay = np.log(2.0) / self.half_life_days
        weights = np.exp(-decay * days_ago)

        self.feature_cols_ = ALL_FEATURES + ["lam_h_m2", "lam_a_m2", "lam_diff_m2"]
        X = encode_categoricals(feats[self.feature_cols_])

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

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_joint(self, matches: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None or self._history is None:
            raise RuntimeError("Model not fitted")
        target_feats = self._features_for(matches, self._history)
        X = encode_categoricals(target_feats[self.feature_cols_])
        proba_flat = self.booster_.predict(X)  # shape (n, N_CLASSES)
        n = proba_flat.shape[0]
        joints = proba_flat.reshape(n, MAX_GOALS + 1, MAX_GOALS + 1)
        # Renormalise defensively
        return joints / joints.sum(axis=(1, 2), keepdims=True)

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        joints = self._predict_joint(matches)
        p_home = np.array([float(np.tril(j, k=-1).sum()) for j in joints])
        p_draw = np.array([float(np.diag(j).sum()) for j in joints])
        p_away = np.array([float(np.triu(j, k=1).sum()) for j in joints])
        return pd.DataFrame(
            {"p_home": p_home, "p_draw": p_draw, "p_away": p_away},
            index=matches.index,
        )

    def predict_score_dist(self, matches: pd.DataFrame) -> list[np.ndarray]:
        return list(self._predict_joint(matches))
