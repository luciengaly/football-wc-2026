"""Post-hoc probability calibration via per-class isotonic regression.

Take any fitted base model that exposes `predict_proba`. Fit isotonic
calibrators on a held-out validation set, then wrap the base model so its
predictions go through the calibrators (with renormalization).

This is multi-class one-vs-rest calibration. Simple, no parametric assumption,
robust on small validation sets.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from wc2026.models.base import Model, encode_outcome

OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}


class CalibratedModel(Model):
    """Wrap a fitted base model with per-class isotonic calibration."""

    def __init__(self, base_model: Model, calibration_window_days: int = 730):
        self.base_model = base_model
        self.calibration_window_days = calibration_window_days
        self.calibrators_: list[IsotonicRegression] = []
        self.name = f"{base_model.name}_iso"

    def fit(self, matches: pd.DataFrame) -> Self:
        """Strict hold-out calibration without invalidating the calibrators.

        Key correctness point: the calibrators are learned on the base model's
        *pre-cutoff* outputs. If we then refit the base on the FULL dataset,
        the calibrators no longer apply to the right distribution and the
        whole pipeline gets worse (observed empirically — see ADR-016).

        So we keep the pre-cutoff base for both calibrator fitting AND live
        prediction. We sacrifice the most-recent N days of training data, but
        the calibration stays valid.
        """
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        cutoff = played["date"].max() - pd.Timedelta(days=self.calibration_window_days)
        base_train = matches[matches["date"] < cutoff].copy()
        cal_set = played[played["date"] >= cutoff].copy()

        if len(cal_set) < 200 or len(base_train) < 1000:
            # Fallback: no calibration, just fit base on full data
            self.base_model.fit(matches)
            self.calibrators_ = [IsotonicRegression(out_of_bounds="clip") for _ in range(3)]
            for c in self.calibrators_:
                c.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            return self

        # Step 1: fit base on pre-cutoff data (this is the model used at predict time)
        self.base_model.fit(base_train)

        # Step 2: get out-of-sample preds on the calibration window
        proba = self.base_model.predict_proba(cal_set)
        y = encode_outcome(cal_set["home_score"], cal_set["away_score"]).map(OUTCOME_TO_INT)
        y_arr = y.to_numpy()

        self.calibrators_ = []
        for class_idx in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(proba.iloc[:, class_idx].to_numpy(), (y_arr == class_idx).astype(float))
            self.calibrators_.append(iso)

        # NB: deliberately no refit on full data — see docstring.
        return self

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        base = self.base_model.predict_proba(matches).to_numpy()
        cal = np.zeros_like(base)
        for c in range(3):
            cal[:, c] = self.calibrators_[c].predict(base[:, c])
        cal = np.clip(cal, 1e-6, None)
        cal = cal / cal.sum(axis=1, keepdims=True)
        return pd.DataFrame(cal, columns=["p_home", "p_draw", "p_away"], index=matches.index)
