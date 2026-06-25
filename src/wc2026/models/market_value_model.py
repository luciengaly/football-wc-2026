"""M12 — Poisson model driven by Transfermarkt squad market values.

Structure (novel vs Peeters who used total value only):

    log(λ_home) = β0 + β_att·logoff_H + β_def·logdef_A + γ·(1 - neutral)
    log(λ_away) = β0 + β_att·logoff_A + β_def·logdef_H

where
    logoff_X = log1p(offensive squad value of X, in M€)   [Attack + Midfield]
    logdef_X = log1p(defensive squad value of X, in M€)   [Defender + GK]

So a team scores more when its own offensive value is high and the opponent's
defensive value is low. The four global coefficients (β0, β_att, β_def, γ)
are fit by Poisson MLE on historical international matches, with the same
exponential time-decay weighting as M2/M3.

Values are looked up **as of the match date** (no leakage) via MarketValueStore.
Dates are quantised to month-start to make the per-(team, month) cache hit.

Coverage: matches where either side has no resolvable market value are skipped
in training, and return NaN at predict time (the blend falls back to M3).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from wc2026.ingestion.market_value import MarketValueStore
from wc2026.models.base import Model

DEFAULT_HALF_LIFE_DAYS = 1095  # 3 years (values are slower-moving than form)
DEFAULT_MAX_GOALS = 10
MIN_TRAIN_MATCHES = 300


def _month_start(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=d.year, month=d.month, day=1)


class MarketValueModel(Model):
    """M12 — Poisson with attack/defense driven by squad market values."""

    name = "M12_market_value"

    def __init__(
        self,
        store: MarketValueStore | None = None,
        half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
        max_goals: int = DEFAULT_MAX_GOALS,
        training_window_days: int = 4015,  # ~11 years (value-dense era)
        use_availability: bool = False,
    ):
        self.store = store
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.training_window_days = training_window_days
        self.use_availability = use_availability
        self.beta_: np.ndarray | None = None  # [b0, b_att, b_def, gamma]
        if use_availability:
            self.name = "M12i_market_value_inj"

    # ------------------------------------------------------------------
    def _ensure_store(self) -> MarketValueStore:
        if self.store is None:
            from wc2026.ingestion.market_value import get_shared_store
            self.store = get_shared_store()
        return self.store

    def _features(self, home: str, away: str, date: pd.Timestamp, neutral: bool):
        """Return (logoff_H, logdef_H, logoff_A, logdef_A, not_neutral) or None."""
        store = self._ensure_store()
        d = _month_start(pd.Timestamp(date))
        _, off_h, def_h = store.team_value_asof(home, d, self.use_availability)
        _, off_a, def_a = store.team_value_asof(away, d, self.use_availability)
        if any(pd.isna(x) for x in (off_h, def_h, off_a, def_a)):
            return None
        logoff_h = np.log1p(off_h / 1e6)
        logdef_h = np.log1p(def_h / 1e6)
        logoff_a = np.log1p(off_a / 1e6)
        logdef_a = np.log1p(def_a / 1e6)
        return logoff_h, logdef_h, logoff_a, logdef_a, (0.0 if neutral else 1.0)

    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> Self:
        self._ensure_store()
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        played = played.sort_values("date")
        if self.training_window_days and not played.empty:
            cutoff = played["date"].max() - pd.Timedelta(days=self.training_window_days)
            played = played[played["date"] >= cutoff]

        rows = []
        latest = played["date"].max()
        for m in played.itertuples(index=False):
            feat = self._features(m.home_team, m.away_team, m.date, bool(m.neutral))
            if feat is None:
                continue
            logoff_h, logdef_h, logoff_a, logdef_a, nn = feat
            days_ago = (latest - m.date).days
            w = np.exp(-np.log(2.0) / self.half_life_days * days_ago)
            # Two observations per match (home goals, away goals)
            rows.append((int(m.home_score), logoff_h, logdef_a, nn, w))
            rows.append((int(m.away_score), logoff_a, logdef_h, 0.0, w))

        if len(rows) < MIN_TRAIN_MATCHES:
            raise ValueError(f"Too few resolvable matches ({len(rows)//2}) to fit M12")

        arr = np.array(rows, dtype=float)
        goals = arr[:, 0]
        logoff = arr[:, 1]
        logdef = arr[:, 2]
        nn = arr[:, 3]
        weights = arr[:, 4]

        def neg_ll(beta: np.ndarray) -> float:
            b0, b_att, b_def, gamma = beta
            log_lam = b0 + b_att * logoff + b_def * logdef + gamma * nn
            lam = np.exp(np.clip(log_lam, -10, 5))
            ll = goals * log_lam - lam
            return float(-(ll * weights).sum())

        res = minimize(neg_ll, np.array([0.0, 0.3, -0.2, 0.2]), method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 5000})
        self.beta_ = res.x
        return self

    # ------------------------------------------------------------------
    def _lambdas(self, home: str, away: str, date: pd.Timestamp, neutral: bool):
        feat = self._features(home, away, date, neutral)
        if feat is None or self.beta_ is None:
            return None
        logoff_h, logdef_h, logoff_a, logdef_a, nn = feat
        b0, b_att, b_def, gamma = self.beta_
        log_lh = b0 + b_att * logoff_h + b_def * logdef_a + gamma * nn
        log_la = b0 + b_att * logoff_a + b_def * logdef_h
        return float(np.exp(log_lh)), float(np.exp(log_la))

    def _score_dist(self, lam_h: float, lam_a: float) -> np.ndarray:
        k = np.arange(self.max_goals + 1)
        joint = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        return joint / joint.sum()

    def _default_date(self, matches: pd.DataFrame) -> pd.Timestamp:
        if "date" in matches.columns:
            return None  # use per-row date
        return pd.Timestamp.today()

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.beta_ is None:
            raise RuntimeError("Model not fitted")
        neutrals = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()
        has_date = "date" in matches.columns
        rows = []
        for i, m in enumerate(matches.itertuples(index=False)):
            d = m.date if has_date else pd.Timestamp.today()
            lams = self._lambdas(m.home_team, m.away_team, d, bool(neutrals[i]))
            if lams is None:
                rows.append((np.nan, np.nan, np.nan))
                continue
            joint = self._score_dist(*lams)
            rows.append((
                float(np.tril(joint, k=-1).sum()),
                float(np.diag(joint).sum()),
                float(np.triu(joint, k=1).sum()),
            ))
        return pd.DataFrame(rows, columns=["p_home", "p_draw", "p_away"], index=matches.index)

    def predict_score_dist(self, matches: pd.DataFrame) -> list[np.ndarray | None]:
        if self.beta_ is None:
            raise RuntimeError("Model not fitted")
        neutrals = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()
        has_date = "date" in matches.columns
        out = []
        for i, m in enumerate(matches.itertuples(index=False)):
            d = m.date if has_date else pd.Timestamp.today()
            lams = self._lambdas(m.home_team, m.away_team, d, bool(neutrals[i]))
            out.append(self._score_dist(*lams) if lams else None)
        return out
