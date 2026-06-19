"""M3 — Dixon-Coles model.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market", JRSS-C.

The model is a bivariate-Poisson with a parametric correction τ(h, a) on the
four low-score cells {(0,0), (0,1), (1,0), (1,1)}. The correction handles the
empirically observed under-dispersion in low-scoring matches.

  log(λ_H) = μ + att[home] + def[away] + γ * (1 - neutral)
  log(λ_A) = μ + att[away] + def[home]

  P(h, a) = Poisson(h | λ_H) * Poisson(a | λ_A) * τ(h, a, λ_H, λ_A, ρ)

  τ(0,0) = 1 - λ_H λ_A ρ
  τ(0,1) = 1 + λ_H ρ
  τ(1,0) = 1 + λ_A ρ
  τ(1,1) = 1 - ρ
  τ(h,a) = 1 otherwise

ρ is a scalar in roughly [-0.2, 0.2] for international football. We bound it to
[-0.49, 0.49] to keep τ strictly positive across realistic λ values.

Fit:
  - Maximum likelihood with analytic gradient
  - Exponential time-decay weights (default half-life: 2 years)
  - Ridge L2 on attack/defense
  - L-BFGS-B with bounds on ρ
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from wc2026.models.base import Model
from wc2026.models.poisson import PoissonIndependent

DEFAULT_HALF_LIFE_DAYS = 730
DEFAULT_MIN_MATCHES = 10
DEFAULT_MAX_GOALS = 10
DEFAULT_RIDGE = 0.05
RHO_BOUND = 0.49


def _tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _tau_matrix(lam_h: float, lam_a: float, rho: float, n: int) -> np.ndarray:
    """Build a (n+1)×(n+1) matrix of τ corrections; mostly 1.0."""
    mat = np.ones((n + 1, n + 1))
    if n >= 0:
        mat[0, 0] = 1.0 - lam_h * lam_a * rho
    if n >= 1:
        mat[0, 1] = 1.0 + lam_h * rho
        mat[1, 0] = 1.0 + lam_a * rho
        mat[1, 1] = 1.0 - rho
    return mat


class DixonColes(Model):
    """M3 — Dixon-Coles bivariate Poisson with low-score correction."""

    name = "M3_dixon_coles"

    def __init__(
        self,
        half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
        min_matches: int = DEFAULT_MIN_MATCHES,
        max_goals: int = DEFAULT_MAX_GOALS,
        ridge: float = DEFAULT_RIDGE,
    ):
        self.half_life_days = half_life_days
        self.min_matches = min_matches
        self.max_goals = max_goals
        self.ridge = ridge
        # Fitted parameters
        self.mu_: float = 0.0
        self.gamma_: float = 0.0
        self.rho_: float = 0.0
        self.att_: dict[str, float] = {}
        self.def_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fit (2-step: Poisson MLE, then ρ alone via 1-D Brent)
    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> Self:
        # --- Step 1: fit the underlying Poisson model -----------------------
        poisson_model = PoissonIndependent(
            half_life_days=self.half_life_days,
            min_matches=self.min_matches,
            max_goals=self.max_goals,
            ridge=self.ridge,
        )
        poisson_model.fit(matches)
        self.mu_ = poisson_model.mu_
        self.gamma_ = poisson_model.gamma_
        self.att_ = poisson_model.att_
        self.def_ = poisson_model.def_

        # --- Step 2: fit ρ alone on the same training matches ---------------
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        team_set = set(self.att_)
        mask = played["home_team"].isin(team_set) & played["away_team"].isin(team_set)
        played = played.loc[mask].copy()

        latest = played["date"].max()
        days_ago = (latest - played["date"]).dt.days.to_numpy()
        decay = np.log(2.0) / self.half_life_days
        weights = np.exp(-decay * days_ago)

        h_goals = played["home_score"].astype(int).to_numpy()
        a_goals = played["away_score"].astype(int).to_numpy()
        not_neutral = (~played["neutral"].to_numpy()).astype(float)

        # Per-match λ values from the fitted Poisson model
        att_arr = np.array([self.att_[t] for t in played["home_team"]])
        att_arr_away = np.array([self.att_[t] for t in played["away_team"]])
        def_arr = np.array([self.def_[t] for t in played["home_team"]])
        def_arr_away = np.array([self.def_[t] for t in played["away_team"]])
        log_lam_h = self.mu_ + att_arr + def_arr_away + self.gamma_ * not_neutral
        log_lam_a = self.mu_ + att_arr_away + def_arr
        lam_h = np.exp(log_lam_h)
        lam_a = np.exp(log_lam_a)

        # Mask the 4 low-score cells
        is_00 = (h_goals == 0) & (a_goals == 0)
        is_01 = (h_goals == 0) & (a_goals == 1)
        is_10 = (h_goals == 1) & (a_goals == 0)
        is_11 = (h_goals == 1) & (a_goals == 1)

        # Pre-compute factors that don't depend on ρ
        ll_h_la = lam_h[is_00] * lam_a[is_00]
        lh_01 = lam_h[is_01]
        la_10 = lam_a[is_10]
        w00, w01, w10, w11 = weights[is_00], weights[is_01], weights[is_10], weights[is_11]

        def neg_log_lik_rho(rho: float) -> float:
            tau_00 = np.maximum(1.0 - ll_h_la * rho, 1e-12)
            tau_01 = np.maximum(1.0 + lh_01 * rho, 1e-12)
            tau_10 = np.maximum(1.0 + la_10 * rho, 1e-12)
            tau_11 = max(1.0 - rho, 1e-12)
            return -float(
                (w00 * np.log(tau_00)).sum()
                + (w01 * np.log(tau_01)).sum()
                + (w10 * np.log(tau_10)).sum()
                + (w11.sum()) * np.log(tau_11)
            )

        # Tight ρ bound: never exceed 1/max(λ_H × λ_A) to keep τ_(0,0) > 0
        max_lh_la = float(ll_h_la.max()) if ll_h_la.size else 1.0
        upper = min(RHO_BOUND, 1.0 / max(max_lh_la, 1.0) - 1e-6)
        lower = max(-RHO_BOUND, -1.0 / max(max(lh_01.max() if lh_01.size else 1.0,
                                                la_10.max() if la_10.size else 1.0), 1.0) + 1e-6)

        res = minimize_scalar(
            neg_log_lik_rho,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-5},
        )
        self.rho_ = float(res.x)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        att_h = self.att_.get(home, 0.0)
        att_a = self.att_.get(away, 0.0)
        def_h = self.def_.get(home, 0.0)
        def_a = self.def_.get(away, 0.0)
        log_lh = self.mu_ + att_h + def_a + self.gamma_ * (0.0 if neutral else 1.0)
        log_la = self.mu_ + att_a + def_h
        return float(np.exp(log_lh)), float(np.exp(log_la))

    def _score_dist(self, lam_h: float, lam_a: float) -> np.ndarray:
        h_probs = poisson.pmf(np.arange(self.max_goals + 1), lam_h)
        a_probs = poisson.pmf(np.arange(self.max_goals + 1), lam_a)
        joint = np.outer(h_probs, a_probs)
        tau = _tau_matrix(lam_h, lam_a, self.rho_, self.max_goals)
        joint = joint * tau
        joint = np.clip(joint, 0.0, None)
        joint = joint / joint.sum()
        return joint

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if not self.att_:
            raise RuntimeError("Model not fitted")
        neutrals = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()
        rows = []
        for i, m in enumerate(matches.itertuples(index=False)):
            lh, la = self._lambdas(m.home_team, m.away_team, bool(neutrals[i]))
            joint = self._score_dist(lh, la)
            p_home = float(np.tril(joint, k=-1).sum())
            p_draw = float(np.diag(joint).sum())
            p_away = float(np.triu(joint, k=1).sum())
            rows.append((p_home, p_draw, p_away))
        return pd.DataFrame(
            rows, columns=["p_home", "p_draw", "p_away"], index=matches.index
        )

    def predict_score_dist(self, matches: pd.DataFrame) -> list[np.ndarray]:
        if not self.att_:
            raise RuntimeError("Model not fitted")
        neutrals = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()
        out = []
        for i, m in enumerate(matches.itertuples(index=False)):
            lh, la = self._lambdas(m.home_team, m.away_team, bool(neutrals[i]))
            out.append(self._score_dist(lh, la))
        return out
