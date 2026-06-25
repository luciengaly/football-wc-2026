"""M5_nb — Negative Binomial model with team attack/defense parameters.

Same linear-predictor structure as the Poisson M2:

    log(λ_home) = μ + att[home] + def[away] + γ * (1 - neutral)
    log(λ_away) = μ + att[away] + def[home]

But the goal count is modelled as Negative Binomial (NB2 parameterisation)
instead of Poisson:

    g ~ NB(μ, α)   with   Var(g) = μ + α μ²

α = 0 reduces exactly to Poisson; for football we expect α ∈ [0.1, 0.4]
(over-dispersion of ~20-40%). The advantage over Poisson is that NB puts
more probability mass on extreme scores (0, 0-0, 3+, 4+) — which is the
empirically observed pattern and directly addresses the user feedback
"the model rarely predicts 3+ goals".

Fit strategy (mirrors M3 Dixon-Coles): 2-step
  1. Fit the underlying Poisson MLE (μ, γ, att, def).
  2. Fit α alone via 1-D Brent given those fixed μ values.

This is fast (~1s extra over M2) and numerically robust.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom

from wc2026.models.base import Model
from wc2026.models.poisson import PoissonIndependent

DEFAULT_HALF_LIFE_DAYS = 730
DEFAULT_MIN_MATCHES = 10
DEFAULT_MAX_GOALS = 10
DEFAULT_RIDGE = 0.05
ALPHA_BOUNDS = (1e-4, 2.0)


def _nb_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """Negative-Binomial PMF in (μ, α) parameterisation.

    Uses scipy.stats.nbinom under the hood; we convert to (n, p) form:
        n = 1/α,  p = n / (n + μ) = 1 / (1 + α μ)
    """
    if alpha <= 0:
        # Degenerate -> Poisson
        from scipy.stats import poisson as _p
        return _p.pmf(k, mu)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return nbinom.pmf(k, n, p)


def _nb_log_pmf_vec(k: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised log-PMF of NB(μ, α) at observed counts k.

    Uses the stable closed form:
      log p(k|μ,α) = gammaln(k + 1/α) - gammaln(k+1) - gammaln(1/α)
                    + k log(α μ) - (k + 1/α) log(1 + α μ)
    """
    if alpha <= 1e-9:
        # Poisson limit
        return k * np.log(np.maximum(mu, 1e-12)) - mu - gammaln(k + 1)
    inv_a = 1.0 / alpha
    return (
        gammaln(k + inv_a)
        - gammaln(k + 1)
        - gammaln(inv_a)
        + k * np.log(alpha * mu)
        - (k + inv_a) * np.log1p(alpha * mu)
    )


class NegativeBinomial(Model):
    """M5_nb — Negative Binomial team-strength model (2-step fit)."""

    name = "M5_nb"

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
        self.alpha_: float = 0.0
        self.att_: dict[str, float] = {}
        self.def_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> Self:
        # --- Step 1: fit Poisson to get μ, γ, att, def --------------------
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

        # --- Step 2: fit α alone via 1-D Brent ---------------------------
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

        att_h = np.array([self.att_[t] for t in played["home_team"]])
        att_a = np.array([self.att_[t] for t in played["away_team"]])
        def_h = np.array([self.def_[t] for t in played["home_team"]])
        def_a = np.array([self.def_[t] for t in played["away_team"]])
        mu_h = np.exp(self.mu_ + att_h + def_a + self.gamma_ * not_neutral)
        mu_a = np.exp(self.mu_ + att_a + def_h)

        def neg_log_lik_alpha(alpha: float) -> float:
            ll_h = _nb_log_pmf_vec(h_goals, mu_h, alpha)
            ll_a = _nb_log_pmf_vec(a_goals, mu_a, alpha)
            return float(-((ll_h + ll_a) * weights).sum())

        res = minimize_scalar(
            neg_log_lik_alpha,
            bounds=ALPHA_BOUNDS,
            method="bounded",
            options={"xatol": 1e-5},
        )
        self.alpha_ = float(res.x)
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

    def _score_dist(self, mu_h: float, mu_a: float) -> np.ndarray:
        k = np.arange(self.max_goals + 1)
        h_probs = _nb_pmf(k, mu_h, self.alpha_)
        a_probs = _nb_pmf(k, mu_a, self.alpha_)
        joint = np.outer(h_probs, a_probs)
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
