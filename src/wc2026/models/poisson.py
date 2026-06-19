"""M2 — Independent Poisson model with team attack/defense parameters.

Specification:
    log(λ_home) = μ + att[home] + def[away] + γ * (1 - neutral)
    log(λ_away) = μ + att[away] + def[home]

    g_home ~ Poisson(λ_home)
    g_away ~ Poisson(λ_away)         (independent)

Fit by maximum likelihood with:
  - exponential time-decay weights (recent matches matter more)
  - L2 ridge penalty on attack/defense for identifiability and stability
  - filtering of low-data teams (default: at least 10 matches in the window)

Prediction:
  - Build the joint score table P(h, a) for h, a in [0..max_goals]
  - Aggregate to W/D/L
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from wc2026.models.base import Model

DEFAULT_HALF_LIFE_DAYS = 730  # 2 years
DEFAULT_MIN_MATCHES = 10
DEFAULT_MAX_GOALS = 10
DEFAULT_RIDGE = 0.05


class PoissonIndependent(Model):
    """M2 — independent Poisson with team strengths and time decay."""

    name = "M2_poisson"

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
        self.att_: dict[str, float] = {}
        self.def_: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> Self:
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        if played.empty:
            raise ValueError("No played matches in training data")

        # Filter to teams with enough matches
        team_counts = pd.concat([played["home_team"], played["away_team"]]).value_counts()
        valid_teams = sorted(team_counts[team_counts >= self.min_matches].index.tolist())
        if len(valid_teams) < 5:
            raise ValueError("Too few teams with min_matches; relax the threshold")

        team_set = set(valid_teams)
        mask = played["home_team"].isin(team_set) & played["away_team"].isin(team_set)
        played = played.loc[mask].copy()

        # Time-decay weights (exponential)
        latest = played["date"].max()
        days_ago = (latest - played["date"]).dt.days.to_numpy()
        decay_rate = np.log(2.0) / self.half_life_days
        weights = np.exp(-decay_rate * days_ago)

        team_to_idx = {t: i for i, t in enumerate(valid_teams)}
        n_teams = len(valid_teams)

        h_idx = played["home_team"].map(team_to_idx).to_numpy()
        a_idx = played["away_team"].map(team_to_idx).to_numpy()
        h_goals = played["home_score"].astype(int).to_numpy()
        a_goals = played["away_score"].astype(int).to_numpy()
        not_neutral = (~played["neutral"].to_numpy()).astype(float)

        # Parameters: [mu, gamma, att_1..N, def_1..N]
        def unpack(theta: np.ndarray):
            mu, gamma = theta[0], theta[1]
            att = theta[2 : 2 + n_teams]
            defs = theta[2 + n_teams : 2 + 2 * n_teams]
            return mu, gamma, att, defs

        def neg_log_lik_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            mu, gamma, att, defs = unpack(theta)
            log_lam_h = mu + att[h_idx] + defs[a_idx] + gamma * not_neutral
            log_lam_a = mu + att[a_idx] + defs[h_idx]
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)
            # Poisson log-likelihood (drop constant log(k!))
            ll = (h_goals * log_lam_h - lam_h) + (a_goals * log_lam_a - lam_a)
            ll_weighted = (ll * weights).sum()
            reg = self.ridge * (att @ att + defs @ defs)
            nll = float(-ll_weighted + reg)

            # Per-match residuals (gradient of LL wrt log-lambda)
            res_h = weights * (h_goals - lam_h)
            res_a = weights * (a_goals - lam_a)

            grad = np.zeros_like(theta)
            grad[0] = -(res_h.sum() + res_a.sum())          # d/d(mu)
            grad[1] = -((res_h * not_neutral).sum())         # d/d(gamma)

            # Vectorized accumulation per team
            grad_att = np.zeros(n_teams)
            grad_def = np.zeros(n_teams)
            np.add.at(grad_att, h_idx, -res_h)               # d/d(att[home])
            np.add.at(grad_att, a_idx, -res_a)               # d/d(att[away])
            np.add.at(grad_def, a_idx, -res_h)               # d/d(def[away])
            np.add.at(grad_def, h_idx, -res_a)               # d/d(def[home])
            # Ridge gradient
            grad_att += 2.0 * self.ridge * att
            grad_def += 2.0 * self.ridge * defs

            grad[2 : 2 + n_teams] = grad_att
            grad[2 + n_teams : 2 + 2 * n_teams] = grad_def
            return nll, grad

        # Initial guess: log of average goals per side, ~0 for everything else
        avg_goals = (h_goals.mean() + a_goals.mean()) / 2.0
        theta0 = np.zeros(2 + 2 * n_teams)
        theta0[0] = np.log(max(avg_goals, 0.1))
        theta0[1] = 0.2  # mild home advantage

        res = minimize(
            neg_log_lik_and_grad,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000, "maxfun": 50000, "gtol": 1e-5},
        )
        if not res.success:
            print(f"  [M2 Poisson] L-BFGS warning: {res.message}")

        mu, gamma, att, defs = unpack(res.x)
        self.mu_ = float(mu)
        self.gamma_ = float(gamma)
        self.att_ = dict(zip(valid_teams, att.tolist(), strict=True))
        self.def_ = dict(zip(valid_teams, defs.tolist(), strict=True))
        return self

    # ------------------------------------------------------------------
    # Prediction helpers
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
        joint = np.outer(h_probs, a_probs)  # shape (max+1, max+1)
        # Re-normalize against truncation tail
        return joint / joint.sum()

    def _wdl_from_joint(self, joint: np.ndarray) -> tuple[float, float, float]:
        # joint[h, a] = P(home_goals = h, away_goals = a)
        # np.tril(_, k=-1) keeps entries where a < h (strict lower triangle) -> home wins
        # np.triu(_, k=1)  keeps entries where a > h (strict upper triangle) -> away wins
        p_home = float(np.tril(joint, k=-1).sum())
        p_draw = float(np.diag(joint).sum())
        p_away = float(np.triu(joint, k=1).sum())
        return p_home, p_draw, p_away

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if not self.att_:
            raise RuntimeError("Model not fitted")
        neutrals = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()
        rows = []
        for i, m in enumerate(matches.itertuples(index=False)):
            lh, la = self._lambdas(m.home_team, m.away_team, bool(neutrals[i]))
            joint = self._score_dist(lh, la)
            ph, pd_, pa = self._wdl_from_joint(joint)
            rows.append((ph, pd_, pa))
        out = pd.DataFrame(rows, columns=["p_home", "p_draw", "p_away"], index=matches.index)
        return out

    def predict_score_dist(self, matches: pd.DataFrame) -> list[np.ndarray]:
        """Return list of joint score probability tables, one per match."""
        if not self.att_:
            raise RuntimeError("Model not fitted")
        neutrals = matches.get("neutral", pd.Series(False, index=matches.index)).to_numpy()
        out = []
        for i, m in enumerate(matches.itertuples(index=False)):
            lh, la = self._lambdas(m.home_team, m.away_team, bool(neutrals[i]))
            out.append(self._score_dist(lh, la))
        return out
