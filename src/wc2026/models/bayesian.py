"""M7 — Hierarchical Bayesian Poisson model (PyMC, ADVI).

Same likelihood structure as M2/M3 but with a **hierarchical prior** on team
strengths. Each team's att and def are drawn from a global Normal whose
standard deviation is itself learned. This produces *shrinkage*: teams with
few matches are pulled toward the population mean, while teams with rich
match history retain their data-driven estimate.

Why this is structurally different from M2 (which has a fixed L2 ridge):
  M2 ridge = 0.05 imposes the SAME shrinkage on every team regardless of
  how much data they have. M7 lets the data decide *how much* shrinkage
  via σ_att and σ_def, and each team gets a personalised effective ridge.

  In Bayesian terms: M2 ridge is a fixed-σ prior with σ chosen by hand.
  M7 puts a hyperprior on σ and integrates it out (or estimates it via
  Type-II MLE / ADVI). It's the principled extension.

Fit via ADVI for speed (~30s on 5-year window). NUTS available as
diagnostic but too slow for the live pipeline (refit per day).

Time-decay weighting via pm.Potential — same exponential half-life as M2.
"""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
import pandas as pd
from scipy.stats import poisson

from wc2026.models.base import Model


class BayesianHierarchical(Model):
    """M7 — hierarchical Bayesian Poisson with PyMC + ADVI."""

    name = "M7_bayesian"

    def __init__(
        self,
        training_window_days: int = 1825,
        half_life_days: int = 730,
        min_matches: int = 10,
        max_goals: int = 10,
        advi_iters: int = 30000,
        random_seed: int = 42,
    ):
        self.training_window_days = training_window_days
        self.half_life_days = half_life_days
        self.min_matches = min_matches
        self.max_goals = max_goals
        self.advi_iters = advi_iters
        self.random_seed = random_seed
        # Fitted parameters (posterior means)
        self.mu_: float = 0.0
        self.gamma_: float = 0.0
        self.sigma_att_: float = 0.0
        self.sigma_def_: float = 0.0
        self.att_: dict[str, float] = {}
        self.def_: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> Self:
        # Import lazily to keep PyMC optional at import time
        import pymc as pm

        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        cutoff = played["date"].max() - pd.Timedelta(days=self.training_window_days)
        played = played[played["date"] >= cutoff].copy()

        team_counts = pd.concat([played["home_team"], played["away_team"]]).value_counts()
        valid_teams = sorted(team_counts[team_counts >= self.min_matches].index.tolist())
        team_set = set(valid_teams)
        mask = played["home_team"].isin(team_set) & played["away_team"].isin(team_set)
        played = played.loc[mask].copy()

        team_to_idx = {t: i for i, t in enumerate(valid_teams)}
        n_teams = len(valid_teams)

        latest = played["date"].max()
        days_ago = (latest - played["date"]).dt.days.to_numpy()
        decay = np.log(2.0) / self.half_life_days
        weights = np.exp(-decay * days_ago)

        h_idx = played["home_team"].map(team_to_idx).to_numpy()
        a_idx = played["away_team"].map(team_to_idx).to_numpy()
        h_goals = played["home_score"].astype(int).to_numpy()
        a_goals = played["away_score"].astype(int).to_numpy()
        not_neutral = (~played["neutral"].to_numpy()).astype(float)

        # Build PyMC model
        with pm.Model() as _:
            # Global parameters
            mu = pm.Normal("mu", mu=0.0, sigma=1.0, initval=np.log(1.5))
            gamma = pm.Normal("gamma", mu=0.2, sigma=0.5)
            sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
            sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)

            # Non-centred parameterisation: better mixing under ADVI
            att_raw = pm.Normal("att_raw", mu=0.0, sigma=1.0, shape=n_teams)
            def_raw = pm.Normal("def_raw", mu=0.0, sigma=1.0, shape=n_teams)
            att = pm.Deterministic("att", att_raw * sigma_att)
            defs = pm.Deterministic("defs", def_raw * sigma_def)

            log_lam_h = mu + att[h_idx] + defs[a_idx] + gamma * not_neutral
            log_lam_a = mu + att[a_idx] + defs[h_idx]
            lam_h = pm.math.exp(log_lam_h)
            lam_a = pm.math.exp(log_lam_a)

            # Weighted Poisson log-likelihood via Potential
            log_p_h = pm.logp(pm.Poisson.dist(mu=lam_h), h_goals)
            log_p_a = pm.logp(pm.Poisson.dist(mu=lam_a), a_goals)
            pm.Potential("weighted_lik", (log_p_h * weights).sum() + (log_p_a * weights).sum())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                approx = pm.fit(
                    n=self.advi_iters,
                    method="advi",
                    progressbar=False,
                    random_seed=self.random_seed,
                )

            # Sample posterior to get means
            trace = approx.sample(draws=500, random_seed=self.random_seed)

        # Extract posterior means
        post = trace.posterior
        self.mu_ = float(post["mu"].mean())
        self.gamma_ = float(post["gamma"].mean())
        self.sigma_att_ = float(post["sigma_att"].mean())
        self.sigma_def_ = float(post["sigma_def"].mean())
        att_mean = post["att"].mean(dim=("chain", "draw")).values
        def_mean = post["defs"].mean(dim=("chain", "draw")).values
        self.att_ = dict(zip(valid_teams, att_mean.tolist(), strict=True))
        self.def_ = dict(zip(valid_teams, def_mean.tolist(), strict=True))
        return self

    # ------------------------------------------------------------------
    # Prediction (same plumbing as M2)
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
        k = np.arange(self.max_goals + 1)
        h_probs = poisson.pmf(k, lam_h)
        a_probs = poisson.pmf(k, lam_a)
        joint = np.outer(h_probs, a_probs)
        return joint / joint.sum()

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
