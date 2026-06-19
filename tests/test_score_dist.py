"""Unit tests locking in score-distribution invariants.

These guard against bugs like the triu/tril swap we hit on M2 the first time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc2026.ingestion.historical import load_results
from wc2026.models.dixon_coles import DixonColes
from wc2026.models.poisson import PoissonIndependent


@pytest.fixture(scope="module")
def fitted_models():
    results = load_results("data/raw/results.csv")
    training = results[results["date"] < "2024-01-01"].copy()
    m2 = PoissonIndependent().fit(training)
    m3 = DixonColes().fit(training)
    return m2, m3


@pytest.fixture
def sample_matches():
    return pd.DataFrame(
        {
            "home_team": ["Brazil", "Germany", "Japan", "Argentina"],
            "away_team": ["Serbia", "Spain", "Costa Rica", "France"],
            "neutral":   [True,     True,    True,         True],
        }
    )


@pytest.mark.parametrize("model_idx", [0, 1])
def test_score_dist_sums_to_one(fitted_models, sample_matches, model_idx):
    model = fitted_models[model_idx]
    joints = model.predict_score_dist(sample_matches)
    for j in joints:
        assert np.all(j >= 0), "score distribution must be non-negative"
        assert abs(j.sum() - 1.0) < 1e-6, f"score distribution must sum to 1, got {j.sum()}"


@pytest.mark.parametrize("model_idx", [0, 1])
def test_wdl_marginals_consistent_with_predict_proba(fitted_models, sample_matches, model_idx):
    model = fitted_models[model_idx]
    joints = model.predict_score_dist(sample_matches)
    proba = model.predict_proba(sample_matches).to_numpy()
    for j, (p_h, p_d, p_a) in zip(joints, proba, strict=True):
        # W/D/L from joint
        from_joint_h = float(np.tril(j, k=-1).sum())
        from_joint_d = float(np.diag(j).sum())
        from_joint_a = float(np.triu(j, k=1).sum())
        # Must match within numerical tolerance
        assert abs(from_joint_h - p_h) < 1e-3
        assert abs(from_joint_d - p_d) < 1e-3
        assert abs(from_joint_a - p_a) < 1e-3


def test_p_home_for_obvious_favourite(fitted_models, sample_matches):
    """Brazil vs Serbia at neutral venue: Brazil must be heavy favourite."""
    m2, _ = fitted_models
    proba = m2.predict_proba(sample_matches.iloc[[0]])
    assert proba["p_home"].iloc[0] > 0.55, (
        f"Brazil vs Serbia: expected P(home) > 0.55, got {proba['p_home'].iloc[0]:.3f}. "
        "Possible triu/tril swap regression."
    )
