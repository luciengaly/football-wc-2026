"""Unit tests for knockout advance probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc2026.models.knockout import (
    add_advance_columns,
    advance_probabilities,
)


def test_advance_split_default():
    p_h_adv, p_a_adv = advance_probabilities(0.5, 0.2, 0.3)
    assert p_h_adv == 0.6
    assert p_a_adv == 0.4
    assert abs(p_h_adv + p_a_adv - 1.0) < 1e-12


def test_advance_sums_to_one_vectorised():
    p_h = np.array([0.7, 0.4, 0.1])
    p_d = np.array([0.2, 0.3, 0.2])
    p_a = np.array([0.1, 0.3, 0.7])
    p_h_adv, p_a_adv = advance_probabilities(p_h, p_d, p_a)
    assert np.allclose(p_h_adv + p_a_adv, 1.0)


def test_add_advance_columns_only_for_knockout():
    df = pd.DataFrame({
        "stage":  ["group", "R16",   "QF"],
        "p_home": [0.5,     0.4,     0.3],
        "p_draw": [0.3,     0.25,    0.4],
        "p_away": [0.2,     0.35,    0.3],
    })
    out = add_advance_columns(df)
    assert "p_home_advances" in out.columns
    # Group row -> NaN
    assert pd.isna(out.loc[0, "p_home_advances"])
    assert pd.isna(out.loc[0, "p_away_advances"])
    # Knockout rows -> non-NaN, sum to 1
    for i in [1, 2]:
        h = float(out.loc[i, "p_home_advances"])
        a = float(out.loc[i, "p_away_advances"])
        assert abs(h + a - 1.0) < 1e-9
        # Should be > the regulation win prob (we absorbed half the draw)
        assert h > float(df.loc[i, "p_home"])


def test_custom_draw_split_biases_toward_favourite():
    # 80% draw weight allocated to home -> home gets a much bigger boost
    p_h_adv, p_a_adv = advance_probabilities(0.4, 0.2, 0.4, draw_split=0.8)
    assert p_h_adv == 0.56
    assert p_a_adv == 0.44
