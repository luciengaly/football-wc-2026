"""Convert regulation-time W/D/L probabilities to knockout advance probabilities.

In knockout matches the draw is resolved by extra time + penalty shootout.
We model the resolution as a configurable split of the regulation draw mass
between the two teams. Default 50/50: ET+pens treated as a coin flip, which
is the most-cited empirical finding for international knockout football.

This is intentionally a thin wrapper — no per-match Elo adjustment yet. We
can plug in an Elo-weighted draw_split later if the simple version proves
miscalibrated.

Usage:
    p_h_adv, p_a_adv = advance_probabilities(p_home, p_draw, p_away)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_DRAW_SPLIT = 0.5


def advance_probabilities(
    p_home: pd.Series | np.ndarray | float,
    p_draw: pd.Series | np.ndarray | float,
    p_away: pd.Series | np.ndarray | float,
    draw_split: float = DEFAULT_DRAW_SPLIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P(home advances), P(away advances)).

    Args:
        p_home, p_draw, p_away: regulation-time outcome probabilities (sum to 1).
        draw_split: fraction of P(draw) allocated to the home team.
                    0.5 (default) = coin-flip ET/pens.

    Returns:
        Two arrays/scalars summing to 1 element-wise.
    """
    p_h = np.asarray(p_home, dtype=float)
    p_d = np.asarray(p_draw, dtype=float)
    p_a = np.asarray(p_away, dtype=float)
    p_h_adv = p_h + draw_split * p_d
    p_a_adv = p_a + (1.0 - draw_split) * p_d
    return p_h_adv, p_a_adv


def add_advance_columns(predictions: pd.DataFrame, draw_split: float = DEFAULT_DRAW_SPLIT) -> pd.DataFrame:
    """Add p_home_advances, p_away_advances columns to a predictions DataFrame.

    Only meaningful for knockout matches (stage != "group"). For group matches
    the columns are filled with NaN so they don't get mistaken for valid info.
    """
    out = predictions.copy()
    is_knockout = out["stage"].astype(str) != "group"
    p_h_adv, p_a_adv = advance_probabilities(
        out["p_home"], out["p_draw"], out["p_away"], draw_split=draw_split
    )
    out["p_home_advances"] = np.where(is_knockout, p_h_adv, np.nan)
    out["p_away_advances"] = np.where(is_knockout, p_a_adv, np.nan)
    return out
