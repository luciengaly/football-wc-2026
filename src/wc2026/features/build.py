"""Feature engineering for match outcome prediction.

All features are computed *as of* the match date, using only strictly-earlier
matches → no temporal leakage. The pipeline iterates chronologically and
maintains per-team rolling state.

Features computed:
  - elo_home, elo_away           : pre-match Elo (uses wc2026.ingestion.elo)
  - elo_diff                     : elo_home - elo_away (+ home advantage if not neutral)
  - form_home_5, form_away_5     : points (3-1-0) over last 5 matches
  - gf_home_10, ga_home_10       : avg goals for/against over last 10
  - gf_away_10, ga_away_10
  - rest_home, rest_away         : days since last match (clipped 1–60)
  - is_neutral
  - stage_encoded                : 0=group, 1=knockout (for WC matches; 0 elsewhere)
  - conf_home, conf_away         : confederation code (categorical)
  - match_importance             : 5=WC/cont. final, 4=qualif, 3=other competitive, 1=friendly
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
import pandas as pd

from wc2026.ingestion.elo import compute_elo
from wc2026.ingestion.teams import confederation

ROLLING_FORM_WINDOW = 5
ROLLING_GOALS_WINDOW = 10
MAX_REST_DAYS = 60


def _importance(tournament: str) -> int:
    t = tournament.lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 5
    if "qualification" not in t and any(
        x in t for x in ("uefa euro", "copa américa", "copa america",
                         "africa cup of nations", "afc asian cup")
    ):
        return 5
    if "qualification" in t:
        return 4
    if "friendly" in t:
        return 1
    return 3


def _stage(tournament: str, date: pd.Timestamp) -> int:
    """0 = group/regular, 1 = knockout (best-effort).

    For WC 2026 specifically the wc2026.py module is the source of truth; here
    we only care about training-data classification.
    """
    if tournament == "FIFA World Cup":
        # Group stage typically first 2 weeks, then knockout
        # We don't actually know per-match here; default to 0 (group)
        # unless we're past 2 weeks into a known WC window
        return 0
    return 0


def build_features(matches: pd.DataFrame, elo_traj: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute features for every match in `matches`, using rolling state.

    `matches` must be sorted by date ascending. Output is aligned with input
    indices.
    """
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date").reset_index(drop=True)

    # Elo trajectory aligned position-wise with `matches` (chronological)
    if elo_traj is None:
        elo_traj, _ = compute_elo(matches)
    elo_home_pre = elo_traj["elo_home_pre"].to_numpy()
    elo_away_pre = elo_traj["elo_away_pre"].to_numpy()

    # Per-team rolling deques
    form: dict[str, deque] = {}
    goals_for: dict[str, deque] = {}
    goals_against: dict[str, deque] = {}
    last_match_date: dict[str, pd.Timestamp] = {}

    rows = []
    for i, m in enumerate(matches.itertuples(index=False)):
        h, a = m.home_team, m.away_team
        d = m.date

        # Pre-match Elo from trajectory (aligned by position)
        elo_h = float(elo_home_pre[i])
        elo_a = float(elo_away_pre[i])

        h_adv = 100.0 if not m.neutral else 0.0
        elo_diff = elo_h - elo_a + h_adv

        # Form: points per game over last N (default 5)
        form_h = (sum(form.get(h, [])) / len(form[h])) if form.get(h) else 1.0
        form_a = (sum(form.get(a, [])) / len(form[a])) if form.get(a) else 1.0

        # Goals averages (last 10)
        gf_h = (sum(goals_for.get(h, [])) / len(goals_for[h])) if goals_for.get(h) else 1.2
        ga_h = (sum(goals_against.get(h, [])) / len(goals_against[h])) if goals_against.get(h) else 1.2
        gf_a = (sum(goals_for.get(a, [])) / len(goals_for[a])) if goals_for.get(a) else 1.2
        ga_a = (sum(goals_against.get(a, [])) / len(goals_against[a])) if goals_against.get(a) else 1.2

        # Rest days
        rest_h = (d - last_match_date[h]).days if h in last_match_date else 30
        rest_a = (d - last_match_date[a]).days if a in last_match_date else 30
        rest_h = min(max(rest_h, 1), MAX_REST_DAYS)
        rest_a = min(max(rest_a, 1), MAX_REST_DAYS)

        rows.append(
            {
                "date": d,
                "home_team": h,
                "away_team": a,
                "elo_home": elo_h,
                "elo_away": elo_a,
                "elo_diff": elo_diff,
                "form_home_5": form_h,
                "form_away_5": form_a,
                "gf_home_10": gf_h,
                "ga_home_10": ga_h,
                "gf_away_10": gf_a,
                "ga_away_10": ga_a,
                "rest_home": rest_h,
                "rest_away": rest_a,
                "is_neutral": int(m.neutral),
                "conf_home": confederation(h),
                "conf_away": confederation(a),
                "match_importance": _importance(m.tournament) if hasattr(m, "tournament") else 3,
                "stage_encoded": _stage(getattr(m, "tournament", ""), d),
            }
        )

        # Update state AFTER feature computation (no leakage)
        # If scores are unknown (future match), don't update
        if pd.notna(m.home_score) and pd.notna(m.away_score):
            gh, ga = int(m.home_score), int(m.away_score)
            pts_h = 3 if gh > ga else (1 if gh == ga else 0)
            pts_a = 3 if ga > gh else (1 if gh == ga else 0)
            form.setdefault(h, deque(maxlen=ROLLING_FORM_WINDOW)).append(pts_h)
            form.setdefault(a, deque(maxlen=ROLLING_FORM_WINDOW)).append(pts_a)
            goals_for.setdefault(h, deque(maxlen=ROLLING_GOALS_WINDOW)).append(gh)
            goals_against.setdefault(h, deque(maxlen=ROLLING_GOALS_WINDOW)).append(ga)
            goals_for.setdefault(a, deque(maxlen=ROLLING_GOALS_WINDOW)).append(ga)
            goals_against.setdefault(a, deque(maxlen=ROLLING_GOALS_WINDOW)).append(gh)
            last_match_date[h] = d
            last_match_date[a] = d
        else:
            # Still update last_match_date so rest is correct for subsequent fixtures
            last_match_date[h] = d
            last_match_date[a] = d

    return pd.DataFrame(rows)


CATEGORICAL_FEATURES: list[str] = ["conf_home", "conf_away"]
NUMERIC_FEATURES: list[str] = [
    "elo_home", "elo_away", "elo_diff",
    "form_home_5", "form_away_5",
    "gf_home_10", "ga_home_10", "gf_away_10", "ga_away_10",
    "rest_home", "rest_away",
    "is_neutral", "match_importance", "stage_encoded",
]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def encode_categoricals(df: pd.DataFrame, categorical_cols: Iterable[str] = CATEGORICAL_FEATURES) -> pd.DataFrame:
    """Return df with categorical columns cast to pandas 'category' dtype for LightGBM."""
    out = df.copy()
    for c in categorical_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out
