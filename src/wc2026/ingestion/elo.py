"""Compute World Football Elo ratings from the historical match dataset.

Reference: https://www.eloratings.net/about — we follow that formula.

  R_new = R_old + K * G * (W - W_e)

where
  K = K-factor depending on tournament importance (60 for WC, 50 for cont.
      finals, 40 for qualifiers, 30 for other competitive, 20 for friendly)
  G = goal-difference multiplier:
        1 if |diff| <= 1
        1.5 if |diff| == 2
        1.75 if |diff| == 3
        (11 + |diff|) / 8 if |diff| >= 4
  W = actual result (1, 0.5, 0)
  W_e = expected result, from Elo difference adjusted for home advantage

  W_e_home = 1 / (1 + 10 ** ((R_away - R_home - H) / 400))
  H = 100 if home team plays at home, 0 if neutral venue
"""

from __future__ import annotations

import pandas as pd

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0


def _k_factor(tournament: str) -> int:
    """Return K factor based on tournament name.

    We use substring matching to be robust to small naming differences.
    """
    t = tournament.lower()
    # World Cup main tournament
    if "fifa world cup" in t and "qualification" not in t:
        return 60
    # Continental finals (Euro, Copa America, AFCON, Asian Cup, Gold Cup, Nations Cup)
    if "qualification" not in t and any(
        x in t
        for x in (
            "uefa euro",
            "copa américa",
            "copa america",
            "african cup of nations",
            "africa cup of nations",
            "afc asian cup",
            "concacaf championship",
            "concacaf gold cup",
            "ofc nations cup",
            "confederations cup",
        )
    ):
        return 50
    # Qualifiers
    if "qualification" in t:
        return 40
    # Other competitive
    if any(
        x in t
        for x in (
            "uefa nations league",
            "concacaf nations league",
            "arab cup",
            "gulf cup",
            "king's cup",
            "kirin cup",
        )
    ):
        return 30
    # Friendlies
    if "friendly" in t:
        return 20
    # Default — treat as a minor competitive game
    return 30


def _goal_diff_index(diff: int) -> float:
    d = abs(int(diff))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    if d == 3:
        return 1.75
    return (11 + d) / 8


def _expected(rating_a: float, rating_b: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a - home_adv) / 400))


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Walk through matches chronologically and compute Elo trajectories.

    Args:
        matches: DataFrame sorted by date ascending, with columns
                 date, home_team, away_team, home_score, away_score,
                 tournament, neutral.

    Returns:
        elo_trajectory: per-match DataFrame with pre-match ratings
                        (elo_home_pre, elo_away_pre) — usable as features
                        with NO leakage (the post-match update is computed
                        AFTER reading the pre values).
        final_ratings:  team -> latest rating dict, as of the last match
                        in the dataset.
    """
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date ascending")

    ratings: dict[str, float] = {}
    rows = []

    for m in matches.itertuples(index=False):
        h, a = m.home_team, m.away_team
        r_h = ratings.get(h, INITIAL_RATING)
        r_a = ratings.get(a, INITIAL_RATING)

        # Skip rating update if score is missing (future matches)
        if pd.isna(m.home_score) or pd.isna(m.away_score):
            rows.append(
                {
                    "date": m.date,
                    "home_team": h,
                    "away_team": a,
                    "elo_home_pre": r_h,
                    "elo_away_pre": r_a,
                    "elo_home_post": r_h,
                    "elo_away_post": r_a,
                    "k": None,
                }
            )
            continue

        home_adv = 0.0 if m.neutral else HOME_ADVANTAGE
        e_h = _expected(r_h, r_a, home_adv)
        e_a = 1.0 - e_h

        gh, ga = int(m.home_score), int(m.away_score)
        if gh > ga:
            w_h, w_a = 1.0, 0.0
        elif gh < ga:
            w_h, w_a = 0.0, 1.0
        else:
            w_h, w_a = 0.5, 0.5

        g = _goal_diff_index(gh - ga)
        k = _k_factor(m.tournament)

        delta_h = k * g * (w_h - e_h)
        delta_a = k * g * (w_a - e_a)

        r_h_post = r_h + delta_h
        r_a_post = r_a + delta_a

        rows.append(
            {
                "date": m.date,
                "home_team": h,
                "away_team": a,
                "elo_home_pre": r_h,
                "elo_away_pre": r_a,
                "elo_home_post": r_h_post,
                "elo_away_post": r_a_post,
                "k": k,
            }
        )

        ratings[h] = r_h_post
        ratings[a] = r_a_post

    return pd.DataFrame(rows), ratings


def ratings_as_of(elo_trajectory: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    """Return team -> rating as of a given date (strictly before `date`).

    Useful for building features without leakage at prediction time.
    """
    df = elo_trajectory[elo_trajectory["date"] < date]
    if df.empty:
        return {}
    # Build per-team most recent rating using the post-match values
    home_view = df[["home_team", "date", "elo_home_post"]].rename(
        columns={"home_team": "team", "elo_home_post": "rating"}
    )
    away_view = df[["away_team", "date", "elo_away_post"]].rename(
        columns={"away_team": "team", "elo_away_post": "rating"}
    )
    all_view = pd.concat([home_view, away_view], ignore_index=True)
    all_view = all_view.sort_values("date").drop_duplicates("team", keep="last")
    return dict(zip(all_view["team"], all_view["rating"], strict=True))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from wc2026.ingestion.historical import load_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("data/raw/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/elo.parquet"))
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results):,} matches")

    traj, final = compute_elo(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traj.to_parquet(args.out, index=False)
    print(f"Wrote Elo trajectory to {args.out}")

    top10 = sorted(final.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 current Elo (as of last match in dataset):")
    for i, (team, r) in enumerate(top10, 1):
        print(f"  {i:2d}. {team:30s}  {r:7.1f}")
