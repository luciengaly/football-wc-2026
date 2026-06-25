"""Backtest M9_odds on historical international tournaments.

Uses the eatpizzanot dataset (~170 matches across WC 2022, Euro 2020/2024,
Copa 2024). The odds were originally fetched from The Odds API and
Football-Data.co.uk — exactly the same kind of signal we now consume live.

This is the only sane way to compare M9 against M2/M3 BEFORE the WC 2026
matches actually play. Pure offline comparison on past tournaments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from wc2026.evaluation.metrics import metrics_summary
from wc2026.evaluation.tournaments import MAJOR_TOURNAMENTS, filter_tournament
from wc2026.ingestion.odds import odds_to_proba
from wc2026.models.base import encode_outcome

HISTORICAL_ODDS_PATH = Path("data/processed/odds_historical.parquet")


def _load_historical_odds() -> pd.DataFrame:
    if not HISTORICAL_ODDS_PATH.exists():
        raise FileNotFoundError(
            f"{HISTORICAL_ODDS_PATH} missing. "
            "Run `python -m wc2026.ingestion.odds_historical` first."
        )
    return pd.read_parquet(HISTORICAL_ODDS_PATH)


def m9_predict_from_odds(matches: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Return M9 predictions on matches, joined to odds_df by (date, teams)."""
    odds_df = odds_df.copy()
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.normalize()

    matches = matches.copy()
    matches["date_n"] = pd.to_datetime(matches["date"]).dt.normalize()

    # Some bookmakers may have multiple rows per fixture — aggregate by median
    grouped = (
        odds_df.groupby(["date", "home_team", "away_team"], as_index=False)
        .agg(odds_home=("odds_home", "median"),
             odds_draw=("odds_draw", "median"),
             odds_away=("odds_away", "median"))
    )

    merged = matches.merge(
        grouped,
        left_on=["date_n", "home_team", "away_team"],
        right_on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_o"),
    )
    rows = []
    for r in merged.itertuples(index=False):
        if pd.isna(r.odds_home):
            rows.append((float("nan"), float("nan"), float("nan")))
            continue
        rows.append(
            odds_to_proba(r.odds_home, r.odds_draw, r.odds_away, method="normalize")
        )
    return pd.DataFrame(
        rows, columns=["p_home", "p_draw", "p_away"], index=matches.index
    )


def backtest_odds_vs_models(
    other_model_factories: dict[str, Callable],
    results: pd.DataFrame,
    tournaments: list[tuple[str, str, str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run M9 + other models on the historical odds dataset.

    Crucially the other models (M2, M3, …) re-fit walk-forward per round on
    martj42 data — same as our standard 6-tournament bench. M9 just uses
    the historical odds directly (no fit).

    Returns:
        per_tourney  : (model, tournament) -> metrics on matches WITH odds
        pooled       : (model) -> metrics pooled across all matches with odds
    """
    from wc2026.evaluation.backtest import round_walk_forward
    odds_df = _load_historical_odds()
    if tournaments is None:
        tournaments = MAJOR_TOURNAMENTS

    per_t_rows = []
    pooled_preds: dict[str, list[pd.DataFrame]] = {n: [] for n in other_model_factories}
    pooled_m9: list[pd.DataFrame] = []

    for tname, start, end, label in tournaments:
        sub = filter_tournament(results, tname, start, end)
        if sub.empty:
            continue

        # M9 first: align odds to these matches
        m9_proba = m9_predict_from_odds(sub, odds_df)
        coverage = m9_proba["p_home"].notna().sum()
        print(f"  {label}: {len(sub)} matches  |  M9 covers {coverage}")
        if coverage == 0:
            continue

        sub_covered = sub.iloc[m9_proba["p_home"].notna().to_numpy()].reset_index(drop=True)
        m9_proba_covered = m9_proba.dropna().reset_index(drop=True)
        m9_proba_covered["model"] = "M9_odds"
        m9_proba_covered["outcome"] = (
            encode_outcome(sub_covered["home_score"], sub_covered["away_score"]).to_numpy()
        )
        for col in ("date", "home_team", "away_team"):
            m9_proba_covered[col] = sub_covered[col].to_numpy()
        m9_proba_covered = m9_proba_covered.dropna(subset=["outcome"])
        pooled_m9.append(m9_proba_covered)
        per_t_rows.append({
            "model": "M9_odds", "tournament": label,
            **metrics_summary(
                m9_proba_covered["outcome"].tolist(),
                m9_proba_covered[["p_home", "p_draw", "p_away"]],
            ),
        })

        # Other models: walk-forward on the same SUBSET of covered matches
        for mname, fac in other_model_factories.items():
            preds = round_walk_forward(fac, results, sub_covered, round_key="date")
            preds = preds.dropna(subset=["outcome"])
            per_t_rows.append({
                "model": mname, "tournament": label,
                **metrics_summary(preds["outcome"].tolist(),
                                  preds[["p_home", "p_draw", "p_away"]]),
            })
            pooled_preds[mname].append(preds)

    per_tournament = pd.DataFrame(per_t_rows).set_index(["model", "tournament"])

    pooled_rows = []
    # Pool M9
    if pooled_m9:
        all_m9 = pd.concat(pooled_m9, ignore_index=True)
        pooled_rows.append({"model": "M9_odds", "n": len(all_m9),
                            **metrics_summary(all_m9["outcome"].tolist(),
                                              all_m9[["p_home", "p_draw", "p_away"]])})
    for mname, frames in pooled_preds.items():
        if not frames:
            continue
        df_all = pd.concat(frames, ignore_index=True)
        pooled_rows.append({"model": mname, "n": len(df_all),
                            **metrics_summary(df_all["outcome"].tolist(),
                                              df_all[["p_home", "p_draw", "p_away"]])})
    pooled = pd.DataFrame(pooled_rows).set_index("model")
    return per_tournament, pooled
