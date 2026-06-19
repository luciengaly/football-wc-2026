"""Backtest harness for multiple major tournaments.

Each tournament gets walk-forward predictions; results are aggregated to give
stable estimates of model performance.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from wc2026.evaluation.backtest import round_walk_forward, score_predictions
from wc2026.evaluation.metrics import metrics_summary


# (tournament name string, start date, end date, friendly_label)
MAJOR_TOURNAMENTS: list[tuple[str, str, str, str]] = [
    ("FIFA World Cup", "2018-06-14", "2018-07-15", "WC 2018"),
    ("FIFA World Cup", "2022-11-20", "2022-12-18", "WC 2022"),
    ("UEFA Euro", "2020-06-11", "2021-07-11", "Euro 2020"),
    ("UEFA Euro", "2024-06-14", "2024-07-14", "Euro 2024"),
    ("Copa América", "2021-06-13", "2021-07-10", "Copa 2021"),
    ("Copa América", "2024-06-20", "2024-07-14", "Copa 2024"),
]


def filter_tournament(
    results: pd.DataFrame,
    tournament_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    mask = (
        (results["tournament"] == tournament_name)
        & (results["date"] >= start_date)
        & (results["date"] <= end_date)
    )
    return results.loc[mask].copy()


def backtest_multi_tournaments(
    factories: dict[str, Callable],
    results: pd.DataFrame,
    tournaments: list[tuple[str, str, str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all model factories across the given tournaments.

    Returns:
        per_tournament: (model, tournament_label) -> metrics
        aggregated:     (model) -> metrics on the *combined* predictions
                        (so log-loss/Brier/RPS are pooled, not averaged)
    """
    if tournaments is None:
        tournaments = MAJOR_TOURNAMENTS

    per_tourney_rows = []
    pooled_preds: dict[str, list[pd.DataFrame]] = {name: [] for name in factories}

    for tname, start, end, label in tournaments:
        sub = filter_tournament(results, tname, start, end)
        if sub.empty:
            print(f"  WARNING: {label} returned no matches; skipping")
            continue
        print(f"  {label}: {len(sub)} matches")
        for model_name, factory in factories.items():
            preds = round_walk_forward(factory, results, sub, round_key="date")
            scores = score_predictions(preds)
            per_tourney_rows.append({"model": model_name, "tournament": label, **scores})
            pooled_preds[model_name].append(preds)

    per_tournament = pd.DataFrame(per_tourney_rows).set_index(["model", "tournament"])

    # Pooled metrics: stack all tournament predictions per model and compute once
    pooled_rows = []
    for model_name, frames in pooled_preds.items():
        if not frames:
            continue
        all_preds = pd.concat(frames, ignore_index=True).dropna(subset=["outcome"])
        if all_preds.empty:
            continue
        m = metrics_summary(
            all_preds["outcome"].tolist(),
            all_preds[["p_home", "p_draw", "p_away"]],
        )
        pooled_rows.append({"model": model_name, "n": len(all_preds), **m})
    aggregated = pd.DataFrame(pooled_rows).set_index("model")

    return per_tournament, aggregated
