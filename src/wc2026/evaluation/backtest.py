"""Walk-forward backtest harness.

Two backtest modes:

1) tournament_backtest(model, results, tournament_filter, start_date)
   Strict walk-forward: for each match in the tournament, train on everything
   strictly before that match's date, then predict. Realistic but slow.

2) round_walk_forward(model, results, tournament_filter, rounds)
   Faster: re-train per "round" (e.g. group MD1, MD2, MD3, R16, ...).
   Trains on everything before the round, predicts all matches of the round.
   This is what we use in production during the WC 2026.
"""

from __future__ import annotations

from typing import Callable, Iterable, Protocol

import pandas as pd

from wc2026.evaluation.metrics import metrics_summary
from wc2026.models.base import encode_outcome


class FittableModel(Protocol):
    name: str

    def fit(self, matches: pd.DataFrame): ...

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame: ...


def round_walk_forward(
    model_factory: Callable[[], FittableModel],
    results: pd.DataFrame,
    target_matches: pd.DataFrame,
    round_key: str = "date",
) -> pd.DataFrame:
    """Walk-forward backtest, one re-fit per unique value of round_key.

    Args:
        model_factory: zero-arg callable returning a fresh model instance.
        results: full historical match dataset (sorted by date).
        target_matches: subset to predict (must have date, home_team, away_team,
                        home_score, away_score, neutral).
        round_key: column to group target_matches into rounds.

    Returns:
        DataFrame with target_matches columns + p_home, p_draw, p_away,
        outcome (true), model.
    """
    out_chunks = []
    for round_value, chunk in target_matches.groupby(round_key, sort=True):
        cutoff = chunk["date"].min()
        training = results[results["date"] < cutoff].copy()
        model = model_factory()
        model.fit(training)
        proba = model.predict_proba(chunk).reset_index(drop=True)
        pred = pd.concat([chunk.reset_index(drop=True), proba], axis=1)
        pred["outcome"] = encode_outcome(pred["home_score"], pred["away_score"]).to_numpy()
        pred["model"] = model.name
        out_chunks.append(pred)
    return pd.concat(out_chunks, ignore_index=True)


def score_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute metric summary over a predictions DataFrame."""
    played = predictions.dropna(subset=["outcome"]).copy()
    if played.empty:
        return {}
    return metrics_summary(played["outcome"].tolist(), played[["p_home", "p_draw", "p_away"]])


def compare_models(
    factories: dict[str, Callable[[], FittableModel]],
    results: pd.DataFrame,
    target_matches: pd.DataFrame,
    round_key: str = "date",
) -> pd.DataFrame:
    """Run multiple models on the same backtest and return a metric comparison table."""
    rows = []
    for name, fac in factories.items():
        preds = round_walk_forward(fac, results, target_matches, round_key=round_key)
        scores = score_predictions(preds)
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows).set_index("model")
