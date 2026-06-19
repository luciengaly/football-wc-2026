"""Walk-forward backtest for score-distribution predictions.

Mirrors `evaluation.backtest` but operates on `predict_score_dist` rather
than `predict_proba`. Only Poisson-family models (M2, M3) expose joint
score distributions.
"""

from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

from wc2026.evaluation.score_metrics import score_metrics_summary
from wc2026.evaluation.tournaments import MAJOR_TOURNAMENTS, filter_tournament


class ScoreModel(Protocol):
    name: str

    def fit(self, matches: pd.DataFrame): ...

    def predict_score_dist(self, matches: pd.DataFrame) -> list: ...


def round_walk_forward_scores(
    model_factory: Callable[[], ScoreModel],
    results: pd.DataFrame,
    target_matches: pd.DataFrame,
    round_key: str = "date",
) -> tuple[list, pd.DataFrame]:
    """Walk-forward score predictions, one re-fit per unique round_key value.

    Returns:
        joints     : list of joint score matrices, one per target match
        meta       : DataFrame with the same number of rows as joints, holding
                     the actual scores and match metadata.
    """
    joints_all = []
    meta_chunks = []
    for _, chunk in target_matches.groupby(round_key, sort=True):
        cutoff = chunk["date"].min()
        training = results[results["date"] < cutoff].copy()
        model = model_factory()
        model.fit(training)
        chunk = chunk.reset_index(drop=True).copy()
        if "neutral" not in chunk.columns:
            chunk["neutral"] = True
        joints = model.predict_score_dist(chunk)
        joints_all.extend(joints)
        meta_chunks.append(chunk)
    meta = pd.concat(meta_chunks, ignore_index=True)
    return joints_all, meta


def backtest_scores_multi_tournaments(
    factories: dict[str, Callable[[], ScoreModel]],
    results: pd.DataFrame,
    tournaments: list[tuple[str, str, str, str]] | None = None,
) -> pd.DataFrame:
    """Aggregate score-level metrics across the given tournaments per model."""
    if tournaments is None:
        tournaments = MAJOR_TOURNAMENTS

    pooled_joints: dict[str, list] = {n: [] for n in factories}
    pooled_meta: dict[str, list[pd.DataFrame]] = {n: [] for n in factories}
    for tname, start, end, label in tournaments:
        sub = filter_tournament(results, tname, start, end)
        if sub.empty:
            continue
        print(f"  {label}: {len(sub)} matches")
        for model_name, factory in factories.items():
            joints, meta = round_walk_forward_scores(factory, results, sub, round_key="date")
            pooled_joints[model_name].extend(joints)
            pooled_meta[model_name].append(meta)

    rows = []
    for model_name, joints in pooled_joints.items():
        if not joints:
            continue
        meta = pd.concat(pooled_meta[model_name], ignore_index=True)
        played = meta.dropna(subset=["home_score", "away_score"])
        kept_joints = [j for j, k in zip(joints, ~meta["home_score"].isna(), strict=True) if k]
        if not kept_joints:
            continue
        m = score_metrics_summary(
            kept_joints,
            played["home_score"].astype(float).to_numpy(),
            played["away_score"].astype(float).to_numpy(),
        )
        rows.append({"model": model_name, "n": len(kept_joints), **m})
    return pd.DataFrame(rows).set_index("model")
