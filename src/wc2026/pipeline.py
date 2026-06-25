"""Orchestration pipeline.

Subcommands:
    ingest     : download/refresh data sources (martj42)
    build      : build the WC 2026 dataset and Elo trajectory
    predict    : fit current models and predict the remaining WC 2026 matches
    refresh    : ingest + build + predict + snapshot (one-shot daily run)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
SNAPSHOTS = DATA_DIR / "snapshots"


def cmd_ingest(_args: argparse.Namespace) -> int:
    from wc2026.ingestion.historical import download_all

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Ingesting martj42 dataset...")
    paths = download_all(RAW)
    for name, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"  {name:12s} -> {p}  ({size_kb:,.1f} KB)")
    return 0


def cmd_build(_args: argparse.Namespace) -> int:
    from wc2026.ingestion.elo import compute_elo
    from wc2026.ingestion.historical import load_results
    from wc2026.wc2026 import build_wc2026_dataset

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Building processed datasets...")

    results = load_results(RAW / "results.csv")
    print(f"  Loaded {len(results):,} historical matches")

    traj, ratings = compute_elo(results)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    traj.to_parquet(PROCESSED / "elo.parquet", index=False)
    print(f"  Wrote Elo trajectory ({len(traj):,} rows) -> {PROCESSED / 'elo.parquet'}")

    wc = build_wc2026_dataset(RAW / "results.csv")
    wc.to_parquet(PROCESSED / "wc2026.parquet", index=False)
    n_played = (wc["status"] == "played").sum()
    n_sched = (wc["status"] == "scheduled").sum()
    print(f"  Wrote WC 2026 matches ({len(wc)} = {n_played} played + {n_sched} scheduled) -> {PROCESSED / 'wc2026.parquet'}")
    return 0


def _registered_models() -> dict[str, type]:
    """Models run in the live pipeline.

    Sprint S7-S9 negative results — kept out of the live registry:
      - M5_nb Negative Binomial (ADR-015): degenerates to Poisson on our data.
      - M6_stack LogReg meta-learner (ADR-016): correlated base models, no gain.
      - M2_cal/M3_cal isotonic calibration (ADR-017): hurts more than helps.
      - M7 PyMC bayesian (ADR-019): empirical Bayes shows no gain expected.
      - M8 LGBM 49-classes (ADR-020): worse than M2 on score metrics.
    Sprint S10 — odds from The Odds API:
      - M9_odds: only available for matches that bookmakers have published
        odds for. Returns NaN for matches without coverage; downstream falls
        back to the generative models.
    """
    from wc2026.models.blend import Blend
    from wc2026.models.blend3 import Blend3
    from wc2026.models.dixon_coles import DixonColes
    from wc2026.models.elo_baseline import EloBaseline
    from wc2026.models.lightgbm_clf import LightGBMClassifier
    from wc2026.models.market_value_model import MarketValueModel
    from wc2026.models.odds_model import OddsModel
    from wc2026.models.poisson import PoissonIndependent
    return {
        "M1_elo": EloBaseline,
        "M2_poisson": PoissonIndependent,
        "M3_dixon_coles": DixonColes,
        "M4_lightgbm": LightGBMClassifier,
        "M9_odds": OddsModel,
        "M11_blend": Blend,
        "M12_market_value": MarketValueModel,
        "M13_blend3": Blend3,
    }


def cmd_predict(args: argparse.Namespace) -> int:
    from wc2026.ingestion.historical import load_results

    today = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Predicting with as_of={today.date()}")

    results = load_results(RAW / "results.csv")
    training = results[results["date"] < today].copy()
    print(f"  Training set: {len(training):,} matches up to {training['date'].max().date()}")

    wc = pd.read_parquet(PROCESSED / "wc2026.parquet")
    # Backfill-friendly: predict matches on or after `today` regardless of status
    upcoming = wc[wc["date"] >= today].copy()
    upcoming["neutral"] = ~upcoming["home_team"].isin(["United States", "Canada", "Mexico"])
    print(f"  Matches to predict: {len(upcoming)}")

    only = args.only.split(",") if getattr(args, "only", None) else None
    models = _registered_models()

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    snap_path = SNAPSHOTS / f"{today.date()}.parquet"
    if snap_path.exists():
        all_preds = pd.read_parquet(snap_path)
    else:
        all_preds = pd.DataFrame()

    import numpy as np

    from wc2026.models.knockout import add_advance_columns

    for mname, Cls in models.items():
        if only and mname not in only:
            continue
        print(f"\n  --- {mname} ---")
        m = Cls()
        m.fit(training)
        proba = m.predict_proba(upcoming).reset_index(drop=True)
        pred = pd.concat([upcoming.reset_index(drop=True), proba], axis=1)
        pred["model"] = mname
        pred["as_of"] = today

        # Add score-level columns when the model exposes a joint score dist
        if hasattr(m, "predict_score_dist"):
            joints = m.predict_score_dist(upcoming)
            modes_h, modes_a, eh, ea, p_btts, p_over25 = [], [], [], [], [], []
            for j in joints:
                mh, ma = np.unravel_index(j.argmax(), j.shape)
                modes_h.append(int(mh))
                modes_a.append(int(ma))
                marg_h = j.sum(axis=1)
                marg_a = j.sum(axis=0)
                eh.append(float((np.arange(j.shape[0]) * marg_h).sum()))
                ea.append(float((np.arange(j.shape[1]) * marg_a).sum()))
                p_btts.append(float(j[1:, 1:].sum()))
                hi, ai = np.indices(j.shape)
                p_over25.append(float(j[(hi + ai) > 2].sum()))
            pred["score_mode_h"] = modes_h
            pred["score_mode_a"] = modes_a
            pred["e_h"] = eh
            pred["e_a"] = ea
            pred["p_btts"] = p_btts
            pred["p_over_2_5"] = p_over25

        # Add knockout advance probabilities (NaN for group matches)
        pred = add_advance_columns(pred)

        if not all_preds.empty:
            all_preds = all_preds[all_preds["model"] != mname]
        all_preds = pd.concat([all_preds, pred], ignore_index=True)

        sample_cols = ["date", "home_team", "away_team", "p_home", "p_draw", "p_away"]
        if "score_mode_h" in pred.columns:
            sample_cols += ["score_mode_h", "score_mode_a"]
        sample = pred.head(5)[sample_cols]
        print(sample.to_string(index=False, formatters={
            "p_home": "{:.1%}".format,
            "p_draw": "{:.1%}".format,
            "p_away": "{:.1%}".format,
        }))

    all_preds.to_parquet(snap_path, index=False)
    print(f"\n  Snapshot -> {snap_path}  ({len(all_preds)} rows total)")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    rc = cmd_ingest(args)
    if rc != 0:
        return rc
    # Best-effort odds refresh — never fail the pipeline if odds are down
    try:
        from wc2026.ingestion.odds import fetch_and_save
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Fetching odds from The Odds API...")
        fetch_and_save(RAW)
    except Exception as e:
        print(f"  WARNING: odds fetch failed ({e}). Continuing without fresh odds.")
    rc = cmd_build(args)
    if rc != 0:
        return rc
    return cmd_predict(args)


def cmd_backfill(args: argparse.Namespace) -> int:
    """Reconstruct daily snapshots for the WC 2026 phase that's already played.

    For each day in [start, end), fit each model on data strictly before that
    day and predict the matches happening on that day. This lets the
    dashboard Performance page evaluate the models on the played matches.
    """
    wc = pd.read_parquet(PROCESSED / "wc2026.parquet")
    start = pd.Timestamp(args.start) if args.start else wc["date"].min()
    end = pd.Timestamp(args.end) if args.end else min(
        pd.Timestamp.today().normalize(), wc["date"].max() + pd.Timedelta(days=1)
    )

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Backfill snapshots from {start.date()} to {end.date()}")

    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    for d in days:
        day_matches = wc[wc["date"] == d]
        if day_matches.empty:
            continue
        print(f"\n  -- {d.date()} : {len(day_matches)} matches --")
        ns = argparse.Namespace(as_of=d.strftime("%Y-%m-%d"), only=args.only)
        cmd_predict(ns)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wc2026")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Download / refresh source data")
    sub.add_parser("build", help="Build processed datasets (Elo, WC 2026)")
    p_pred = sub.add_parser("predict", help="Fit models and predict remaining matches")
    p_pred.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD (default: today)")
    p_pred.add_argument("--only", type=str, default=None, help="comma-separated model names to run")
    p_ref = sub.add_parser("refresh", help="ingest + build + predict")
    p_ref.add_argument("--as-of", type=str, default=None)
    p_ref.add_argument("--only", type=str, default=None)
    p_bf = sub.add_parser("backfill",
                          help="reconstruct daily snapshots for the past WC days")
    p_bf.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: WC start)")
    p_bf.add_argument("--end", type=str, default=None, help="YYYY-MM-DD exclusive (default: today)")
    p_bf.add_argument("--only", type=str, default=None)

    args = parser.parse_args(argv)
    return {
        "ingest": cmd_ingest,
        "build": cmd_build,
        "predict": cmd_predict,
        "refresh": cmd_refresh,
        "backfill": cmd_backfill,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
