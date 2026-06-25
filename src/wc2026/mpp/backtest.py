"""Backtest MPP strategies on played WC 2026 matches.

For each played match (with MPP cotes filled), fit our models walk-forward on
data strictly before the match day, then compare the realised MPP points of:
  * ours_ev   : argmax expected points (M13 result probs + M2 score dist)
  * user      : the user's actual prediction (my_pred_home/away)
  * favorite  : back the market favourite, modal score
  * model_mode: M2 most-likely score overall (ignores cotes)

Realised points = base (cote if result correct) + bonus if exact score (we
report base separately and count exact hits, since the rarity bonus needs
per-score popularity we don't have historically).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from wc2026.ingestion.historical import load_results
from wc2026.ingestion.market_value import get_shared_store
from wc2026.models.blend3 import Blend3
from wc2026.models.poisson import PoissonIndependent
from wc2026.mpp.optimizer import favorite_score, optimal_score
from wc2026.mpp.scoring import mpp_points, result_sign

MAX_GOALS = 6


def _score_dist_from_lambdas(lam_h: float, lam_a: float, g: int = MAX_GOALS) -> np.ndarray:
    k = np.arange(g + 1)
    joint = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
    return joint / joint.sum()


def run_backtest(cotes_csv: Path = Path("data/mpp/mpp_cotes.csv"),
                 bonus: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cotes_csv)
    played = df[(df["status"] == "played") & df["cote_home_win"].notna()].copy()
    played["date"] = pd.to_datetime(played["date"])

    results = load_results("data/raw/results.csv")
    get_shared_store()  # warm market-value cache

    # Authoritative neutral flag from the source data (martj42): neutral=True
    # for every WC match EXCEPT when a host nation plays at home. World Cups are
    # at neutral venues; only USA/Canada/Mexico get a home edge in 2026.
    wc = pd.read_parquet("data/processed/wc2026.parquet")[["date", "home_team", "away_team", "neutral"]]
    wc["date"] = pd.to_datetime(wc["date"])
    played = played.merge(wc, on=["date", "home_team", "away_team"], how="left")
    played["neutral"] = played["neutral"].fillna(True)

    rows = []
    for day, chunk in played.groupby("date", sort=True):
        training = results[results["date"] < day]
        m13 = Blend3().fit(training)
        m2 = PoissonIndependent().fit(training)

        feats = chunk.copy()  # `neutral` already merged from source data
        p13 = m13.predict_proba(feats).to_numpy()
        m2_dist = m2.predict_score_dist(feats)

        for i, (_, m) in enumerate(chunk.iterrows()):
            pr = (p13[i, 0], p13[i, 1], p13[i, 2])
            sd = m2_dist[i]
            cotes = (m["cote_home_win"], m["cote_draw"], m["cote_away_win"])
            ah, aa = int(m["home_score"]), int(m["away_score"])

            ours, ev = optimal_score(*cotes, pr, sd, bonus=bonus)
            fav = favorite_score(*cotes, sd)
            mode = tuple(int(x) for x in np.unravel_index(sd.argmax(), sd.shape))
            user = (int(m["my_pred_home"]), int(m["my_pred_away"])) \
                if pd.notna(m["my_pred_home"]) else None

            row = {
                "date": day.date(), "match": f"{m['home_team']}-{m['away_team']}",
                "actual": f"{ah}-{aa}",
                "ours": f"{ours[0]}-{ours[1]}", "ours_pts": mpp_points(*ours, ah, aa, *cotes, bonus=bonus),
                "favorite": f"{fav[0]}-{fav[1]}", "fav_pts": mpp_points(*fav, ah, aa, *cotes, bonus=bonus),
                "mode": f"{mode[0]}-{mode[1]}", "mode_pts": mpp_points(*mode, ah, aa, *cotes, bonus=bonus),
            }
            if user is not None:
                row["user"] = f"{user[0]}-{user[1]}"
                row["user_pts"] = mpp_points(*user, ah, aa, *cotes, bonus=bonus)
                row["user_exact"] = int(user == (ah, aa))
            row["ours_exact"] = int(ours == (ah, aa))
            rows.append(row)

    detail = pd.DataFrame(rows)

    # Summary: total points + exact-score hits + result-accuracy per strategy.
    # (strategy label -> (score column, points column))
    strat_cols = {"ours": ("ours", "ours_pts"), "favorite": ("favorite", "fav_pts"),
                  "mode": ("mode", "mode_pts")}
    if "user_pts" in detail:
        strat_cols["user"] = ("user", "user_pts")

    def res_ok(pred_str, actual_str):
        ph, pa = map(int, pred_str.split("-"))
        ah, aa = map(int, actual_str.split("-"))
        return result_sign(ph, pa) == result_sign(ah, aa)

    summ = []
    for label, (score_col, pts_col) in strat_cols.items():
        pts = detail[pts_col].sum()
        exacts = (detail[score_col] == detail["actual"]).sum()
        acc = detail.apply(lambda r: res_ok(r[score_col], r["actual"]), axis=1).mean()
        summ.append({"strategy": label, "total_pts": pts, "avg_pts": pts / len(detail),
                     "result_acc": acc, "exact_hits": exacts})
    summary = pd.DataFrame(summ).sort_values("total_pts", ascending=False)
    return detail, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonus", type=float, default=0.0,
                        help="average exact-score bonus (default 0 = base only)")
    args = parser.parse_args()

    detail, summary = run_backtest(bonus=args.bonus)
    print(f"MPP backtest on {len(detail)} played matches (bonus={args.bonus}):\n")
    print(summary.to_string(index=False, formatters={
        "total_pts": "{:.0f}".format, "avg_pts": "{:.1f}".format,
        "result_acc": "{:.1%}".format, "exact_hits": "{:.0f}".format,
    }))
