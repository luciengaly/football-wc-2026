"""Streamlit dashboard for the WC 2026 prediction system.

Pages:
  - 🏆 Tournament      : groups, calendar, current standings
  - 🔮 Predictions     : upcoming matches with model probabilities
  - 📊 Performance     : log-loss / Brier / RPS on played matches (cumulative)
  - 🎲 Simulation      : (placeholder — built in S4)

Launch:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path so `import wc2026...` works when launched via Streamlit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc2026.evaluation.cumulative import cumulative_metrics, cumulative_score_metrics  # noqa: E402
from wc2026.evaluation.metrics import metrics_summary  # noqa: E402
from wc2026.models.base import encode_outcome  # noqa: E402

DATA_DIR = ROOT / "data"
PROCESSED = DATA_DIR / "processed"
SNAPSHOTS = DATA_DIR / "snapshots"

st.set_page_config(page_title="WC 2026 Predictor", page_icon="🏆", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_wc2026() -> pd.DataFrame:
    path = PROCESSED / "wc2026.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_latest_snapshot() -> pd.DataFrame:
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    files = sorted(SNAPSHOTS.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


@st.cache_data(ttl=300)
def load_all_snapshots() -> pd.DataFrame:
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(SNAPSHOTS.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["snapshot_date"] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar — global navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏆 WC 2026 Predictor")
page = st.sidebar.radio(
    "Page",
    ["🏆 Tournament", "🔮 Predictions", "🎯 Score detail", "🃏 Conseil MPP",
     "📊 Performance", "ℹ️ About"],
)

wc = load_wc2026()
latest = load_latest_snapshot()

if wc.empty:
    st.warning(
        "No WC 2026 dataset found. Run "
        "`python -m wc2026.pipeline refresh --as-of 2026-06-19` first."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Page: Tournament overview
# ---------------------------------------------------------------------------
def page_tournament() -> None:
    st.title("🏆 World Cup 2026 — Overview")

    n_played = (wc["status"] == "played").sum()
    n_total = len(wc)
    cols = st.columns(4)
    cols[0].metric("Total matches", n_total)
    cols[1].metric("Played", int(n_played))
    cols[2].metric("Scheduled", int(n_total - n_played))
    cols[3].metric("Teams", len(set(wc["home_team"]) | set(wc["away_team"])))

    st.divider()
    st.subheader("Groups")

    group_stage = wc[wc["stage"] == "group"].copy()
    standings = compute_group_standings(group_stage)

    # 3-column layout of groups
    groups = sorted(g for g in standings["group"].unique() if g is not None)
    n_cols = 3
    for i in range(0, len(groups), n_cols):
        row_cols = st.columns(n_cols)
        for col, g in zip(row_cols, groups[i : i + n_cols], strict=False):
            with col:
                sub = standings[standings["group"] == g].drop(columns=["group"])
                st.markdown(f"**Group {g}**")
                st.dataframe(sub, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Schedule")
    show_cols = ["date", "stage", "group", "home_team", "home_score",
                 "away_score", "away_team", "status"]
    schedule = wc[show_cols].copy()
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.date
    st.dataframe(schedule, hide_index=True, use_container_width=True, height=500)


def compute_group_standings(group_matches: pd.DataFrame) -> pd.DataFrame:
    """Compute current group standings (points, GF, GA, GD)."""
    played = group_matches[group_matches["status"] == "played"].copy()
    rows = []
    for g, sub in group_matches.groupby("group"):
        teams = set(sub["home_team"]) | set(sub["away_team"])
        for team in teams:
            home = played[(played["group"] == g) & (played["home_team"] == team)]
            away = played[(played["group"] == g) & (played["away_team"] == team)]
            gp = len(home) + len(away)
            gf = int(home["home_score"].sum() + away["away_score"].sum())
            ga = int(home["away_score"].sum() + away["home_score"].sum())
            w = ((home["home_score"] > home["away_score"]).sum()
                 + (away["away_score"] > away["home_score"]).sum())
            d = ((home["home_score"] == home["away_score"]).sum()
                 + (away["away_score"] == away["home_score"]).sum())
            l_ = gp - w - d
            pts = w * 3 + d
            rows.append({
                "group": g,
                "team": team,
                "P": gp, "W": int(w), "D": int(d), "L": int(l_),
                "GF": gf, "GA": ga, "GD": gf - ga, "Pts": int(pts),
            })
    df = pd.DataFrame(rows).sort_values(
        ["group", "Pts", "GD", "GF"], ascending=[True, False, False, False]
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Page: Predictions
# ---------------------------------------------------------------------------
# Designated models (configuration — the live ranking lives on the Performance
# page, the rationale + numbers in docs/DECISIONS.md):
#   - W/D/L            -> M13_blend3 (results + odds + injury-adjusted value)
#   - Score distribution -> M2 (odds and value give no score distribution)
CHAMPION_MODEL = "M13_blend3"
SCORE_CHAMPION_MODEL = "M2_poisson"


def page_predictions() -> None:
    st.title("🔮 Match Predictions")

    if latest.empty:
        st.warning("No predictions yet. Run "
                   "`python -m wc2026.pipeline predict --as-of 2026-06-19`.")
        return

    snap_date = latest["as_of"].iloc[0] if "as_of" in latest else "—"
    available_models = sorted(latest["model"].unique().tolist())

    # Move champion to the top of the dropdown when present
    if CHAMPION_MODEL in available_models:
        ordered = [CHAMPION_MODEL] + [m for m in available_models if m != CHAMPION_MODEL]
    else:
        ordered = available_models

    selected_model = st.selectbox(
        "Model",
        ordered,
        index=0,
        format_func=lambda m: (
            f"⭐ {m} (W/D/L champion)" if m == CHAMPION_MODEL
            else f"📊 {m} (score champion)" if m == SCORE_CHAMPION_MODEL
            else m
        ),
    )
    st.caption(
        f"Snapshot: **{snap_date}** · ⭐ **{CHAMPION_MODEL}** — W/D/L "
        f"(blend résultats + cotes + valeur d'effectif) · "
        f"📊 **{SCORE_CHAMPION_MODEL}** — distribution de scores. "
        f"Classement live sur la page Performance. "
        f"Modèles disponibles : {', '.join(available_models)}"
    )

    rows = latest[latest["model"] == selected_model].sort_values("date").copy()
    if rows.empty:
        st.warning(f"Model {selected_model} not in snapshot.")
        return
    rows["date"] = pd.to_datetime(rows["date"]).dt.date

    if "score_mode_h" in rows.columns and rows["score_mode_h"].notna().any():
        # Modal score (integer) on this page; E[H]:E[A] decimal lives on Score detail.
        rows["Score"] = (
            rows["score_mode_h"].astype("Int64").astype(str)
            + " – "
            + rows["score_mode_a"].astype("Int64").astype(str)
        )
        cols = ["date", "stage", "group", "home_team", "away_team",
                "p_home", "p_draw", "p_away", "Score", "p_btts", "p_over_2_5"]
        if "p_home_advances" in rows.columns and rows["p_home_advances"].notna().any():
            cols += ["p_home_advances", "p_away_advances"]
        show = rows[cols].rename(columns={
            "p_home": "P(H)", "p_draw": "P(D)", "p_away": "P(A)",
            "p_btts": "BTTS", "p_over_2_5": "O2.5",
            "p_home_advances": "P(H adv)", "p_away_advances": "P(A adv)",
        })
        fmt = {"P(H)": "{:.0%}", "P(D)": "{:.0%}", "P(A)": "{:.0%}",
               "BTTS": "{:.0%}", "O2.5": "{:.0%}",
               "P(H adv)": "{:.0%}", "P(A adv)": "{:.0%}"}
    else:
        # Model has no score distribution (e.g. M1 Elo, M4 LightGBM W/D/L only)
        cols = ["date", "stage", "group", "home_team", "away_team",
                "p_home", "p_draw", "p_away"]
        if "p_home_advances" in rows.columns and rows["p_home_advances"].notna().any():
            cols += ["p_home_advances", "p_away_advances"]
        show = rows[cols].rename(columns={
            "p_home": "P(H)", "p_draw": "P(D)", "p_away": "P(A)",
            "p_home_advances": "P(H adv)", "p_away_advances": "P(A adv)",
        })
        fmt = {"P(H)": "{:.0%}", "P(D)": "{:.0%}", "P(A)": "{:.0%}",
               "P(H adv)": "{:.0%}", "P(A adv)": "{:.0%}"}

    # Single 0%→white / 100%→dark-green scale, shared across all probability
    # columns so the colours are directly comparable cell-to-cell. Default
    # pandas behaviour colours each column independently (min→red, max→green
    # of that column), which was confusing for probability columns where the
    # max naturally varies.
    proba_cols = [c for c in ["P(H)", "P(D)", "P(A)", "P(H adv)", "P(A adv)"]
                  if c in show.columns]
    st.dataframe(
        show.style.format(fmt).background_gradient(
            subset=proba_cols, cmap="Greens", vmin=0.0, vmax=1.0, axis=None,
        ),
        hide_index=True,
        use_container_width=True,
        height=620,
    )


# ---------------------------------------------------------------------------
# Page: Performance
# ---------------------------------------------------------------------------
def page_performance() -> None:
    st.title("📊 Model Performance")
    st.caption(
        "Live performance on played WC 2026 matches. Each match is evaluated "
        "against the prediction made on the snapshot of the match day itself "
        "(`as_of == match_date`) — the model's last word before kickoff."
    )
    all_snaps = load_all_snapshots()
    if all_snaps.empty:
        st.info("No snapshots yet. Run `python -m wc2026.pipeline backfill` "
                "to reconstruct them for past WC days.")
        return

    played_wc = wc[wc["status"] == "played"].copy()
    if played_wc.empty:
        st.info("No WC matches played yet.")
        return

    # Predictions made on the match's own day (as_of == match_date)
    all_snaps["as_of"] = pd.to_datetime(all_snaps["as_of"]).dt.normalize()
    all_snaps["date"] = pd.to_datetime(all_snaps["date"]).dt.normalize()
    same_day = all_snaps[all_snaps["as_of"] == all_snaps["date"]].copy()

    played_keys = played_wc[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    played_keys["date"] = pd.to_datetime(played_keys["date"]).dt.normalize()

    merged = same_day.merge(
        played_keys.rename(columns={"home_score": "true_home", "away_score": "true_away"}),
        on=["date", "home_team", "away_team"],
        how="inner",
    )
    if merged.empty:
        st.info("No matched predictions yet. Run backfill to align snapshots with match days.")
        return
    merged["outcome"] = encode_outcome(merged["true_home"], merged["true_away"]).to_numpy()
    merged = merged.dropna(subset=["outcome"])

    st.subheader("W/D/L metrics (pooled over played WC 2026 matches)")
    rows = []
    for model_name, grp in merged.groupby("model"):
        m = metrics_summary(grp["outcome"].tolist(),
                            grp[["p_home", "p_draw", "p_away"]])
        rows.append({"model": model_name, "n_matches": len(grp), **m})
    perf = pd.DataFrame(rows).sort_values("log_loss")
    st.dataframe(
        perf.style.format({
            "log_loss": "{:.4f}", "brier": "{:.4f}", "rps": "{:.4f}",
            "accuracy": "{:.2%}", "ece": "{:.4f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # --- Cumulative trajectory chart ---------------------------------------
    import plotly.express as px

    st.subheader("Cumulative metrics over time")
    cumul = cumulative_metrics(merged)
    if not cumul.empty:
        metric = st.selectbox(
            "Metric",
            ["log_loss", "rps", "brier", "accuracy", "ece"],
            index=0,
        )
        fig = px.line(
            cumul,
            x="date",
            y=metric,
            color="model",
            markers=True,
            title=f"Cumulative {metric} as matches are played",
        )
        fig.update_layout(
            height=400,
            yaxis_title=metric,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Score-level metrics filtered to the score champion (M2 — best on exact-acc)
    has_scores = "score_mode_h" in merged.columns and merged["score_mode_h"].notna().any()
    if has_scores:
        st.subheader(f"Score-level live metrics ({SCORE_CHAMPION_MODEL} — score champion)")
        m2 = merged[merged["model"] == SCORE_CHAMPION_MODEL].dropna(subset=["score_mode_h"]).copy()
        if not m2.empty:
            m2["score_correct"] = (
                (m2["score_mode_h"].astype(int) == m2["true_home"].astype(int))
                & (m2["score_mode_a"].astype(int) == m2["true_away"].astype(int))
            )
            cols = st.columns(4)
            cols[0].metric("Matches", len(m2))
            cols[1].metric("Exact-score accuracy", f"{m2['score_correct'].mean():.1%}")
            mae_h = (m2["e_h"] - m2["true_home"].astype(float)).abs().mean()
            mae_a = (m2["e_a"] - m2["true_away"].astype(float)).abs().mean()
            cols[2].metric("MAE home goals", f"{mae_h:.2f}")
            cols[3].metric("MAE away goals", f"{mae_a:.2f}")

            # Cumulative score trajectory (score champion only)
            score_traj = cumulative_score_metrics(
                merged[merged["model"] == SCORE_CHAMPION_MODEL].assign(
                    true_home=lambda d: d["true_home"],
                    true_away=lambda d: d["true_away"],
                )
            )
            if not score_traj.empty:
                metric = st.radio(
                    "Score-level metric",
                    ["exact_acc", "mae_h", "mae_a"],
                    horizontal=True,
                )
                fig = px.line(
                    score_traj, x="date", y=metric, markers=True,
                    title=f"Cumulative {metric} (score champion = {SCORE_CHAMPION_MODEL})",
                )
                fig.update_layout(height=350, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Match-by-match (champion):**")
            show = m2[["date", "home_team", "away_team",
                       "true_home", "true_away",
                       "score_mode_h", "score_mode_a", "e_h", "e_a",
                       "p_home", "p_draw", "p_away"]].rename(columns={
                "true_home": "Actual H", "true_away": "Actual A",
                "score_mode_h": "Pred H", "score_mode_a": "Pred A",
                "e_h": "E[H]", "e_a": "E[A]",
                "p_home": "P(H)", "p_draw": "P(D)", "p_away": "P(A)",
            })
            show["date"] = show["date"].dt.date
            st.dataframe(
                show.style.format({
                    "E[H]": "{:.2f}", "E[A]": "{:.2f}",
                    "P(H)": "{:.0%}", "P(D)": "{:.0%}", "P(A)": "{:.0%}",
                }),
                hide_index=True,
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page: Score detail (per-match score distribution heatmap)
# ---------------------------------------------------------------------------
def page_score_detail() -> None:
    import plotly.express as px

    st.title("🎯 Score Distribution per Match")
    st.caption(
        "Joint score distribution P(home_goals, away_goals) under the "
        f"score-distribution model ({SCORE_CHAMPION_MODEL}). Pick a match and "
        "inspect the heatmap, expected score, and aggregated markets (BTTS, O/U 2.5)."
    )

    upcoming = wc[wc["status"] == "scheduled"].sort_values("date").copy()
    if upcoming.empty:
        st.info("No scheduled matches left — the tournament is over (or data is stale).")
        return

    upcoming["label"] = (
        upcoming["date"].dt.strftime("%Y-%m-%d")
        + " · "
        + upcoming["home_team"]
        + " vs "
        + upcoming["away_team"]
    )
    choice = st.selectbox("Match", upcoming["label"].tolist())
    row = upcoming[upcoming["label"] == choice].iloc[0]

    if st.button("Compute score distribution", type="primary"):
        from wc2026.ingestion.historical import load_results
        from wc2026.models.poisson import PoissonIndependent

        with st.spinner("Fitting M2 Poisson on all available history..."):
            results = load_results(DATA_DIR / "raw" / "results.csv")
            training = results[results["date"] < pd.Timestamp.today().normalize()].copy()
            model = PoissonIndependent()
            model.fit(training)

        match_df = pd.DataFrame([{
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "neutral": row["home_team"] not in {"United States", "Canada", "Mexico"},
        }])
        joint = model.predict_score_dist(match_df)[0]

        # Truncate display to scores 0..6 each side
        max_disp = 6
        disp = joint[: max_disp + 1, : max_disp + 1]
        disp = disp / disp.sum()

        # Heatmap
        fig = px.imshow(
            disp * 100.0,
            labels={"x": f"{row['away_team']} goals", "y": f"{row['home_team']} goals", "color": "P (%)"},
            x=list(range(max_disp + 1)),
            y=list(range(max_disp + 1)),
            origin="lower",
            color_continuous_scale="Viridis",
            text_auto=".1f",
            aspect="equal",
        )
        fig.update_layout(height=550, title=f"{row['home_team']} vs {row['away_team']} — score probabilities (%)")
        st.plotly_chart(fig, use_container_width=True)

        # Marginals & aggregates
        h_marg = joint.sum(axis=1)
        a_marg = joint.sum(axis=0)
        e_h = float((np.arange(joint.shape[0]) * h_marg).sum())
        e_a = float((np.arange(joint.shape[1]) * a_marg).sum())
        most_prob = np.unravel_index(joint.argmax(), joint.shape)
        p_most = float(joint[most_prob])

        h_idx, a_idx = np.indices(joint.shape)
        btts = float(joint[(h_idx > 0) & (a_idx > 0)].sum())
        over25 = float(joint[(h_idx + a_idx) > 2].sum())

        cols = st.columns(5)
        cols[0].metric("Expected score", f"{e_h:.2f} – {e_a:.2f}")
        cols[1].metric("Modal score", f"{most_prob[0]} – {most_prob[1]}", f"{p_most:.1%}")
        cols[2].metric("BTTS yes", f"{btts:.1%}")
        cols[3].metric("Over 2.5 goals", f"{over25:.1%}")
        p_h = float(np.tril(joint, k=-1).sum())
        p_d = float(np.diag(joint).sum())
        p_a = float(np.triu(joint, k=1).sum())
        cols[4].metric("W/D/L", f"{p_h:.0%}/{p_d:.0%}/{p_a:.0%}")

        # Top-5 most likely scores — reveals the spread that the mode masks
        flat = joint.flatten()
        top5 = np.argpartition(-flat, 5)[:5]
        top5 = top5[np.argsort(-flat[top5])]
        top5_rows = []
        for idx in top5:
            h, a = np.unravel_index(idx, joint.shape)
            top5_rows.append({"score": f"{h} – {a}", "probability": float(joint[h, a])})
        st.markdown("**Top-5 most-likely scores**")
        st.dataframe(
            pd.DataFrame(top5_rows).style.format({"probability": "{:.1%}"}),
            hide_index=True,
            use_container_width=False,
        )
    else:
        st.info("Click *Compute score distribution* to render the heatmap.")


# ---------------------------------------------------------------------------
# Page: Conseil MPP (betting advisor)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _fit_mpp_models(as_of_str: str):
    """Fit P(result)=M13 and P(score)=M2 on data strictly before as_of. Cached."""
    from wc2026.ingestion.historical import load_results
    from wc2026.ingestion.market_value import get_shared_store
    from wc2026.models.blend3 import Blend3
    from wc2026.models.poisson import PoissonIndependent

    results = load_results(DATA_DIR / "raw" / "results.csv")
    training = results[results["date"] < pd.Timestamp(as_of_str)].copy()
    get_shared_store()
    return Blend3().fit(training), PoissonIndependent().fit(training)


def page_mpp() -> None:
    from wc2026.mpp.optimizer import favorite_score, optimal_score
    from wc2026.mpp.scoring import result_sign

    st.title("🃏 Conseil MPP — Mon Petit Prono")
    st.caption(
        "Score à pronostiquer pour **maximiser l'espérance de points MPP** : "
        "`score* = argmax_s [ cote(R_s)·P(résultat) + bonus·P(score) ]`. "
        "P(résultat) via M13, P(score) via M2. Avantage terrain appliqué "
        "uniquement aux nations hôtes (USA/Canada/Mexique)."
    )

    cotes_path = DATA_DIR / "mpp" / "mpp_cotes.csv"
    if not cotes_path.exists():
        st.warning("Fichier `data/mpp/mpp_cotes.csv` introuvable. Remplis les cotes d'abord.")
        return
    mpp = pd.read_csv(cotes_path)
    upcoming = mpp[(mpp["status"] == "scheduled") & mpp["cote_home_win"].notna()].copy()
    if upcoming.empty:
        st.info("Aucun match à venir avec cotes renseignées dans `mpp_cotes.csv`.")
        return
    upcoming["date"] = pd.to_datetime(upcoming["date"])

    bonus = st.slider("Bonus rareté moyen (points si score exact)", 0, 100, 40, 5)

    with st.spinner("Calcul des modèles (M13 + M2)…"):
        as_of = upcoming["date"].min().strftime("%Y-%m-%d")
        m13, m2 = _fit_mpp_models(as_of)
        # authoritative neutral flag from source
        wc_n = wc[["date", "home_team", "away_team", "neutral"]].copy()
        wc_n["date"] = pd.to_datetime(wc_n["date"])
        feats = upcoming.merge(wc_n, on=["date", "home_team", "away_team"], how="left")
        feats["neutral"] = feats["neutral"].fillna(True)
        p13 = m13.predict_proba(feats).to_numpy()
        sd = m2.predict_score_dist(feats)

    rows = []
    for i, (_, m) in enumerate(feats.iterrows()):
        cotes = (m["cote_home_win"], m["cote_draw"], m["cote_away_win"])
        pr = (p13[i, 0], p13[i, 1], p13[i, 2])
        (h, a), ev = optimal_score(*cotes, pr, sd[i], bonus=bonus)
        fav = favorite_score(*cotes, sd[i])
        reco_res = {"H": m["home_team"], "D": "Nul", "A": m["away_team"]}[result_sign(h, a)]
        is_value = (h, a) != fav
        rows.append({
            "Date": m["date"].date(),
            "Match": f"{m['home_team']} – {m['away_team']}",
            "Cotes H/N/A": f"{int(cotes[0])}/{int(cotes[1])}/{int(cotes[2])}",
            "P(H/N/A)": f"{pr[0]:.0%}/{pr[1]:.0%}/{pr[2]:.0%}",
            "Score conseillé": f"{h}-{a}",
            "Pari": reco_res,
            "EV (pts)": ev,
            "💎 Value": "✅" if is_value else "",
        })
    table = pd.DataFrame(rows).sort_values("EV (pts)", ascending=False)
    st.dataframe(
        table.style.format({"EV (pts)": "{:.1f}"}).background_gradient(
            subset=["EV (pts)"], cmap="Greens"
        ),
        hide_index=True, use_container_width=True, height=560,
    )

    st.caption(
        "💎 Value = le score conseillé diffère du favori marché (cote compressée "
        "ou foule sous-cotée). C'est là que la stratégie EV gagne des places — "
        "au prix d'une précision résultat plus basse mais de plus de points sur la durée."
    )


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about() -> None:
    st.title("ℹ️ About")
    st.markdown("""
    **WC 2026 Match Prediction System** — prédiction par match (W/D/L et
    distribution de scores) pour la Coupe du Monde 2026, avec un optimiseur de
    pronostics *Mon Petit Prono*.

    **Sources de données (100% gratuites)**
    - Matchs internationaux : [martj42/international_results](https://github.com/martj42/international_results)
    - Elo : calculé en interne (formule World Football Elo)
    - Cotes : The Odds API · valeurs d'effectif & blessures : Transfermarkt

    **Approche** — plusieurs familles de modèles (Elo, Poisson/Dixon-Coles,
    gradient boosting, cotes de marché, valeur marchande) et leurs blends. Le
    modèle mis en avant pour chaque tâche est indiqué dans la page Predictions ;
    leur **classement live** est sur la page **Performance**.

    **Hors périmètre** : vainqueur du tournoi, simulation du bracket, qualifs par groupe.

    Détails et justifications : `docs/ROADMAP.md`, `docs/DECISIONS.md`, `docs/MPP.md`.

    ---
    _🎨 vibe-codé par Lucien Galy_
    """)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "🏆 Tournament":
    page_tournament()
elif page == "🔮 Predictions":
    page_predictions()
elif page == "🎯 Score detail":
    page_score_detail()
elif page == "🃏 Conseil MPP":
    page_mpp()
elif page == "📊 Performance":
    page_performance()
elif page == "ℹ️ About":
    page_about()
