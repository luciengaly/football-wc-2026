"""M9_odds — market-implied probabilities from The Odds API.

This isn't a trained model. It looks up the latest odds for each requested
match, devigs them, and returns calibrated W/D/L probabilities. The market
is well known as the most predictive single signal in football (log-loss
typically 0.85-0.90 vs 0.95 for the best academic models), so M9 is our
benchmark for "best possible single-signal prediction with our data".

Aggregation across bookmakers: median odds per outcome, then proportional
devig (1/odds renormalised). The median is more robust than the mean
against bookmaker quirks (mispricing, slow updates).

Fallback: if a match has no odds (e.g. R32 fixtures not yet open by
bookmakers), the model returns NaN for that match. Calling code can choose
to fall back to a generative model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from wc2026.ingestion.odds import odds_to_proba
from wc2026.ingestion.teams import normalize
from wc2026.models.base import Model


class OddsModel(Model):
    """M9_odds — direct market-implied probabilities."""

    name = "M9_odds"

    def __init__(self, odds_dir: Path | str = "data/raw"):
        self.odds_dir = Path(odds_dir)
        self.odds_df_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def _load_latest_odds(self) -> pd.DataFrame:
        files = sorted(self.odds_dir.glob("odds_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No odds_*.parquet file in {self.odds_dir}. "
                "Run `python -m wc2026.ingestion.odds` first."
            )
        return pd.read_parquet(files[-1])

    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> Self:
        # No training — just load the latest odds snapshot
        self.odds_df_ = self._load_latest_odds()
        return self

    # ------------------------------------------------------------------
    def _aggregate(self, sub: pd.DataFrame) -> tuple[float, float, float] | None:
        sub = sub.dropna(subset=["odds_home", "odds_draw", "odds_away"])
        if sub.empty:
            return None
        m_h = float(sub["odds_home"].median())
        m_d = float(sub["odds_draw"].median())
        m_a = float(sub["odds_away"].median())
        return odds_to_proba(m_h, m_d, m_a, method="normalize")

    def predict_proba(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.odds_df_ is None:
            raise RuntimeError("Model not fitted (call .fit() first)")

        # Normalise team names defensively (in case of upstream skips)
        odds = self.odds_df_.copy()
        odds["home_team"] = odds["home_team"].map(normalize)
        odds["away_team"] = odds["away_team"].map(normalize)
        odds["match_date"] = pd.to_datetime(odds["match_date"]).dt.normalize()

        # martj42 dates the match by *local kickoff date*. The Odds API returns
        # UTC commence_time, which can be +1 day for evening kickoffs in the
        # Americas (where WC 2026 is hosted). We match teams exactly and allow
        # date ±1 day to absorb the timezone shift.
        rows = []
        for m in matches.itertuples(index=False):
            mdate = pd.Timestamp(m.date).normalize()
            home = normalize(m.home_team)
            away = normalize(m.away_team)
            date_window = (
                (odds["match_date"] >= mdate - pd.Timedelta(days=1))
                & (odds["match_date"] <= mdate + pd.Timedelta(days=1))
            )
            sub = odds[date_window & (odds["home_team"] == home) & (odds["away_team"] == away)]
            agg = self._aggregate(sub)
            if agg is None:
                # Try same teams reversed (some APIs flip home/away on neutrals)
                sub_rev = odds[date_window & (odds["home_team"] == away) & (odds["away_team"] == home)]
                agg_rev = self._aggregate(sub_rev)
                if agg_rev is not None:
                    p_h, p_d, p_a = agg_rev
                    rows.append((p_a, p_d, p_h))
                    continue
                rows.append((np.nan, np.nan, np.nan))
            else:
                rows.append(agg)

        return pd.DataFrame(
            rows, columns=["p_home", "p_draw", "p_away"], index=matches.index
        )

    def n_covered(self, matches: pd.DataFrame) -> int:
        """How many of the requested matches have odds coverage."""
        proba = self.predict_proba(matches)
        return int(proba["p_home"].notna().sum())
