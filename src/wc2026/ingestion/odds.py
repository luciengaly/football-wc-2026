"""Client for The Odds API (the-odds-api.com).

Free tier: 500 requests/month. Each "request" returns either upcoming odds
or historical scores for one sport.

The WC 2026 sport key on The Odds API is ``soccer_fifa_world_cup``.

Output schema (saved to data/raw/odds_YYYY-MM-DD.parquet):
    api_match_id, commence_time, home_team, away_team,
    bookmaker, last_update,
    odds_home, odds_draw, odds_away

We do NOT pre-aggregate across bookmakers at ingestion time — keeping the
raw per-bookmaker rows so downstream code can choose how to combine them
(mean, median, devig, etc.).

Caching:
    Each successful API call dumps both the raw JSON to data/raw/odds_*.json
    and the parsed parquet, with a date stamp. We refresh at most once per
    day to stay under quota.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from wc2026.ingestion.teams import normalize

BASE_URL = "https://api.the-odds-api.com/v4"
WC_SPORT_KEY = "soccer_fifa_world_cup"
DEFAULT_REGIONS = "eu"      # european bookmakers (Pinnacle, Bet365, William Hill, etc.)
DEFAULT_MARKETS = "h2h"     # head-to-head, i.e. match winner (W/D/L)


class OddsAPIClient:
    """Thin wrapper around The Odds API HTTP endpoints."""

    def __init__(self, api_key: str | None = None, *, timeout: float = 20.0):
        load_dotenv()  # load .env from cwd
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ODDS_API_KEY not set. Add it to .env or pass api_key=..."
            )
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    # ------------------------------------------------------------------
    def list_sports(self) -> list[dict]:
        r = self._client.get(f"{BASE_URL}/sports", params={"apiKey": self.api_key})
        r.raise_for_status()
        return r.json()

    def get_upcoming_odds(
        self,
        sport_key: str = WC_SPORT_KEY,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
    ) -> tuple[list[dict], dict]:
        """Returns (events JSON, response headers).

        Headers include quota information :
            x-requests-used, x-requests-remaining
        """
        r = self._client.get(
            f"{BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        r.raise_for_status()
        return r.json(), dict(r.headers)

    def get_scores(
        self,
        sport_key: str = WC_SPORT_KEY,
        days_from: int = 3,
    ) -> tuple[list[dict], dict]:
        """Recent completed games + scheduled in next ~24h. days_from <=3 free."""
        r = self._client.get(
            f"{BASE_URL}/sports/{sport_key}/scores",
            params={
                "apiKey": self.api_key,
                "daysFrom": days_from,
                "dateFormat": "iso",
            },
        )
        r.raise_for_status()
        return r.json(), dict(r.headers)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
def parse_odds_payload(events: list[dict]) -> pd.DataFrame:
    """Flatten the events JSON into one row per (match, bookmaker)."""
    rows = []
    for ev in events:
        api_match_id = ev.get("id")
        commence_time = ev.get("commence_time")
        home_team = normalize(ev.get("home_team", ""))
        away_team = normalize(ev.get("away_team", ""))
        for bk in ev.get("bookmakers", []):
            bookmaker = bk.get("key")
            last_update = bk.get("last_update")
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {normalize(o.get("name", "")): o.get("price")
                            for o in market.get("outcomes", [])}
                # h2h markets contain home / away / "Draw" (or "draw")
                odds_h = outcomes.get(home_team)
                odds_a = outcomes.get(away_team)
                odds_d = outcomes.get("Draw") or outcomes.get("draw")
                rows.append({
                    "api_match_id": api_match_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bookmaker,
                    "last_update": last_update,
                    "odds_home": odds_h,
                    "odds_draw": odds_d,
                    "odds_away": odds_a,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    df["match_date"] = df["commence_time"].dt.tz_convert(None).dt.normalize()
    return df


# ----------------------------------------------------------------------
# Helper: implied probabilities from decimal odds
# ----------------------------------------------------------------------
def odds_to_proba(
    odds_home: float, odds_draw: float, odds_away: float,
    method: str = "normalize",
) -> tuple[float, float, float]:
    """Convert decimal odds → (P_home, P_draw, P_away), sum to 1.

    method:
        "normalize"  — naive 1/odds renormalised (most common, simple).
        "shin"       — Shin (1993) devig (slightly more accurate but more code).
    """
    inv_h, inv_d, inv_a = 1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away
    if method == "normalize":
        s = inv_h + inv_d + inv_a
        return inv_h / s, inv_d / s, inv_a / s
    raise NotImplementedError(method)


# ----------------------------------------------------------------------
# CLI for manual refresh
# ----------------------------------------------------------------------
def fetch_and_save(raw_dir: Path) -> Path | None:
    """Fetch current WC 2026 odds and save to raw_dir. Returns parquet path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    with OddsAPIClient() as client:
        events, headers = client.get_upcoming_odds()
    print(f"Quota: used={headers.get('x-requests-used')}, "
          f"remaining={headers.get('x-requests-remaining')}")
    print(f"Events fetched: {len(events)}")

    raw_json = raw_dir / f"odds_{today}.json"
    raw_json.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"  Raw JSON -> {raw_json}")

    df = parse_odds_payload(events)
    if df.empty:
        print("  WARNING: no parsed rows (no bookmakers, or wrong sport key).")
        return None
    parquet_path = raw_dir / f"odds_{today}.parquet"
    df.to_parquet(parquet_path, index=False)
    n_matches = df[["home_team", "away_team"]].drop_duplicates().shape[0]
    n_bk = df["bookmaker"].nunique()
    print(f"  Parquet -> {parquet_path}  ({n_matches} matches × {n_bk} bookmakers)")
    return parquet_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    fetch_and_save(args.out)
