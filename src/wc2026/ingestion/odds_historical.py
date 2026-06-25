"""Historical international odds from the eatpizzanot/soccer-dataset GitHub repo.

The repo aggregates odds from:
  - Football-Data.co.uk CSV dumps (club football)
  - The Odds API snapshots (June 2020 onward) — what we need for international
    tournaments.

We download fixtures.parquet, odds.parquet, and leagues.parquet from main,
filter to international competitions, and emit a tidy parquet with the same
(date, home_team, away_team, odds_home, odds_draw, odds_away) schema as the
live Odds-API table from ``odds.py``. This lets M9_odds use either source
transparently.

Coverage of our 6-tournament backtest (as of 2026-06):
  WC 2018          : 0
  WC 2022          : 56 / 64
  Euro 2020        : 44 / 51
  Euro 2024        : 44 / 51
  Copa 2021        : 0
  Copa 2024        : 26 / 32
  Total            : 170 matches with odds

That's enough for a stable M9 backtest at the tournament level.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from wc2026.ingestion.teams import normalize

REPO = "eatpizzanot/soccer-dataset"
RAW_BASE = f"https://github.com/{REPO}/raw/main/parquet"


def download_all(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in ("fixtures.parquet", "odds.parquet", "leagues.parquet"):
            target = raw_dir / f"eatpizzanot_{name}"
            r = client.get(f"{RAW_BASE}/{name}")
            r.raise_for_status()
            target.write_bytes(r.content)
            paths[name] = target
    return paths


def build_historical_odds(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Filter to international tournaments and emit a tidy DataFrame.

    Output schema (one row per match):
        date          : match date (datetime64, no tz)
        league_name   : e.g. "FIFA World Cup", "UEFA Euro"
        home_team     : normalised
        away_team     : normalised
        odds_home, odds_draw, odds_away
        bookmaker     : whichever single bookmaker the dataset captured
    """
    fixtures = pd.read_parquet(raw_dir / "eatpizzanot_fixtures.parquet")
    odds = pd.read_parquet(raw_dir / "eatpizzanot_odds.parquet")
    leagues = pd.read_parquet(raw_dir / "eatpizzanot_leagues.parquet")

    intl_pattern = r"World Cup|Euro|Copa|Nations|Africa|Asian"
    intl_ids = leagues.loc[
        leagues["name"].str.contains(intl_pattern, case=False, na=False), "id"
    ].tolist()

    intl_fix = fixtures[fixtures["league_id"].isin(intl_ids)].copy()
    merged = odds.merge(
        intl_fix[["id", "date", "league_name", "home_team", "away_team",
                  "goals_home", "goals_away"]],
        left_on="fixture_id",
        right_on="id",
        suffixes=("_o", "_f"),
    )

    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged["home_team"] = merged["home_team"].map(normalize)
    merged["away_team"] = merged["away_team"].map(normalize)

    # Source column names: home_win / draw / away_win → standardise
    return merged[
        ["date", "league_name", "home_team", "away_team",
         "home_win", "draw", "away_win", "bookmaker",
         "goals_home", "goals_away"]
    ].rename(columns={"home_win": "odds_home", "draw": "odds_draw", "away_win": "odds_away"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/odds_historical.parquet"))
    parser.add_argument("--download", action="store_true",
                        help="Download fresh parquets from GitHub (otherwise use cached)")
    args = parser.parse_args()

    if args.download or not (args.raw / "eatpizzanot_fixtures.parquet").exists():
        print("Downloading eatpizzanot dataset...")
        for name, p in download_all(args.raw).items():
            print(f"  {name}: {p.stat().st_size:,} bytes")

    df = build_historical_odds(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(df):,} historical odds rows -> {args.out}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Bookmakers: {df['bookmaker'].nunique()} ({', '.join(sorted(df['bookmaker'].unique()))})")
    print()
    print("Coverage by tournament:")
    print(df.groupby('league_name').size().sort_values(ascending=False).to_string())
