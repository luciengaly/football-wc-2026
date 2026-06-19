"""Download and load the martj42 international football results dataset.

Source: https://github.com/martj42/international_results
Files used:
  - results.csv      : all international matches since 1872
  - shootouts.csv    : penalty shootout outcomes
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from wc2026.ingestion.teams import normalize

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
RESULTS_URL = f"{BASE}/results.csv"
SHOOTOUTS_URL = f"{BASE}/shootouts.csv"
GOALSCORERS_URL = f"{BASE}/goalscorers.csv"


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        r = client.get(url)
        r.raise_for_status()
        target.write_bytes(r.content)
    return target


def download_all(raw_dir: Path) -> dict[str, Path]:
    """Download the three CSV files into raw_dir. Returns {name: path}."""
    paths = {
        "results": _download(RESULTS_URL, raw_dir / "results.csv"),
        "shootouts": _download(SHOOTOUTS_URL, raw_dir / "shootouts.csv"),
        "goalscorers": _download(GOALSCORERS_URL, raw_dir / "goalscorers.csv"),
    }
    return paths


def load_results(path: Path) -> pd.DataFrame:
    """Load and clean the results.csv file.

    Returns a DataFrame with columns:
      date (datetime), home_team, away_team, home_score, away_score,
      tournament, city, country, neutral (bool)

    Team names are normalized.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalize)
    df["away_team"] = df["away_team"].map(normalize)
    # Ensure types
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    df["neutral"] = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_shootouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalize)
    df["away_team"] = df["away_team"].map(normalize)
    df["winner"] = df["winner"].map(normalize)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    print(f"Downloading martj42 dataset into {args.out}/ ...")
    paths = download_all(args.out)
    for name, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"  {name:12s} -> {p}  ({size_kb:,.1f} KB)")

    df = load_results(paths["results"])
    print(f"\nResults: {len(df):,} matches from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Tournaments: {df['tournament'].nunique()} unique")
    print(f"Teams (home+away unique): {len(set(df['home_team']) | set(df['away_team'])):,}")
