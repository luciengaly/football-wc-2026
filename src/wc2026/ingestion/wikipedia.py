"""Scrape WC 2026 calendar and live results from Wikipedia.

Strategy: hit the dedicated sub-pages (group stage, knockout stage) and
parse the schedule tables. Wikipedia tables for football tournaments
follow a predictable shape — but the markup does change over time, so
this module is intentionally defensive and verbose about what it found.

Output schema (data/raw/wc2026_matches.csv):
  match_id, date, kickoff_utc, stage, group, home_team, away_team,
  home_score, away_score, venue, city, status
    status in {"played", "scheduled", "live", "postponed"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as date_t
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from wc2026.ingestion.teams import normalize

USER_AGENT = "wc2026-predictor/0.1 (research; lucien.galy@trad.fr)"

MAIN_PAGE = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
GROUP_PAGE = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_group_stage"
KO_PAGE = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"


@dataclass
class FetchResult:
    url: str
    status_code: int
    html: str | None = None


def _fetch(url: str) -> FetchResult:
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=30.0, headers=headers) as c:
        r = c.get(url)
    return FetchResult(url=url, status_code=r.status_code, html=r.text if r.status_code == 200 else None)


_SCORE_RE = re.compile(r"^\s*(\d+)\s*[-–:]\s*(\d+)\s*(?:\(.*\))?\s*$")


def _parse_score(cell: str) -> tuple[int | None, int | None, str]:
    """Extract score from a Wikipedia match cell.

    Returns (home_score, away_score, status). Status is 'played' if a score
    is found, 'scheduled' otherwise (cell will then contain something like
    'v', '—', a time, or be empty).
    """
    if not isinstance(cell, str):
        return None, None, "scheduled"
    cell = cell.strip()
    m = _SCORE_RE.match(cell)
    if m:
        return int(m.group(1)), int(m.group(2)), "played"
    return None, None, "scheduled"


def parse_group_stage(html: str) -> pd.DataFrame:
    """Extract group-stage matches from the group-stage page HTML.

    Wikipedia formats each group's matches inside a ``<div class="vevent">``
    block (one per match) or in summary tables. We try the structured
    ``vevent`` approach first as it's robust.
    """
    soup = BeautifulSoup(html, "lxml")
    matches: list[dict] = []

    # Approach 1: vevent blocks (one per match)
    for v in soup.select("div.vevent, table.vevent"):
        try:
            home_el = v.select_one(".fhome, .home, .team-home, .team1")
            away_el = v.select_one(".faway, .away, .team-away, .team2")
            score_el = v.select_one(".fscore, .score")
            date_el = v.select_one(".dtstart")

            if not (home_el and away_el):
                continue

            home = normalize(home_el.get_text(strip=True))
            away = normalize(away_el.get_text(strip=True))
            score_txt = score_el.get_text(strip=True) if score_el else ""
            hs, as_, status = _parse_score(score_txt)
            date_txt = date_el.get("datetime") if date_el else None
            d = pd.to_datetime(date_txt, errors="coerce") if date_txt else pd.NaT

            matches.append(
                {
                    "date": d,
                    "stage": "group",
                    "group": None,  # filled later by surrounding header if possible
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": as_,
                    "status": status,
                }
            )
        except (AttributeError, ValueError):
            continue

    return pd.DataFrame(matches)


def parse_knockout(html: str) -> pd.DataFrame:
    """Extract knockout matches from the KO-stage page HTML.

    Same vevent strategy as group stage, with stage labelled from bracket
    headings (Round of 32, Round of 16, Quarter-finals, Semi-finals,
    Third-place play-off, Final).
    """
    soup = BeautifulSoup(html, "lxml")
    matches: list[dict] = []

    for v in soup.select("div.vevent, table.vevent"):
        try:
            home_el = v.select_one(".fhome, .home, .team-home, .team1")
            away_el = v.select_one(".faway, .away, .team-away, .team2")
            score_el = v.select_one(".fscore, .score")
            date_el = v.select_one(".dtstart")

            if not (home_el and away_el):
                continue

            home = normalize(home_el.get_text(strip=True))
            away = normalize(away_el.get_text(strip=True))
            score_txt = score_el.get_text(strip=True) if score_el else ""
            hs, as_, status = _parse_score(score_txt)
            date_txt = date_el.get("datetime") if date_el else None
            d = pd.to_datetime(date_txt, errors="coerce") if date_txt else pd.NaT

            matches.append(
                {
                    "date": d,
                    "stage": "knockout",
                    "group": None,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": as_,
                    "status": status,
                }
            )
        except (AttributeError, ValueError):
            continue

    return pd.DataFrame(matches)


def fetch_wc2026(raw_dir: Path) -> pd.DataFrame:
    """Download and parse WC 2026 matches. Writes raw HTML and parsed CSV.

    Returns the parsed match DataFrame.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {GROUP_PAGE} ...")
    group = _fetch(GROUP_PAGE)
    print(f"  status={group.status_code}, html_size={len(group.html or ''):,} chars")
    if group.html:
        (raw_dir / "wikipedia_group_stage.html").write_text(group.html, encoding="utf-8")

    print(f"Fetching {KO_PAGE} ...")
    ko = _fetch(KO_PAGE)
    print(f"  status={ko.status_code}, html_size={len(ko.html or ''):,} chars")
    if ko.html:
        (raw_dir / "wikipedia_knockout_stage.html").write_text(ko.html, encoding="utf-8")

    dfs: list[pd.DataFrame] = []
    if group.html:
        g_df = parse_group_stage(group.html)
        print(f"  parsed {len(g_df)} group-stage matches")
        dfs.append(g_df)
    if ko.html:
        k_df = parse_knockout(ko.html)
        print(f"  parsed {len(k_df)} knockout matches")
        dfs.append(k_df)

    if not dfs:
        print("WARNING: no matches parsed. Wikipedia structure may have changed.")
        return pd.DataFrame()

    matches = pd.concat(dfs, ignore_index=True)
    matches["match_id"] = range(1, len(matches) + 1)

    out_csv = raw_dir / "wc2026_matches.csv"
    matches.to_csv(out_csv, index=False)
    print(f"Wrote {len(matches)} matches to {out_csv}")
    return matches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    df = fetch_wc2026(args.out)
    if not df.empty:
        n_played = (df["status"] == "played").sum()
        n_sched = (df["status"] == "scheduled").sum()
        print(f"\nSummary: {n_played} played, {n_sched} scheduled")
        if "date" in df.columns and df["date"].notna().any():
            print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
