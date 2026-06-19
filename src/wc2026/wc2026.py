"""Extract and structure the WC 2026 match dataset from martj42.

Tournament structure (first edition with 48 teams):
- 12 groups (A-L) of 4 teams → 72 group-stage matches.
- Top 2 of each group + 8 best 3rd-placed teams → 32 teams in Round of 32.
- Knockout: R32 → R16 → QF → SF → Final + 3rd-place play-off.
- Total matches: 72 + 16 + 8 + 4 + 2 + 1 + 1 = 104.

This module:
- Filters WC 2026 matches from the martj42 results.
- Detects the group of each team via graph connectivity within MD1+MD2+MD3.
- Tags each match with stage (group / R32 / R16 / QF / SF / F / 3rd).
- Exposes a tidy DataFrame `data/processed/wc2026.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc2026.ingestion.historical import load_results

TOURNAMENT_START = pd.Timestamp("2026-06-11")
TOURNAMENT_END = pd.Timestamp("2026-07-19")

# Group stage spans the first ~17 days of the tournament.
GROUP_STAGE_END = pd.Timestamp("2026-06-27")


def _filter_wc(results: pd.DataFrame) -> pd.DataFrame:
    is_wc = results["tournament"] == "FIFA World Cup"
    in_window = (results["date"] >= TOURNAMENT_START) & (results["date"] <= TOURNAMENT_END)
    df = results[is_wc & in_window].copy()
    return df.reset_index(drop=True)


def _detect_groups(group_stage_matches: pd.DataFrame) -> dict[str, str]:
    """Detect group membership via connected components.

    Each group's 4 teams play a round-robin (6 matches), so the matches
    among the 4 teams form a connected subgraph isolated from other groups.
    """
    teams = set(group_stage_matches["home_team"]) | set(group_stage_matches["away_team"])
    parent = dict.fromkeys(teams, "")
    for t in teams:
        parent[t] = t

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for m in group_stage_matches.itertuples(index=False):
        union(m.home_team, m.away_team)

    # Map each root -> group letter
    roots = sorted({find(t) for t in teams})
    if len(roots) != 12:
        # Fallback: don't assign groups if detection failed
        return {}

    group_letters = "ABCDEFGHIJKL"
    root_to_letter = dict(zip(roots, group_letters, strict=True))
    return {team: root_to_letter[find(team)] for team in teams}


def _classify_stage(date: pd.Timestamp) -> str:
    """Best-effort stage classification based on match date.

    Calendar reference (FIFA 2026 schedule):
      - Group stage:      11–27 June
      - Round of 32:      28 June – 3 July
      - Round of 16:      4 – 7 July
      - Quarter-finals:   9 – 11 July
      - Semi-finals:      14 – 15 July
      - 3rd place:        18 July
      - Final:            19 July
    """
    if date <= pd.Timestamp("2026-06-27"):
        return "group"
    if date <= pd.Timestamp("2026-07-03"):
        return "R32"
    if date <= pd.Timestamp("2026-07-07"):
        return "R16"
    if date <= pd.Timestamp("2026-07-11"):
        return "QF"
    if date <= pd.Timestamp("2026-07-15"):
        return "SF"
    if date == pd.Timestamp("2026-07-18"):
        return "3rd"
    return "F"


def build_wc2026_dataset(results_path: Path) -> pd.DataFrame:
    """Build the WC 2026 match dataset from martj42 results.

    Output columns:
      match_id, date, stage, group, home_team, away_team,
      home_score, away_score, neutral, status
    """
    results = load_results(results_path)
    wc = _filter_wc(results)

    # Status from score presence
    wc["status"] = wc["home_score"].notna().map({True: "played", False: "scheduled"})

    # Stage classification
    wc["stage"] = wc["date"].map(_classify_stage)

    # Group detection from group-stage matches only
    group_matches = wc[wc["stage"] == "group"]
    team_to_group = _detect_groups(group_matches)
    wc["group_home"] = wc["home_team"].map(team_to_group)
    wc["group_away"] = wc["away_team"].map(team_to_group)
    wc["group"] = wc["group_home"].where(wc["stage"] == "group", None)

    wc = wc.sort_values(["date", "home_team"]).reset_index(drop=True)
    wc["match_id"] = range(1, len(wc) + 1)

    cols = [
        "match_id", "date", "stage", "group", "home_team", "away_team",
        "home_score", "away_score", "neutral", "status",
    ]
    return wc[cols]


def qualified_teams(wc: pd.DataFrame) -> pd.DataFrame:
    """Return the 48 qualified teams with their group letter."""
    group_stage = wc[wc["stage"] == "group"]
    teams = pd.concat([
        group_stage[["home_team", "group"]].rename(columns={"home_team": "team"}),
        group_stage[["away_team", "group"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates("team").sort_values(["group", "team"]).reset_index(drop=True)
    return teams


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("data/raw/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/wc2026.parquet"))
    args = parser.parse_args()

    wc = build_wc2026_dataset(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wc.to_parquet(args.out, index=False)

    print(f"Wrote {len(wc)} WC 2026 matches to {args.out}")
    print(f"\nStage breakdown:\n{wc['stage'].value_counts().to_string()}")
    print(f"\nStatus breakdown:\n{wc['status'].value_counts().to_string()}")

    teams = qualified_teams(wc)
    print(f"\n{len(teams)} qualified teams, grouped:")
    for g in sorted(teams["group"].dropna().unique()):
        members = teams.loc[teams["group"] == g, "team"].tolist()
        print(f"  Group {g}: {', '.join(members)}")
