"""Transfermarkt market-value ingestion → dated national-team strength.

Source: salimt/football-datasets (GitHub), tables:
  - player_market_value.csv          : (player_id, date, value) dated time series
  - player_profiles.csv              : player_id → main_position, citizenship, dob
  - player_national_performances.csv : player_id → national team_id (actual caps)

Why this design (leakage-free):
  * For a match on date D, a team's "strength" is the aggregate market value of
    its most valuable capped players, valued **as of D** (latest valuation
    ≤ D). Market value is a dated quantity, so this respects the timeline.
  * We resolve each national team_id → country via the *modal citizenship* of
    its capped players (data-driven, robust to the diaspora double-count that
    plagues a naive citizenship filter — e.g. France-born players holding an
    Ivorian passport).
  * Position split: Attack + Midfield → offensive value; Defender + Goalkeeper
    → defensive value. This feeds directly into a Poisson att/def structure.

Reference: Peeters (2018), "Testing the Wisdom of Crowds in the field:
Transfermarkt valuations and international soccer results", Int. J.
Forecasting — market values beat FIFA ranking and Elo for international games.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from wc2026.ingestion.teams import normalize

REPO_BASE = "https://github.com/salimt/football-datasets/raw/main/datalake/transfermarkt"
FILES = {
    "market_value": "player_market_value/player_market_value.csv",
    "profiles": "player_profiles/player_profiles.csv",
    "national": "player_national_performances/player_national_performances.csv",
    "injuries": "player_injuries/player_injuries.csv",
}

SQUAD_SIZE = 26
OFFENSIVE_POSITIONS = {"Attack", "Midfield"}
DEFENSIVE_POSITIONS = {"Defender", "Goalkeeper"}

# Canonical team name → Transfermarkt citizenship token used to resolve the
# national team_id. Only teams whose TM spelling differs need an entry; the
# rest match by exact normalize() name.
TM_CITIZENSHIP = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Curaçao": "Curacao",
    "Ivory Coast": "Cote d'Ivoire",
    "South Korea": "Korea, South",
    "North Korea": "Korea, North",
    "Turkey": "Türkiye",
    "United States": "United States",
    "Czech Republic": "Czech Republic",
    "Cape Verde": "Cape Verde",
    "DR Congo": "DR Congo",
}


_SHARED_STORE: "MarketValueStore | None" = None


def get_shared_store(raw_dir: Path = Path("data/raw")) -> "MarketValueStore":
    """Process-wide singleton: load the TM tables + panel once, reuse everywhere.

    Avoids re-reading the 900k-row market value CSV for every model instance
    in the pipeline (M12 directly, and M13 which wraps an M12).
    """
    global _SHARED_STORE
    if _SHARED_STORE is None:
        store = MarketValueStore(raw_dir)
        # Deployment-friendly: if the heavy TM CSVs aren't present but the
        # cached panels are, run panel-only (enough for fit + predict).
        if (raw_dir / "tm_player_market_value.csv").exists():
            store.load()
            store.build_panel(available_only=False)
            store.build_panel(available_only=True)
        else:
            store.load_panels_only()
        _SHARED_STORE = store
    return _SHARED_STORE


def download_all(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for key, rel in FILES.items():
            target = raw_dir / f"tm_{Path(rel).name}"
            r = client.get(f"{REPO_BASE}/{rel}")
            r.raise_for_status()
            target.write_bytes(r.content)
            paths[key] = target
    return paths


class MarketValueStore:
    """Loads the TM tables and answers `team_value_asof` queries."""

    def __init__(self, raw_dir: Path = Path("data/raw")):
        self.raw_dir = raw_dir
        self._mv: pd.DataFrame | None = None
        self._pos: dict[int, str] = {}
        self._team_pool: dict[str, set[int]] = {}  # country token -> capped player ids
        self._injured: set[tuple[int, pd.Timestamp]] = set()  # (player_id, month_start)

    # ------------------------------------------------------------------
    def load(self) -> "MarketValueStore":
        mv = pd.read_csv(self.raw_dir / "tm_player_market_value.csv")
        mv["date"] = pd.to_datetime(mv["date_unix"], errors="coerce")
        mv = mv.dropna(subset=["date", "value"]).sort_values("date")
        self._mv = mv[["player_id", "date", "value"]]

        prof = pd.read_csv(
            self.raw_dir / "tm_player_profiles.csv",
            usecols=["player_id", "main_position", "citizenship"],
        )
        prof["citizenship"] = prof["citizenship"].fillna("")
        self._pos = dict(zip(prof["player_id"], prof["main_position"]))

        nat = pd.read_csv(
            self.raw_dir / "tm_player_national_performances.csv",
            usecols=["player_id", "team_id", "matches"],
        )
        # Resolve each national team_id → modal citizenship token of its capped
        # players (≥1 cap), then build country → capped player pool.
        prof_cit = dict(zip(prof["player_id"], prof["citizenship"]))
        nat = nat[nat["matches"] >= 1].copy()
        nat["primary_cit"] = nat["player_id"].map(
            lambda pid: prof_cit.get(pid, "").split("  ")[0].strip()
        )
        # team_id → modal citizenship
        team_country = (
            nat.groupby("team_id")["primary_cit"]
            .agg(lambda s: s.value_counts().idxmax() if len(s) else "")
            .to_dict()
        )
        # country token → set of capped player_ids
        pool: dict[str, set[int]] = {}
        for tid, grp in nat.groupby("team_id"):
            country = team_country.get(tid, "")
            if not country:
                continue
            pool.setdefault(country, set()).update(grp["player_id"].tolist())
        self._team_pool = pool

        # Injury intervals → set of (player_id, month_start) the player misses
        inj_path = self.raw_dir / "tm_player_injuries.csv"
        if inj_path.exists():
            inj = pd.read_csv(inj_path, usecols=["player_id", "from_date", "end_date"])
            inj["from_date"] = pd.to_datetime(inj["from_date"], errors="coerce")
            inj["end_date"] = pd.to_datetime(inj["end_date"], errors="coerce")
            inj = inj.dropna(subset=["from_date", "end_date"])
            injured: set[tuple[int, pd.Timestamp]] = set()
            for r in inj.itertuples(index=False):
                # Mark every month-start covered by [from_date, end_date]
                start = pd.Timestamp(year=r.from_date.year, month=r.from_date.month, day=1)
                for month in pd.date_range(start, r.end_date, freq="MS"):
                    injured.add((r.player_id, month))
            self._injured = injured
        return self

    # ------------------------------------------------------------------
    def load_panels_only(self) -> "MarketValueStore":
        """Load ONLY the cached monthly panels, skipping the heavy TM CSVs.

        Used for deployment (Streamlit Cloud): the ~50 MB raw Transfermarkt
        CSVs aren't committed, but the small precomputed panels are. This
        supports `team_value_asof` (hence M12/M13 fit + predict) fully — only
        `build_panel` (which needs the CSVs) is unavailable.
        """
        for available_only in (False, True):
            path = self._panel_path(available_only)
            if not path.exists():
                raise FileNotFoundError(
                    f"Panel {path} missing — run market_value once locally to build it."
                )
            df = pd.read_parquet(path)
            panel = {
                (r.country, pd.Timestamp(r.month)): (r.total, r.off, r.deff)
                for r in df.itertuples(index=False)
            }
            setattr(self, "_panel_avail" if available_only else "_panel", panel)
        return self

    # ------------------------------------------------------------------
    def _country_token(self, team: str) -> str:
        canon = normalize(team)
        return TM_CITIZENSHIP.get(canon, canon)

    # ------------------------------------------------------------------
    # Fast path: a precomputed monthly panel (team, month) -> (total, off, def)
    # ------------------------------------------------------------------
    _panel: dict[tuple[str, pd.Timestamp], tuple[float, float, float]] | None = None
    _panel_avail: dict[tuple[str, pd.Timestamp], tuple[float, float, float]] | None = None

    def _panel_path(self, available_only: bool) -> Path:
        name = "tm_value_panel_avail.parquet" if available_only else "tm_value_panel.parquet"
        return self.raw_dir.parent / "processed" / name

    def build_panel(self, start: str = "2008-01-01", end: str = "2026-12-01",
                    available_only: bool = False, cache_path: Path | None = None) -> "MarketValueStore":
        """Precompute (country, month) -> (total, off, def) for all pools.

        If available_only, players injured at month-start (per the injury
        intervals) are excluded before taking the top-SQUAD_SIZE — i.e. the
        value of the *available* squad. Persisted to its own parquet.
        """
        cache_path = cache_path or self._panel_path(available_only)
        target_attr = "_panel_avail" if available_only else "_panel"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            panel = {
                (r.country, pd.Timestamp(r.month)): (r.total, r.off, r.deff)
                for r in df.itertuples(index=False)
            }
            setattr(self, target_attr, panel)
            return self

        months = pd.date_range(start, end, freq="MS")
        pos_off = {p for p, v in self._pos.items() if v in OFFENSIVE_POSITIONS}
        rows = []
        for country, pids in self._team_pool.items():
            pool_mv = self._mv[self._mv["player_id"].isin(pids)]
            if pool_mv.empty:
                continue
            for month in months:
                sub = pool_mv[pool_mv["date"] <= month]
                if available_only and self._injured:
                    sub = sub[~sub["player_id"].map(lambda pid, m=month: (pid, m) in self._injured)]
                if sub.empty:
                    continue
                latest = sub.groupby("player_id").tail(1)
                top = latest.nlargest(SQUAD_SIZE, "value")
                total = float(top["value"].sum())
                off = float(top[top["player_id"].isin(pos_off)]["value"].sum())
                rows.append((country, month, total, off, total - off))
        df = pd.DataFrame(rows, columns=["country", "month", "total", "off", "deff"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        panel = {
            (r.country, pd.Timestamp(r.month)): (r.total, r.off, r.deff)
            for r in df.itertuples(index=False)
        }
        setattr(self, target_attr, panel)
        return self

    @lru_cache(maxsize=16384)
    def team_value_asof(self, team: str, date: pd.Timestamp,
                        available_only: bool = False) -> tuple[float, float, float]:
        """Return (total, offensive, defensive) squad value in EUR as of `date`.

        Top-SQUAD_SIZE most valuable capped players, split by position. With
        available_only, injured players (per dated injury intervals) are
        excluded. Uses the precomputed monthly panel. (nan,...) if unresolved.
        """
        token = self._country_token(team)
        panel = self._panel_avail if available_only else self._panel
        if panel is None:
            raise RuntimeError(
                f"Panel (available_only={available_only}) not built. "
                "Call build_panel(available_only=...) first."
            )
        month = pd.Timestamp(year=date.year, month=date.month, day=1)
        val = panel.get((token, month))
        return val if val is not None else (np.nan, np.nan, np.nan)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download or not (args.raw / "tm_player_market_value.csv").exists():
        print("Downloading Transfermarkt tables...")
        for k, p in download_all(args.raw).items():
            print(f"  {k}: {p.stat().st_size:,} bytes")

    store = MarketValueStore(args.raw).load()
    print(f"Resolved {len(store._team_pool)} national-team pools, "
          f"{len(store._injured):,} (player, month) injury cells")
    store.build_panel(available_only=False)
    store.build_panel(available_only=True)

    d = pd.Timestamp("2022-11-01")
    demo = ["France", "England", "Brazil", "Spain", "Argentina", "Germany",
            "Ivory Coast", "Morocco", "Senegal", "Japan", "Saudi Arabia", "Haiti"]
    print(f"\nTop-{SQUAD_SIZE} squad value as-of {d.date()} (M EUR): full vs available")
    for t in demo:
        full = store.team_value_asof(t, d, available_only=False)
        avail = store.team_value_asof(t, d, available_only=True)
        if full[0] != full[0]:
            print(f"  {t:<15s}  UNRESOLVED")
        else:
            print(f"  {t:<15s}  full={full[0]/1e6:>7.0f}   avail={avail[0]/1e6:>7.0f}   "
                  f"(injured drop {100*(1-avail[0]/full[0]):>4.1f}%)")
