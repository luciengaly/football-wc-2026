# Operations

Quick reference for running the system day-to-day during the tournament.

## Daily workflow

```cmd
REM Activate venv
.venv\Scripts\activate

REM Pull new results, recompute features, predict the rest, snapshot
python -m wc2026.pipeline refresh
```

This single command runs three steps:
1. `ingest`  — downloads the latest `results.csv` from martj42
2. `build`   — rebuilds `data/processed/elo.parquet` and `data/processed/wc2026.parquet`
3. `predict` — fits the 4 models on data strictly before today and snapshots
   predictions to `data/snapshots/YYYY-MM-DD.parquet`

Open the dashboard:

```cmd
streamlit run dashboard/app.py
```

## Automated daily refresh (Windows Task Scheduler)

The repo ships with `scripts/daily_refresh.bat`. To schedule it:

1. Open **Task Scheduler** (`taskschd.msc`).
2. **Action ▸ Create Basic Task…**
3. Name: `WC 2026 Refresh`. Description: anything.
4. **Trigger**: *Daily* — pick a time that's after FIFA usually publishes
   results (e.g. 09:00 local time).
5. **Action**: *Start a program*.
   - Program/script: `<repo>\scripts\daily_refresh.bat`
   - Start in (optional): `<repo>`
6. Finish.

The script writes to `data/snapshots/refresh.log` on every run — tail it to
check status:

```cmd
type data\snapshots\refresh.log
```

## Backfill / reconstruct past snapshots

If you need to re-predict matches that already happened (e.g. you started
late, or want to compare model versions on the same matches):

```cmd
python -m wc2026.pipeline backfill --start 2026-06-11 --end 2026-06-20
```

This produces one snapshot per day, fitting the model with **only** data
strictly earlier than that day (no leakage).

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `httpx.ConnectError` during `ingest` | Network / GitHub down | Re-run later |
| `M3 Dixon-Coles L-BFGS warning: ABNORMAL` | Known convergence issue on small/old data | Cosmetic — predictions still valid |
| Dashboard: "No snapshots yet" | Never ran predict | `python -m wc2026.pipeline backfill --start 2026-06-11 --end <today>` |
| `ModuleNotFoundError` after pulling new code | New dep added | `pip install -e ".[dev]"` |

## Tests

```cmd
python -m pytest tests/ -v
```
