# World Cup 2026 — Match Prediction System

Système de prédiction des matchs de la Coupe du Monde 2026 (USA / Canada / Mexique, 11 juin – 19 juillet 2026).

## Objectif

Prédire **par match** :
1. P(victoire / nul / défaite) ✅ (modèle champion : M2 Poisson)
2. La distribution complète des scores P(h, a) + marchés annexes (BTTS, O/U 2.5) 🟡 en cours

Le système se met à jour incrémentalement après chaque journée.

**Hors périmètre** : prédiction du vainqueur du tournoi, simulation Monte Carlo du bracket, probabilités de qualification par groupe.

## Stack

- **Python 3.11+**
- **Modèles** : Elo baseline, Poisson, Dixon-Coles, LightGBM, Ensemble
- **UI** : Streamlit dashboard
- **Données** : 100% sources publiques gratuites

## Structure

```
src/wc2026/
├── ingestion/    # Téléchargement données (historique, Elo, WC live)
├── features/     # Feature engineering, splits temporels
├── models/       # M1..M5
├── evaluation/   # log-loss, Brier, RPS, calibration
└── simulation/   # Monte Carlo tournoi
dashboard/        # Streamlit app
docs/             # Roadmap, ADRs
data/             # raw / processed / snapshots
```

## Documentation

- [docs/ROADMAP.md](docs/ROADMAP.md) — plan détaillé et statut des sprints
- [docs/DECISIONS.md](docs/DECISIONS.md) — décisions techniques et leur justification

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
```

## Usage

```bash
# Activer l'environnement
.venv\Scripts\activate          # Windows

# Pipeline complet (ingest + build datasets + prédire les matchs restants)
python -m wc2026.pipeline refresh --as-of 2026-06-19

# Étapes individuelles
python -m wc2026.pipeline ingest                       # télécharge martj42
python -m wc2026.pipeline build                        # construit elo.parquet + wc2026.parquet
python -m wc2026.pipeline predict --as-of 2026-06-19   # prédit + snapshot

# Lancer le dashboard
streamlit run dashboard/app.py
# Puis ouvrir http://localhost:8501
```

## État actuel

- ✅ S1 Setup + Ingestion (49 477 matchs historiques, 72 matchs WC 2026, 48 équipes en 12 groupes)
- ✅ S2 + S3 — 4 modèles benchmarkés sur 290 matchs (6 tournois) : M1 Elo, M2 Poisson, M3 Dixon-Coles, M4 LightGBM. **M2 désigné champion W/D/L.**
- 🟡 S4 — Prédiction de scores : modèles déjà capables (M2, M3 exposent `predict_score_dist`), métriques score-niveau implémentées, backtest en cours.
- 🟡 S5 — Dashboard 5 pages (Tournament, Predictions, Score detail, Performance, About).

Voir [docs/ROADMAP.md](docs/ROADMAP.md) pour le plan détaillé.
