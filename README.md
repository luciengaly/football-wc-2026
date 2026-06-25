# World Cup 2026 — Match Prediction System

Système de prédiction des matchs de la Coupe du Monde 2026 (USA / Canada / Mexique, 11 juin – 19 juillet 2026).

## Objectif

Prédire **par match** :
1. P(victoire / nul / défaite)
2. La distribution complète des scores P(h, a) + marchés annexes (BTTS, O/U 2.5)

Le système se met à jour incrémentalement après chaque journée. Application
concrète : un optimiseur de pronostics pour "Mon Petit Prono" (voir [docs/MPP.md](docs/MPP.md)).

**Hors périmètre** : prédiction du vainqueur du tournoi, simulation Monte Carlo du bracket, probabilités de qualification par groupe.

## Stack

- **Python 3.11+**
- **Modèles** : familles Elo, Poisson / Dixon-Coles, gradient boosting, cotes de marché, valeur marchande, et leurs blends
- **UI** : Streamlit dashboard
- **Données** : 100% sources publiques gratuites

## Structure

```
src/wc2026/
├── ingestion/    # Téléchargement données (historique, Elo, cotes, valeurs)
├── features/     # Feature engineering, splits temporels
├── models/       # Modèles de prédiction (interface commune base.py)
├── evaluation/   # log-loss, Brier, RPS, calibration, backtests
└── mpp/          # Optimiseur de pronostics Mon Petit Prono
dashboard/        # Streamlit app
docs/             # Roadmap, ADRs, doc MPP
data/             # raw / processed / snapshots / mpp
```

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

# Pipeline complet (ingest + fetch cotes + build datasets + prédire les matchs restants)
python -m wc2026.pipeline refresh

# Étapes individuelles
python -m wc2026.pipeline ingest      # télécharge martj42
python -m wc2026.pipeline build       # construit elo.parquet + wc2026.parquet
python -m wc2026.pipeline predict     # prédit + snapshot (--as-of YYYY-MM-DD optionnel)

# Lancer le dashboard
streamlit run dashboard/app.py
# Puis ouvrir http://localhost:8501
```

## Documentation

- [docs/ROADMAP.md](docs/ROADMAP.md) — plan, sprints et avancement
- [docs/DECISIONS.md](docs/DECISIONS.md) — journal des décisions (ADR), avec les chiffres de chaque modèle
- [docs/MPP.md](docs/MPP.md) — optimiseur de pronostics Mon Petit Prono
- [DEPLOY.md](DEPLOY.md) — déploiement Streamlit Cloud

Le classement live des modèles est visible dans le dashboard (page **Performance**).
