# Roadmap — WC 2026 Prediction System

> Tournoi en cours : 11 juin 2026 → 19 juillet 2026
> Aujourd'hui : 19 juin 2026 (J0). Phase de groupes journée 2.
> Sortie cible MVP fonctionnel : J+9 (28 juin), avant les 16es.

## Statut sprints

| Sprint | Période | Statut | Livrables |
|---|---|---|---|
| S1 — Setup + Ingestion | J0 | ✅ fait | Projet, martj42 + Elo + WC 2026 dataset |
| S2 — Modèles M1 + M2 | J0 | ✅ fait | M1 Elo, M2 Poisson, backtest WC 2022 |
| S3 — Bench complet + désignation | J0 | ✅ fait | M3 Dixon-Coles, M4 LightGBM, calibration, ensemble, **M2 désigné gagnant** |
| S4 — Prédiction de scores | J0 | ✅ fait (J0) | Distributions de scores M2/M3 exposées, métriques score-niveau, **M2 confirmé champion** sur 4/6 métriques score |
| S5 — Dashboard complet | J0 | ✅ fait | Score detail (heatmap Plotly), Predictions enrichie, Performance live avec match-par-match |
| S6 — Phase tournoi live | J0 → 19 juillet | 🟡 en cours | Backfill snapshots passés ✅, refresh quotidien ⚪, monitoring continu ⚪ |

**Hors périmètre** : prédiction du vainqueur du tournoi, simulation Monte Carlo du bracket, probabilités de qualification par groupe. Le focus est la prédiction *par match* (W/D/L et score).

Légende : ⚪ pas commencé · 🟡 en cours · ✅ fait · 🔴 bloqué

---

## S1 — Setup + Ingestion (J0) ✅ FAIT

### Objectifs
Avoir un dataset reproductible en local avec tous les matchs internationaux + Elo + WC 2026.

### Tâches
- [x] Squelette de projet (`pyproject.toml`, structure src/, docs/)
- [x] `ROADMAP.md` + `DECISIONS.md`
- [x] Module `ingestion/teams.py` — normalisation noms équipes + confédérations
- [x] Module `ingestion/historical.py` — dataset martj42 → `data/raw/results.csv`
- [x] Module `ingestion/elo.py` — **Elo calculé en interne** (formule World Football Elo) → `data/processed/elo.parquet`
- [x] Module `ingestion/wikipedia.py` — fallback (non utilisé car martj42 suffit)
- [x] Module `wc2026.py` — extraction WC 2026 + détection groupes par connectivité

### Critères de sortie (atteints)
- ✅ Pipeline `python -m wc2026.pipeline ingest` opérationnel
- ✅ 49 477 matchs historiques chargés (1872 → 2026-06-27)
- ✅ 72 matchs WC 2026 identifiés (28 joués, 44 à jouer)
- ✅ 48 équipes qualifiées correctement réparties dans 12 groupes A-L

### Top 10 Elo actuel (sanity check)
1. Argentine 2200.6 · 2. Espagne 2180.4 · 3. France 2139.8 · 4. Angleterre 2113.8
5. Colombie 2075.8 · 6. Brésil 2057.0 · 7. Allemagne 2015.9 · 8. Portugal 2008.6
9. Pays-Bas 2007.6 · 10. Japon 1990.7

---

## S2 — Modèles M1 + M2 + backtest 🟡 EN COURS

### Tâches
- [x] `models/base.py` — interface `Model.fit / predict_proba`
- [x] `models/elo_baseline.py` — **M1 Elo** (Davidson-style W/D/L) + fit du `draw_param`
- [x] `models/poisson.py` — **M2 Poisson indépendant** (MLE custom, time-decay 2 ans, ridge L2)
- [x] `evaluation/baselines.py` — Uniform + Marginal (sanity baselines)
- [x] `evaluation/metrics.py` — log-loss, Brier multi-classes, RPS, ECE, accuracy
- [x] `evaluation/backtest.py` — walk-forward harness (re-fit par "round")
- [x] Backtest M1 sur WC 2022 (cf. tableau ci-dessous)
- [ ] Backtest M2 sur WC 2022 (en cours)
- [ ] `features/build.py` — features pour M4 LightGBM (déferré à S3)

### Résultats backtest sur WC 2022 (64 matchs, re-fit par jour)

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| uniform | 1.0986 | 0.6667 | 0.2387 | 43.75% | 0.0000 |
| marginal | 1.0743 | 0.6512 | 0.2358 | 43.75% | 0.0359 |
| M1_elo | 1.0813 | 0.6263 | 0.2279 | 51.56% | 0.0661 |
| **M2_poisson** | **1.0426** | **0.6061** | **0.2152** | **53.12%** | **0.0602** |

**Lecture** :
- M1 bat les baselines sur Brier, RPS, accuracy ; perd sur log-loss à cause des upsets WC 2022 où Elo est trop confiant.
- **M2 bat M1 sur les 5 métriques** y compris log-loss. Le critère de sortie S2 est validé.
- ECE ~0.06 reste à améliorer en S3 par isotonic regression / Platt scaling.

### Critères de sortie restants
- [x] M2 bat M1 sur Brier *ou* RPS ✅ (sur les deux + 3 autres)
- [ ] Calibration ECE < 0.05 (cible S3)
- [ ] Backtest étendu : Euro 2024 + Copa America 2024 pour stabilité (S3)

---

## S3 — Modèles avancés + Ensemble (J+6 → J+8)

### Tâches
- [ ] `models/dixon_coles.py` — **M3 Dixon-Coles** (Poisson bivarié + correction faible score + décroissance temporelle)
- [ ] `models/lightgbm.py` — **M4 LightGBM** classifier 3-classes
- [ ] `models/ensemble.py` — **M5 Ensemble** (moyenne pondérée, poids optimisés en CV temporelle)
- [ ] `evaluation/calibration.py` — reliability diagrams + isotonic regression
- [ ] Benchmark final sur backtest croisé → désigner **le modèle vainqueur** (livrable obligatoire)

### Critères de sortie
- Tableau comparatif `model × {log-loss, Brier, RPS, calibration ECE}`.
- Décision documentée dans `DECISIONS.md` : *modèle X retenu pour les prédictions live*.

---

## S4 — Prédiction de scores (J0) ✅ FAIT

### Tâches
- [x] `predict_score_dist(matches)` exposé proprement sur M2 et M3
- [x] Score le plus probable (mode) + E[h], E[a] dans le snapshot du pipeline
- [x] Marchés annexes : BTTS, O/U 2.5 calculés et stockés
- [x] Métriques score-niveau (`evaluation/score_metrics.py`) :
  - exact_score_accuracy, top_k_accuracy, score_log_loss
  - btts_brier, over_under_brier, goal_diff_rps
- [x] 5 tests pytest verrouillent les invariants (somme=1, cohérence W/D/L↔joint, anti-régression triu/tril)
- [x] Backtest M2 vs M3 sur 6 tournois (290 matchs) → résultats ci-dessous

### Résultats backtest score (290 matchs)

| Modèle | exact-acc | top3-acc | score log-loss | BTTS Brier | O/U 2.5 Brier | goal-diff RPS |
|---|---|---|---|---|---|---|
| **M2 Poisson** | **13.79%** | 37.24% | **2.7707** | 0.2565 | **0.2571** | **0.1519** |
| M3 Dixon-Coles | 12.41% | **37.93%** | 2.7902 | **0.2547** | 0.2580 | 0.1542 |

**Décision** : M2 reste le champion (4/6 métriques) — voir [ADR-010](DECISIONS.md). M3 ne gagne que sur top-3 et BTTS, là où la correction τ aide les scores faibles.

---

## S5 — Dashboard complet (J0) ✅ FAIT

### Tâches
- [x] Page Predictions enrichie avec score le plus probable + E[buts] + BTTS + O/U
- [x] Page **Score detail** avec heatmap Plotly P(h, a) cliquable + agrégats
- [x] Page **Performance** étendue : metrics live W/D/L + score-level + match-par-match
- [ ] Page Snapshots history : navigation jour par jour (deferred, low value)
- [ ] Bouton "Rafraîchir données" dans l'UI (deferred ; on utilise le CLI)

---

## S6 — Phase tournoi live (J0 → 19 juillet) 🟡 EN COURS

### Tâches
- [x] CLI `backfill --start --end` pour reconstruire les snapshots passés
- [x] 9 snapshots datés (11–19 juin) générés pour les 28 matchs joués + 44 à venir
- [x] Page Performance affiche les métriques W/D/L et score-level live
- [x] **Cumulative metrics chart** : trajectoire log-loss / RPS / accuracy / ECE au fil des journées
- [x] **Script `scripts/daily_refresh.bat`** + doc Windows Task Scheduler dans `docs/OPERATIONS.md`
- [x] **Knockout phase support** : `advance_probabilities` (split du nul, 50/50 par défaut) + tests, colonnes `p_home_advances` / `p_away_advances` dans le snapshot
- [ ] Configurer le Task Scheduler (utilisateur, manuel)
- [x] Review J+0 (28 matchs) — M2 reste champion, M3 marginalement meilleur ECE ([ADR-014](DECISIONS.md))
- [x] M3 refactor en fit 2-étapes pour éliminer les warnings L-BFGS ([ADR-013](DECISIONS.md))
- [ ] Re-review fin de phase de groupes (~J+8, 72 matchs joués) : bascule de champion si M3 confirme son avantage
- [ ] Fin de tournoi (J+30) : audit complet calibration + exact-score accuracy + ratio knockout

### Critères de sortie
- À chaque journée : `python -m wc2026.pipeline refresh` met à jour les snapshots et les métriques live.
- Le Task Scheduler le lance automatiquement chaque matin.
- Pour la phase à élimination directe : les snapshots contiennent les probas d'accession (`p_home_advances`, `p_away_advances`) ; le dashboard les affiche dès le 1er match de R32.

## Risques actifs

| Risque | Probabilité | Sévérité | Mitigation |
|---|---|---|---|
| Calendrier serré (1 mois) | Certain | Moyenne | MVP modèle simple privilégié, raffinement après |
| martj42 mise à jour tardive d'un match | Moyenne | Faible | Vérification quotidienne, fallback Wikipedia possible |
| Distribution de scores difficile à valider sur 1 tournoi | Élevée | Moyenne | Backtest pluri-tournois (290 matchs déjà disponibles) |
| Différence score/exacte M2 vs M3 dans le bruit | Élevée | Faible | Comparer multi-tournois, choisir M3 si correction τ aide réellement les low-scores |
