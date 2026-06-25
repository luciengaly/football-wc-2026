# Roadmap — WC 2026 Prediction System

> Tournoi en cours : 11 juin 2026 → 19 juillet 2026
> Aujourd'hui : 19 juin 2026 (J0). Phase de groupes journée 2.
>
> **Objectif principal mis à jour** : maximiser la précision de prédiction de **score** et de **résultat** (log-loss minimal, exact-score accuracy maximale). Pas de gestion de bankroll — c'est un système gratuit, on empile les modèles tant qu'ils apportent du signal.

## Statut sprints

| Sprint | Période | Statut | Livrables |
|---|---|---|---|
| S1 — Setup + Ingestion | J0 | ✅ fait | Projet, martj42 + Elo + WC 2026 dataset |
| S2 — Modèles M1 + M2 | J0 | ✅ fait | M1 Elo, M2 Poisson, backtest WC 2022 |
| S3 — Bench complet + désignation | J0 | ✅ fait | M3 Dixon-Coles, M4 LightGBM, calibration, ensemble, **M2 désigné gagnant** |
| S4 — Prédiction de scores | J0 | ✅ fait (J0) | Distributions de scores M2/M3 exposées, métriques score-niveau, **M2 confirmé champion** sur 4/6 métriques score |
| S5 — Dashboard complet | J0 | ✅ fait | Score detail (heatmap Plotly), Predictions enrichie, Performance live avec match-par-match |
| S6 — Phase tournoi live | J0 → 19 juillet | 🟡 en cours | Backfill ✅, refresh quotidien ⚪, monitoring ✅, knockout ✅ |
| S7 — Push précision : quick wins | J0 | ✅ clos (négatif) | M5_nb dégénère, M6_stack & calibration n'aident pas. M3 promu champion W/D/L (ADR-018) |
| S8 — Bayesian hiérarchique | J0 | ⏸️ skip (négatif empirical Bayes) | Ridge CV montre M2 déjà optimal — PyMC ne va pas aider (ADR-019) |
| S9 — M8 score-direct LightGBM | J0 | ✅ clos (négatif) | M8 ne bat pas M2 sur exact-acc ni score log-loss (ADR-020) |
| **🛑 Plafond atteint sans données externes** | J0 | 📌 état | M2/M3 = SOTA-relatif avec données martj42 seules (ADR-021) |
| S10 — Cotes bookmakers | J+3 | ✅ clos (positif) | M9_odds bat M2/M3 (ADR-023). Blend M11=M3+M9 (ADR-024). |
| **S11 — Valeur marchande (Transfermarkt)** | J+4 | ✅ clos (positif) | M12 valeur joueurs bat l'Elo (Peeters confirmé, ADR-025). Signal orthogonal → **champion W/D/L = M13_blend3 (M3+M9+M12i)**, meilleur log-loss + ECE sur le régime live (ADR-026). **Bonus blessures** : M12i (squad disponible) bat M12 sur les 5 métriques, intégré au champion (ADR-027). |
| **S12 — Optimiseur MPP** | J+14 | ✅ clos (positif) | Module de pronostic "Mon Petit Prono" : maximise l'espérance de points (`argmax_s cote·P(R)+bonus·P(score)`). Backtest : **+32% de points vs prono réel** (ADR-028). Page dashboard "🃏 Conseil MPP", doc `docs/MPP.md`. |

**Hors périmètre** : prédiction du vainqueur du tournoi, simulation Monte Carlo du bracket, probabilités de qualification par groupe. Le focus est la prédiction *par match* (W/D/L et score).

**Mis hors priorité** : cotes des bookmakers (légal/ToS), données joueur (engineering trop lourd vs gain) — voir [DECISIONS.md] pour la justification.

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
| Différence M2 vs M3 vs M5_nb dans le bruit | Élevée | Faible | Backtest étendu, désigner champion sur métrique stable (RPS) |
| Stacking overfit la validation window | Moyenne | Moyenne | Window 365j, ridge sur meta-learner, monitoring live |
| Bayesian (PyMC) trop lent pour refit quotidien | Moyenne | Moyenne | Si > 5min/fit : fit hebdomadaire + warm-start posterior |

---

## S7 — Push précision : quick wins (J0) ✅ FERMÉ (résultats négatifs)

### Bilan
Tous les leviers de la vague 1 ont été essayés et **aucun n'a battu M2/M3** sur le bench étendu (290 matchs / 6 tournois). Les trois résultats négatifs sont documentés en détail :

- [x] **A — M5_nb Negative Binomial** : dégénère exactement en Poisson (α optimal ≈ 0). L'over-dispersion marginale (var/mean=1.80) est entièrement expliquée par l'hétérogénéité des forces d'équipes. Voir [ADR-015].
- [x] **C — Calibration isotonic** : sacrifie 1 an de training pour fitter les calibrateurs → log-loss 0.96 → 0.99 et accuracy 57% → 53%. Voir [ADR-017].
- [x] **D — M6_stack LogReg multinomial** : ne bat ni M2 ni M3 sur 290 matchs (gain WC 2022 isolé sur 64 matchs = bruit). M1-M4 trop corrélés pour exploiter du stacking. Voir [ADR-016].

### Bénéfice indirect
Le refactor 2-étapes de M3 (initialement pour éliminer les warnings L-BFGS, [ADR-013]) combiné au bench étendu a révélé que **M3 dépasse M2 sur l'accuracy W/D/L** (57.24% vs 56.55%) et l'**ECE** (0.026 vs 0.032). M3 est promu champion W/D/L. M2 reste champion score-distribution (exact-acc 13.79% vs 12.41%). Voir [ADR-018].

### Apprentissage pour S8
Pour qu'un nouveau modèle apporte du signal il faut une **vraie décorrélation** des erreurs. M7 bayésien hiérarchique est de bonne foi un modèle différent (priors structurels, shrinkage), pas une variation de M2. C'est notre meilleure chance d'améliorer sur 290 matchs.

---

## S8 — Push précision : Bayesian hiérarchique (J+2 → J+4) ⚪ À FAIRE

### Objectif
M7 = Dixon-Coles hiérarchique bayésien (PyMC, NUTS sampling). Les forces d'équipes sont des draws d'un prior global (shrinkage), ce qui aide particulièrement les ~6 équipes peu données (Curaçao, Cap-Vert, Jordanie…).

### Tâches
- [ ] Spec model PyMC : `att[t] ~ Normal(μ_att, σ_att)`, `def[t] ~ Normal(μ_def, σ_def)`, hyperpriors sur σ, hyperpriors sur μ
- [ ] Likelihood Poisson bivarié + correction τ (Dixon-Coles) en PyMC
- [ ] Time-decay weights via observed `weight`
- [ ] Diagnostics : R-hat, ESS, posterior predictive checks
- [ ] Backtest sur 6 tournois (sera ~5x plus lent que M2)
- [ ] Comparaison vs M6_stack

### Critères de sortie
- M7 converge proprement (R-hat < 1.01 sur tous les params globaux)
- M7 expose `predict_proba` et `predict_score_dist` qui prennent le posterior mean
- Si M7 bat M6_stack : intégrer dans M6_stack (nouvelle vague)
- Si M7 n'apporte rien : documenter pourquoi et passer à S9

---

## S9 — Spécialisation score exact (optionnel, J+5 → J+7)

### Objectif
**M8 = LightGBM 49-classes** (h, a) ∈ {0..6} × {0..6}, entraîné avec class weights inversement proportionnels à la fréquence empirique. Vise spécifiquement l'exact-score accuracy (où les modèles génératifs plafonnent ~14%).

Optionnel : à activer seulement si la métrique exact-score reste > 4% en-dessous de l'objectif après S7+S8.
