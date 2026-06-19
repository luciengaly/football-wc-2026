# Decision log — ADR style

Toutes les décisions structurantes du projet sont consignées ici, du plus récent au plus ancien.
Format : numéro, titre, contexte, décision, alternatives, conséquences.

---

## ADR-014 — Mid-tournament review J+0 sur 28 matchs

**Date** : 2026-06-19
**Statut** : ✅ snapshot pris

**Contexte** : Engagement pris dans [ADR-011] de ré-évaluer le classement des modèles à mi-phase de groupes. À J+0 (19 juin 2026), 28 matchs joués (MD1 complète + début MD2). Échantillon encore petit mais permet une lecture précoce.

**Performance live (28 matchs, prédictions same-day, M3 2-step après refactor)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M3_dixon_coles | 0.9493 | 0.5830 | 0.1674 | 57.14% | 0.0503 |
| M2_poisson | 0.9579 | 0.5859 | 0.1679 | 57.14% | 0.0730 |
| M1_elo | 1.0892 | 0.6614 | 0.1969 | 53.57% | 0.1114 |
| M4_lightgbm | 1.2166 | 0.7286 | 0.2264 | 46.43% | 0.1315 |

**Observations** :
- **M2 et M3 toujours dans le mouchoir de poche** (Δlog-loss = 0.0086). M3 marginalement devant mais 28 matchs = écart-type estimé ~0.03 sur log-loss → différence non significative.
- **M2 reste le champion officiel** (basé sur 290 matchs, 6 tournois — voir [ADR-008] et [ADR-010]). Pas de bascule au champion sans signal robuste.
- **M4 LightGBM confirme** sa sous-performance : 46% accuracy, log-loss 1.22, ECE 0.131. À retoucher (tuning ou features supplémentaires) ou retirer en v2.
- **M3 ECE = 0.0503** : meilleure calibration que M2 (0.0730) sur ce sous-échantillon. Cohérent avec la correction τ qui lisse les faibles scores.
- **Score-level (champion = M2)** : 3.6% exact-acc (1/28, Mexique-Corée 1-0), MAE_H = 1.21, MAE_A = 0.71. Bruit normal sur 28 matchs (cible historique ~13.8%).

**Décision** : Pas de changement de champion. Re-review prévue à **fin de phase de groupes (~J+8, ~72 matchs joués)** ; si l'écart M2-M3 se stabilise ou s'amplifie en faveur de M3, on basculera.

**Conséquences** :
- Continue avec M2 comme prédicteur principal.
- M4 LightGBM laissé en bench passif. Pas de tuning prioritaire vu le retour sur investissement faible.
- L'ECE de M3 (0.0503) déclenchera une re-évaluation si elle persiste — la calibration est un critère majeur pour la fiabilité long terme.

---

## ADR-013 — M3 Dixon-Coles : fit 2-étapes (Poisson puis ρ seul)

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : L'implémentation initiale de M3 faisait un MLE joint sur 403 paramètres (μ, γ, ρ, 200 att, 200 def) avec L-BFGS-B + gradient analytique. Symptôme : des warnings "ABNORMAL TERMINATION IN LNSRCH" sur certains training subsets (typiquement les tournois plus anciens avec moins de données récentes). La cause racine est probablement l'interaction non-lisse entre la borne sur ρ (∈ [-0.49, 0.49]) et le clip τ ≥ 1e-9 qui crée des kinks dans le gradient.

**Décision** : Refactor en **fit 2-étapes** :
1. Fit Poisson indépendant complet (μ, γ, att, def) — convexe, converge proprement.
2. Fit ρ seul via `scipy.optimize.minimize_scalar(method='bounded')` avec bornes adaptées par match (basé sur les λ_H × λ_A observés).

**Alternatives évaluées** :
- Joint MLE plus robuste (transformation `tanh` de ρ) : ajoute de la complexité, marginal sur la performance.
- IRLS sur le Poisson puis grid search sur ρ : équivalent en performance, plus lourd.

**Comparaison empirique** (live perf 28 matchs WC 2026) :
- Joint (avant) : log-loss 0.9488, ECE 0.0590
- 2-step (après) : log-loss 0.9493, ECE 0.0503 (légère amélioration ECE)

Différence dans le bruit, mais **2-step élimine les warnings**, est **2x plus rapide** (test suite 5.9s → 2.4s), et correspond à la pratique standard de la littérature (Dixon-Coles 1997 utilise une approche jointe mais beaucoup d'implémentations modernes — `regista` en R, `penaltyblog` en Python — utilisent un fit 2-étapes).

**Conséquences** :
- M3 est désormais robuste sur tout training subset.
- Le fit 2-étapes est légèrement sous-optimal en théorie (la prise en compte de τ pourrait re-biaiser μ/γ/att/def), mais la différence est < 0.001 sur log-loss.
- Tests pytest restent verts (9/9).

---

## ADR-012 — Phase à élimination directe : split du nul 50/50 par défaut

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : À partir du R32 (28 juin), les matchs ne peuvent pas se terminer par un nul — prolongations puis tirs au but tranchent. Nos modèles (M1-M4) prédisent en temps réglementaire et exposent toujours P(W/D/L) avec une masse non-nulle sur D. Sans correction, le dashboard afficherait pour un R32 "Brésil 60%, Nul 25%, Cameroun 15%" — pénalisant et mal interprétable.

**Décision** : Implémenter une conversion `(p_h, p_d, p_a) → (p_h_advances, p_a_advances)` via `wc2026.models.knockout.advance_probabilities`. Le mass de P(draw) est réparti par défaut **50/50** entre les deux équipes, paramétrable via `draw_split ∈ [0, 1]`.

**Alternatives évaluées** :
- *Split proportionnel* : allouer le nul au prorata de `p_home / (p_home + p_away)`. Avantage : favorise légèrement le favori. Inconvénient : la littérature empirique sur les tirs au but internationaux donne ~52/48 pour le favori — quasi coin-flip, donc le 50/50 simple est proche de l'optimum.
- *Split Elo-pondéré* : `weight = sigmoid((R_H - R_A) / 100)`. Plus principal mais marginal en gain.

**Décision** : 50/50 par défaut, hyper-paramètre exposé pour itération ultérieure. Plus simple à expliquer, défensible empiriquement, et facile à tester (`tests/test_knockout.py` : 4 tests passent).

**Conséquences** :
- Le snapshot inclut désormais `p_home_advances`, `p_away_advances` (NaN pour les matchs de groupe).
- Le dashboard pourra afficher la proba d'accession dès le premier R32.
- À l'issue du tournoi, on pourra évaluer la calibration du 50/50 sur les ~10-15 matchs allés en ET/pens et ajuster `draw_split` si nécessaire (probable v2).

---

## ADR-011 — Backfill des snapshots pour la phase WC 2026 déjà jouée + live monitoring

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Pour évaluer la performance des modèles sur la WC 2026 *au fur et à mesure*, il faut disposer de prédictions datées au plus tard la veille de chaque match (snapshot `as_of = match_date`). On n'a pas eu cette infrastructure pendant les MD1+MD2 déjà joués (28 matchs sur 72 du group stage).

**Décision** :
- Ajouter une CLI `python -m wc2026.pipeline backfill --start YYYY-MM-DD --end YYYY-MM-DD` qui reconstruit les snapshots pour chaque jour passé en n'utilisant que les données strictement antérieures (pas de leakage).
- Le snapshot stocke W/D/L + score le plus probable + E[buts] + BTTS + O/U 2.5 pour M2 et M3.
- La page Performance du dashboard joint chaque match joué à la prédiction du jour de match (`as_of == date`), pas n'importe quel snapshot.

**Première lecture sur 28 matchs WC 2026** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M3_dixon_coles | 0.9488 | 0.5827 | 0.1673 | 57.14% | 0.0590 |
| M2_poisson | 0.9579 | 0.5859 | 0.1679 | 57.14% | 0.0730 |
| M1_elo | 1.0892 | 0.6614 | 0.1969 | 53.57% | 0.1114 |
| M4_lightgbm | 1.2166 | 0.7286 | 0.2264 | 46.43% | 0.1315 |

- Sur ce petit échantillon WC 2026, M3 devance marginalement M2 (~0.01 sur log-loss). Cohérent avec le caveat noté dans [ADR-010].
- On **conserve M2** comme champion (décision basée sur 290 matchs et 6 tournois) — on ré-évaluera à fin de phase de groupes (72 matchs).
- M4 LightGBM est confirmé sous-performant sur la WC 2026 aussi (46% accuracy, log-loss 1.22).
- Score-level M2 : 3.6% exact-score (1/28). Bruit normal sur 28 matchs ; la cible historique est ~13.8%.

**Conséquences** :
- `python -m wc2026.pipeline refresh` reste la commande de mise à jour quotidienne (à enchaîner via cron ou manuel).
- Le dashboard Performance affiche maintenant les métriques live et le détail match par match du champion.
- À mi-phase de groupes ou en fin de phase, ré-évaluer le ranking M2/M3 avec un échantillon plus stable.

---

## ADR-010 — M2 Poisson est aussi champion sur la prédiction de scores

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Le pivot de scope (ADR-009) a transformé S4 en sprint "prédiction de scores". L'hypothèse initiale était que M3 Dixon-Coles, dont la correction τ vise précisément les scores faibles, reprendrait l'avantage sur M2 lorsqu'on évalue au niveau du score exact (pas seulement W/D/L).

**Résultats — backtest score-niveau sur 290 matchs / 6 tournois** (WC 2018, WC 2022, Euro 2020/2024, Copa America 2021/2024) :

| Modèle | exact-acc | top3-acc | score log-loss | BTTS Brier | O/U 2.5 Brier | goal-diff RPS |
|---|---|---|---|---|---|---|
| **M2 Poisson** | **13.79%** | 37.24% | **2.7707** | 0.2565 | **0.2571** | **0.1519** |
| M3 Dixon-Coles | 12.41% | **37.93%** | 2.7902 | **0.2547** | 0.2580 | 0.1542 |

**Décision** : **M2 Poisson reste le modèle de référence**, y compris pour la prédiction de scores. Il gagne sur 4 métriques sur 6 (exact-score acc, score log-loss, O/U Brier, goal-diff RPS) avec une marge nette sur le critère central (mode = vrai score).

**Analyse** :
- M3 ne gagne que sur **top-3 acc** et **BTTS Brier** — exactement les marchés où la correction τ aide les scores faibles (0-0, 1-0, 0-1, 1-1 dominent ces tableaux). Cohérent avec la théorie.
- M2 est plus piqué sur le mode (mieux le score le plus probable) et plus calibré sur le goal-diff RPS (mieux la "direction" du score).
- L'exact-score accuracy de 13.79% est au-dessus de la fourchette typique de la littérature (8-12% pour les bons modèles publiés).

**Conséquences** :
- Le snapshot live de M2 inclut `score_mode_h`, `score_mode_a`, `e_h`, `e_a`, `p_btts`, `p_over_2_5`.
- Le dashboard affiche le score le plus probable via M2.
- M3 reste disponible si on veut spécifiquement prédire BTTS / scores faibles (page Score detail laisse facilement basculer entre les deux dans une v2).
- L'écart M2-M3 est modeste (1.4pp sur exact-acc) ; sur la WC 2026 spécifique le ranking pourrait s'inverser.

---

## ADR-009 — Pivot scope : prédiction de scores, pas de simulation du tournoi

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : La roadmap initiale incluait une simulation Monte Carlo du bracket et des probabilités de qualification / vainqueur du tournoi (sprint S4 originel). À mi-J0, le besoin métier précisé par l'utilisateur est de se concentrer sur la **prédiction par match** : d'abord le résultat W/D/L (déjà fait), ensuite le **score exact**.

**Décision** :
- Supprimer du périmètre toutes les tâches de simulation du tournoi (vainqueur, probas de qualification par groupe, simulation du bracket).
- Recentrer S4 sur l'extension des modèles M2 / M3 à la distribution de scores P(h, a), avec métriques score-niveau (exact-score accuracy, BTTS Brier, O/U 2.5 Brier, goal-diff RPS).
- Supprimer le module `src/wc2026/simulation/tournament.py` (créé par anticipation, jamais wired).

**Alternatives écartées** : garder la simulation comme bonus optionnel. Rejet : ajoute du bruit dans la roadmap et la UI ; pas de demande utilisateur.

**Conséquences** :
- ROADMAP S4 → "Prédiction de scores", S5 → "Dashboard complet", S6 → "Phase tournoi live".
- Dashboard : la page "Simulation" disparaît, remplacée par une page "🎯 Score detail" (heatmap P(h, a) par match).
- Les modèles Poisson (M2) et Dixon-Coles (M3) deviennent encore plus pertinents : M3 pourrait reprendre l'avantage sur les scores faibles grâce à la correction τ, alors que M2 reste le champion W/D/L.

---

## ADR-008 — M2 Poisson désigné modèle de référence pour les prédictions live

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : À l'issue du sprint S3, benchmark complet de 4 modèles + 1 baseline sur 290 matchs étalés sur 6 tournois récents (WC 2018, WC 2022, Euro 2020, Euro 2024, Copa America 2021, Copa America 2024), avec re-fit walk-forward par journée.

**Résultats poolés (290 matchs)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| marginal | 1.0946 | 0.6652 | 0.2362 | 41.0% | 0.0540 |
| M1_elo | 0.9807 | 0.5775 | 0.1941 | 55.5% | 0.0314 |
| **M2_poisson** | **0.9606** | **0.5661** | **0.1889** | **56.6%** | 0.0324 |
| M3_dixon_coles | 0.9755 | 0.5771 | 0.1925 | 54.8% | **0.0309** |
| M4_lightgbm | 1.0649 | 0.6309 | 0.2106 | 51.0% | 0.0802 |

**Décision** : **M2 Poisson** est désigné modèle de référence pour les prédictions live de la WC 2026. Il gagne sur 4 métriques sur 5 (log-loss, Brier, RPS, accuracy), ne perd que sur l'ECE (calibration) face à M3 — et de très peu (0.0324 vs 0.0309).

**Analyse des perdants** :
- **M3 Dixon-Coles** : la correction τ aide pour le score exact, pas pour le W/D/L. Sur 290 matchs il est dans le bruit de M2 mais légèrement inférieur. Il restera utile en v2 pour les scores exacts.
- **M4 LightGBM** : à peine meilleur que le marginal. Cause probable : (1) features peu informatives au-delà d'Elo, (2) pas de tuning des hyperparams, (3) recency weighting réduit l'échantillon utile. À tuner si on veut le sauver, sinon retiré du run live.

**Conséquences** :
- Le pipeline live utilise M2 comme prédicteur principal.
- M1, M3, M4 restent disponibles dans le dashboard pour comparaison transparente.
- Prochain effort : recalibrer M2 via isotonic regression pour viser ECE < 0.02, et tester si l'ensemble M1+M2 améliore encore.

---

## ADR-007 — M2 Poisson : MLE custom + gradient analytique (pas statsmodels)

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Le M2 Poisson indépendant avec ~400 paramètres (mu, gamma, att[t], def[t] pour ~200 équipes) doit être ajusté à chaque round du backtest. Naïvement avec `scipy.optimize.minimize` sans gradient, l'optimiseur sature le `maxiter` et ne converge pas → modèle complètement biaisé (accuracy 23% sur WC 2022, pire que l'uniforme).

**Décision** : Implémenter le gradient analytique de la log-vraisemblance Poisson et le passer à L-BFGS-B via `jac=True`. La forme vectorisée (avec `np.add.at` pour l'accumulation par équipe) tourne en quelques secondes par fit.

**Conséquences** :
- M2 converge proprement et bat M1 sur les 5 métriques en backtest WC 2022.
- On garde l'indépendance vis-à-vis de statsmodels pour ce composant (plus de contrôle, plus facile à étendre à Dixon-Coles ensuite).
- Bug initial corrigé : dans `_wdl_from_joint`, `np.triu(k=1)` correspondait à "away wins" (h<a) et non "home wins" — les colonnes p_home et p_away étaient inversées. Tests à ajouter en S3 pour bloquer ce type de régression.

---

## ADR-006 — martj42 est la source principale pour les matchs WC 2026 (Wikipedia en backup)

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Initialement, on prévoyait de scraper Wikipedia pour les résultats live de la WC 2026. À la première ingestion, on a constaté que le dataset martj42 est mis à jour quotidiennement et contient déjà **les 72 matchs de phase de groupes WC 2026 avec scores pour ceux joués** (date range observée : 2026-06-11 → 2026-06-27).

**Décision** : martj42 devient la source canonique pour les matchs WC 2026 aussi. Le scraper Wikipedia reste implémenté comme fallback / source d'enrichissement (groupes, ville hôte, heure de coup d'envoi).

**Conséquences** :
- Moins de code fragile à maintenir.
- On dépend d'un seul mainteneur (martj42) — risque de retard d'update. Mitigation : on peut compléter avec Wikipedia si on observe un retard > 24h.
- Le mapping `match -> groupe` n'est pas dans martj42 ; on le déduira par auto-détection (4 équipes qui se rencontrent toutes en round-robin = même groupe) ou via une table statique des groupes.

---

## ADR-005 — Designer un modèle vainqueur à l'issue du benchmark

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : L'utilisateur veut comparer plusieurs modèles, mais doit pouvoir s'appuyer sur **un** modèle de référence pour les prédictions live et la simulation tournoi.

**Décision** : À la fin du sprint S3, comparer M1..M5 sur backtest (WC 2022 + Euro 2024) avec le triplet de métriques `(log-loss, Brier, RPS)`. Désigner officiellement le modèle gagnant. La décision est documentée ici (ADR à venir) avec les chiffres.

**Conséquences** : Le dashboard met en avant ce modèle, les autres restent dispos en mode comparaison. La simulation Monte Carlo utilise les probas du modèle gagnant.

---

## ADR-004 — Sources de données : 100% gratuit, prioriser robustesse

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Aucun accès à des sources payantes (StatsBomb, Opta). Besoin de données historiques propres, d'Elo dynamique, et des résultats WC 2026 en live.

**Décision** :
- **Historique** : dataset `martj42/international_results` (GitHub, CSV public, mis à jour quotidiennement)
- **Elo** : **calculé en interne** à partir du dataset historique (formule World Football Elo standard, K dépendant de la compétition, ajustement par différence de buts). Avantages : pas de dépendance externe fragile, série temporelle complète et reproductible, possibilité d'ajuster la formule.
- **WC 2026 live** : scraping de la page Wikipedia "2026 FIFA World Cup" (table calendrier + résultats)
- **Fallback WC live** : `football-data.org` (API gratuite, 10 calls/min, nécessite clé email)

**Alternatives écartées** :
- *Kaggle* : redondant avec martj42 mais nécessite token.
- *Transfermarkt* : valeur de squad utile, mais ToS scraping fragile → reporté en bonus.
- *FBref xG* : intéressant mais coût d'intégration trop élevé pour un MVP.

**Conséquences** : Le projet est totalement reproductible sans secret/API key (sauf football-data.org en fallback). Risque : si Wikipedia change la mise en forme, le scraping casse → mitigation via fallback API.

---

## ADR-003 — Quatuor de modèles M1..M4 + ensemble M5

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Comparer plusieurs approches pour identifier la meilleure.

**Décision** : 4 modèles cœur + un ensemble :
- **M1 Elo** — conversion Elo → P(W/D/L) (Davidson model). Baseline robuste.
- **M2 Poisson indépendant** — GLM, attaque/défense par équipe. Standard académique.
- **M3 Dixon-Coles** — Poisson bivarié + correction faible score + décroissance temporelle.
- **M4 LightGBM** — classification 3-classes sur features riches.
- **M5 Ensemble** — moyenne pondérée calibrée (poids appris en CV).

**Alternatives écartées (pour MVP)** :
- Bayésien hiérarchique PyMC : reporté en v2 (coût d'implémentation trop élevé pour un MVP en 1 mois).
- Réseau de neurones : sample size trop faible pour justifier la complexité.
- Modèles xG : sources gratuites limitées et bruit élevé sur l'international.

**Conséquences** : Tous les modèles partagent la même interface `Model.fit / predict_proba / predict_score_dist` pour faciliter le benchmark et le swap.

---

## ADR-002 — MVP focus W/D/L, scores exacts en v2

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Prédire un score exact est plus dur à valider (espace de sortie ~50 combinaisons utiles) et nécessite plus de calibration. L'utilisateur veut commencer simple.

**Décision** : v1 livre uniquement P(W/D/L). Les modèles Poisson/Dixon-Coles produisent déjà une distribution de scores en interne, on l'expose plus tard.

**Conséquences** : La métrique principale en v1 est le log-loss multi-classes (et RPS). On évalue avec WD/L pour rester comparable entre tous les modèles, y compris LightGBM qui ne prédit pas de score.

---

## ADR-001 — Pipeline strictement chronologique pour anti-leakage

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Le risque N°1 du projet est le data leakage temporel. Toute feature utilisée pour prédire un match à la date `t` doit être dérivée d'événements **strictement antérieurs** à `t`.

**Décision** :
- Chaque feature porte une date `as_of` ; un assert `as_of < match_date` est appliqué dans le pipeline.
- Les splits de validation sont **walk-forward uniquement**. Pas de k-fold aléatoire.
- Les snapshots de prédictions sont datés et stockés (`data/snapshots/YYYY-MM-DD.parquet`) → audit a posteriori impossible à falsifier.
- Les tests unitaires couvrent les fonctions de construction de features avec des fixtures temporelles.

**Conséquences** : Les performances annoncées en backtest seront représentatives du déploiement live. Coût : un peu plus de code et de tests, mais c'est non-négociable.
