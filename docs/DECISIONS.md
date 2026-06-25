# Decision log — ADR style

Toutes les décisions structurantes du projet sont consignées ici, du plus récent au plus ancien.
Format : numéro, titre, contexte, décision, alternatives, conséquences.

---

## ADR-028 — Module MPP : optimiseur de pronostics en espérance de points

**Date** : 2026-06-25
**Statut** : ✅ accepté

**Contexte** : Application concrète du système à "Mon Petit Prono" (MPP), où l'on pronostique un score par match : gain = cote du résultat (indexée sur les bookmakers) + bonus de rareté si score exact, 0 si résultat faux. Pas de bankroll, mise unitaire, sans downside.

**Décision** : Comme les paris sont à poids unitaire, indépendants et sans ruine, la stratégie optimale est de **maximiser l'espérance de points match par match** (pas de Kelly, la variance n'intervient pas). Règle de décision de Bayes :

`s* = argmax_s [ cote(R_s)·P(résultat=R_s) + bonus·P(score=s) ]`

avec `P(résultat)` = M13 (champion W/D/L) et `P(score)` = M2 (champion score). Bonus traité comme constante paramétrable (la popularité par score n'est pas disponible historiquement).

**Validation (backtest 54 matchs joués, bonus moyen 40)** :

| Stratégie | Points | Précision résultat | Scores exacts |
|---|---|---|---|
| **EV (nous)** | **2476** | 55.6% | 5 |
| Toujours favori | 2135 | 64.8% | 5 |
| Prono réel utilisateur | 1879 | 55.6% | 4 |
| Mode M2 | 1714 | 53.7% | 4 |

Notre stratégie marque **+32% vs le prono réel**. Insight central : précision résultat *plus basse* mais bien plus de points — on échange précision contre value (cotes élevées : outsiders, nuls sous-cotés).

**Avantage terrain** : appliqué uniquement aux nations hôtes via le flag `neutral` de martj42 (vérifié : identique à l'heuristique `home_team ∈ hôtes`, 0 écart sur 72 matchs). Conforme à la réalité d'une CdM sur terrain neutre.

**Conséquences** :
- Nouveau module `src/wc2026/mpp/` (scoring, optimizer, backtest) + page dashboard "🃏 Conseil MPP".
- Saisie des cotes via `data/mpp/mpp_cotes.csv`.
- Doc dédiée : `docs/MPP.md`.

**Limites** : bonus rareté constant (pas la popularité par score) ; calibration des nuls = risque clé ; hôte en extérieur légèrement sous-crédité.

---

## ADR-027 — Valeur ajustée des blessures (M12i) : bat M12, intégrée au champion M13

**Date** : 2026-06-23
**Statut** : ✅ accepté

**Contexte** : Bonus du sprint S11. La table `player_injuries.csv` (salimt) fournit des périodes de blessure datées (`from_date`, `end_date`) — exploitables sans leakage pour savoir qui était indisponible à la date d'un match. Idée : remplacer la valeur du squad par celle du **squad réellement disponible** (top-26 parmi les joueurs non blessés à la date).

**Implémentation** : second panel `(pays, mois) → (valeur off/def)` calculé en excluant les joueurs blessés au début du mois (346 235 cellules (joueur, mois) blessées). Flag `use_availability` sur M12. Validation qualitative : France à -19.5% avant WC 2022 (Pogba/Kanté/Benzema blessés — historiquement exact).

**Backtest standalone (290 matchs)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M12 (valeur pleine) | 0.9710 | 0.5741 | 0.1939 | 53.8% | 0.0203 |
| **M12i (disponibilité)** | **0.9644** | **0.5707** | **0.1921** | **54.5%** | **0.0178** |

M12i **bat M12 sur les 5 métriques** — gain net.

**Dans le blend 3-voies (134 matchs)** :

| Blend | log_loss | brier | rps | acc | ece |
|---|---|---|---|---|---|
| M3+M9+M12 | 1.0158 | 0.6040 | 0.2018 | 51.5% | 0.0191 |
| **M3+M9+M12i** | **1.0136** | **0.6031** | **0.2014** | 51.5% | 0.0297 |

Améliore log-loss/Brier/RPS (ECE légèrement dégradée mais reste bonne).

**Décision** : Le champion **M13_blend3 utilise désormais la valeur ajustée des blessures** (M12i) comme composante valeur. Le M12 standalone reste en valeur pleine dans le registry (référence Peeters de l'ADR-025, comparaison transparente).

**Conséquences** :
- `Blend3.value_ = MarketValueModel(use_availability=True)`
- Deux panels cachés : `tm_value_panel.parquet` (plein) + `tm_value_panel_avail.parquet` (disponibilité)
- Snapshots re-backfillés
- `get_shared_store()` construit les deux panels une fois

**Caveats** :
- Résolution mensuelle approximative (un joueur blessé mi-mois compte comme indispo tout le mois si la blessure couvre le 1er).
- Qualité variable des données blessures (Japon -48% en 2022 semble surévalué). Le gain agrégé reste positif malgré ce bruit.

---

## ADR-026 — M13_blend3 (M3 + M9 + M12, équipondéré) désigné champion W/D/L

**Date** : 2026-06-23
**Statut** : ✅ accepté

**Contexte** : Avec M12 (valeur marchande, ADR-025) on dispose pour la première fois de **trois signaux orthogonaux** : résultats historiques (M3), marché des paris (M9), valeur des joueurs (M12). C'est précisément la décorrélation qui manquait aux blends ratés de S7 (M5/M6 étaient des variantes de M2).

**Sweep des poids sur 134 matchs (les 3 signaux présents)** :

| Poids M3/M9/M12 | log_loss | brier | rps | acc | ece |
|---|---|---|---|---|---|
| (1,0,0) M3 | 1.0368 | 0.6140 | 0.2062 | 52.2% | 0.0456 |
| (0,1,0) M9 | 1.0195 | 0.6091 | 0.2034 | 51.5% | 0.0514 |
| (0,0,1) M12 | 1.0238 | 0.6080 | 0.2042 | 52.2% | 0.0450 |
| M11 (0.5,0.5,0) | 1.0236 | 0.6089 | 0.2037 | 52.2% | 0.0373 |
| **(⅓,⅓,⅓)** | **1.0159** | **0.6041** | **0.2018** | 51.5% | **0.0191** |

L'équipondéré 3-voies **bat tous les modèles précédents** sur log-loss, Brier, RPS et ECE simultanément.

**Décision** : `M13_blend3` = **moyenne équipondérée des composantes disponibles** par match :
- 3 signaux présents → (⅓, ⅓, ⅓)
- M3 + M12 (pas d'odds) → (½, ½)
- M3 seul → M3
M3 est toujours disponible, donc il y a toujours ≥1 composante.

**Pourquoi équipondéré et pas un simplex optimisé** : sur 134 matchs un simplex sur-apprend ; la surface log-loss(w) est plate autour du centre. L'équipondéré est le choix robuste et explicable.

**Pourquoi champion pour le live malgré un match nul sur l'ensemble 290** : sur les 290 matchs (dont 156 sans odds), M13 ≈ M11 (log-loss 0.9548 vs 0.9546, M13 légèrement moins bon en accuracy). MAIS ce régime n'est pas représentatif du live : **31/32 matchs WC 2026 à venir ont odds + valeur**, donc le régime pertinent est celui des 134 matchs où M13 gagne nettement. M13 est désigné champion sur cette base, avec le caveat documenté.

**Conséquences** :
- `CHAMPION_MODEL = "M13_blend3"` (étoile ⭐ dashboard)
- M12 et M13 ajoutés au registry du pipeline (8 modèles)
- M2 reste champion score-distribution (ni odds ni valeur ne donnent de distribution de scores)
- Singleton `get_shared_store()` pour ne charger les 900k valeurs qu'une fois par run
- M11 reste disponible mais déclassé

**Caveats** :
- Bench 134 matchs, sample modeste.
- M12 sans odds peut légèrement dégrader l'accuracy → surveiller en live ; si le régime sans-odds devient fréquent (knockout, fixtures lointaines), envisager de ne pas inclure M12 quand odds absentes.

---

## ADR-025 — M12_market_value : la valeur Transfermarkt bat l'Elo (Peeters confirmé)

**Date** : 2026-06-23
**Statut** : ✅ accepté

**Contexte** : Direction "approches avancées" — implémentation d'un modèle piloté par la **valeur marchande des joueurs** (Transfermarkt), motivé par Peeters (2018, *Int. J. Forecasting*) qui montre que les valeurs agrégées battent FIFA ranking et Elo pour l'international.

**Données** : dataset GitHub `salimt/football-datasets` — 901k valeurs marchandes datées (2003-2025), positions, sélections nationales. Résolution `team_id → pays` par citoyenneté modale des joueurs sélectionnés (corrige le biais diaspora : la Côte d'Ivoire passe de 716M€ gonflés à 407M€ corrects). Panel mensuel `(pays, mois) → (valeur off/def)` précalculé et caché (`data/processed/tm_value_panel.parquet`, 40k entrées) → lookups instantanés.

**Structure du modèle (novatrice vs Peeters qui n'utilise que la valeur totale)** :
```
log(λ_home) = β0 + β_att·val_off_H + β_def·val_def_A + γ·domicile
log(λ_away) = β0 + β_att·val_off_A + β_def·val_def_H
```
La valeur offensive de l'équipe (Attack+Midfield) pilote son attaque ; la valeur défensive de l'adversaire (Defender+GK) module les buts encaissés.

**Coefficients ajustés (MLE, time-decay 3 ans)** : β_att = +0.186, β_def = -0.198, γ = +0.317 — exactement le signe attendu par la théorie.

**Bench 290 matchs (6 tournois)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M1_elo | 0.9807 | 0.5775 | 0.1941 | 55.5% | 0.0314 |
| M2_poisson | 0.9606 | 0.5661 | 0.1889 | 56.6% | 0.0324 |
| M3_dixon_coles | 0.9607 | 0.5662 | 0.1889 | 57.2% | 0.0259 |
| **M12_market_value** | 0.9710 | 0.5741 | 0.1939 | 53.8% | **0.0203** |

**Lecture** :
- **M12 bat M1 (Elo)** sur log-loss/Brier/RPS → confirme Peeters sur notre pipeline.
- M12 seul est sous M2/M3 (notamment en accuracy), MAIS a la **meilleure ECE de tous** (0.0203).
- Surtout : M12 est **orthogonal** à M2/M3/M9 (valeur joueurs ≠ résultats ≠ odds). C'est ce qui le rend précieux **en blend** (cf. ADR-026), là où les variantes de M2 (S7) avaient échoué.

**Détails techniques** :
- Couverture : 48/48 équipes WC 2026 résolues, 290/290 matchs du bench.
- Coût : panel 390s une fois, puis fit 0.2s.
- Bonus disponible mais non exploité : `player_injuries.csv` (pondération par dispo réelle) — piste future.

**Conséquences** : voir [ADR-026] pour l'intégration en blend champion.

---

## ADR-024 — M11_blend (M3 + M9, 50/50) désigné nouveau champion W/D/L

**Date** : 2026-06-22
**Statut** : ✅ accepté

**Contexte** : Suite à l'évidence empirique d'ADR-023 (M9 bat M3 sur log-loss/Brier/RPS), on cherche s'il existe un blend M3 + M9 qui combine les forces des deux : la calibration de M3 (ECE 0.046) et le signal raffiné de M9 (log-loss 1.020).

**Méthode** : sweep sur 134 matchs historiques, blend pondéré `p = w * p_M9 + (1-w) * p_M3`, w ∈ {0, 0.3, 0.5, 0.6, 0.7, 1.0} :

| w_M9 | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| 0.0 (M3) | 1.0368 | 0.6140 | 0.2062 | 52.2% | 0.0456 |
| 0.30 | 1.0276 | 0.6103 | 0.2044 | 52.2% | **0.0320** |
| **0.50** | **1.0236** | **0.6089** | **0.2037** | **52.2%** | 0.0373 |
| 0.70 | 1.0209 | 0.6083 | 0.2033 | 51.5% | 0.0390 |
| 1.00 (M9) | 1.0195 | 0.6091 | 0.2034 | 51.5% | 0.0514 |

**Décision** : `M11_blend` = M3 ⊕ M9 avec **w = 0.5** (no-information prior). À cette valeur, M11 gagne sur **Brier ET RPS**, conserve l'accuracy de M3 (52.2%), et présente une ECE de 0.037 qui est meilleure que les deux modèles séparés (M3 0.046, M9 0.051).

**Comportement de fallback** : quand M9 retourne NaN (matchs sans odds publiés), M11 prend purement M3. Donc M11 ≥ M3 sur toutes les métriques garanti — pire cas il est identique à M3.

**Conséquences** :
- `CHAMPION_MODEL = "M11_blend"` dans le dashboard (étoile ⭐)
- M11 ajouté au registry du pipeline live
- M2 reste **champion score-distribution** (M11 ne produit pas de distribution de scores — M9 n'en a pas)
- Choix du w = 0.5 documenté ici ; pourrait être tuné dynamiquement après ~50 matchs WC 2026 si les pondérations divergent
- Snapshots 11-22 juin re-backfillés avec M11

**Caveats** :
- Bench sur 134 matchs uniquement. Sample modeste, écart-type sur log-loss ~0.04.
- Le blend a été choisi sur le même set qui sert à le valider — léger risque de "best on test". Mais la surface log-loss(w) est très lisse (0.5 à 0.7 quasi-identiques) → robuste.

---

## ADR-023 — M9_odds bat M2/M3 sur le bench historique 134 matchs

**Date** : 2026-06-22
**Statut** : ✅ accepté (positif)

**Contexte** : ADR-022 documente l'intégration live de M9_odds via The Odds API mais ne pouvait pas le bencher historiquement (free tier = pas d'odds passés). Découverte d'un dataset GitHub (eatpizzanot/soccer-dataset) qui aggrège The Odds API + Football-Data.co.uk depuis juin 2020 — couvre 170/290 matchs de notre bench 6-tournois (WC 2022 + Euro 2020/2024 + Copa 2024).

**Résultats (134 matchs avec coverage des outcomes connus)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M1_elo | 1.0892 | 0.6428 | 0.2191 | 50.0% | 0.0613 |
| M2_poisson | 1.0374 | 0.6148 | 0.2063 | 51.5% | 0.0493 |
| M3_dixon_coles | 1.0368 | 0.6140 | 0.2062 | **52.2%** | **0.0456** |
| **M9_odds** | **1.0195** | **0.6091** | **0.2034** | 51.5% | 0.0514 |

**Lecture** : M9 gagne **log-loss, Brier, RPS** (les 3 métriques calibration-sensibles). Marge nette sur log-loss (+1.7% vs M3). M3 reste devant sur accuracy et ECE.

**Per-tournament log-loss** :
- Copa 2024 (3 matchs, sample minuscule) : M1 best
- Euro 2020 (34) : M3 best
- Euro 2024 (42) : **M9 best**
- WC 2022 (55) : **M9 best**

Les wins de M9 sont sur les plus gros échantillons, ce qui renforce la conclusion.

**Caveats** :
- Bench réduit à 134 matchs (WC 2018 et Copa 2021 sans odds disponibles)
- Le dataset eatpizzanot ne donne qu'**1 bookmaker par fixture** (probablement le premier disponible) au lieu d'une médiane sur plusieurs. En live on aggrège 24 bookmakers → M9 live devrait être encore meilleur calibré.
- Le free tier de The Odds API ne permet pas la reproduction directe — on a documenté la dépendance externe à eatpizzanot/soccer-dataset.

**Décision** : voir [ADR-024] pour la désignation du blend M11 comme champion. M9 seul est jugé légèrement trop confiant (ECE 0.051 vs M3 0.046).

---

## ADR-022 — Sprint S10 : intégration des cotes via The Odds API

**Date** : 2026-06-22
**Statut** : ✅ accepté (eval live en attente)

**Contexte** : Suite à la conclusion de [ADR-021] (plafond atteint avec données seules), passage à la source externe la plus prometteuse : les cotes bookmakers. The Odds API offre un tier gratuit (500 req/mois) couvrant le `soccer_fifa_world_cup`. 24 bookmakers européens couverts (Pinnacle, Bet365, William Hill, Betclic, etc.).

**Décision** :
- Intégrer un nouveau modèle **M9_odds** qui n'a pas de phase fit : il fait juste un lookup des cotes médianes (sur les 24 bookmakers) et les devig via 1/odds renormalisé.
- M9 retourne NaN pour les matchs sans cotes (typiquement les fixtures trop éloignées). Le dashboard et l'évaluation traitent les NaN.
- Le pipeline `refresh` fetche les cotes en best-effort (échec non bloquant).
- Quota géré conservativement : 1 fetch quotidien max (`x-requests-remaining` est loggé à chaque appel).

**Couverture initiale** (J+3, fetch du 2026-06-22) :
- 31 matchs × 24 bookmakers
- Coverage : 31/32 matchs upcoming WC 2026 (Argentina-Austria du jour manquant car déjà clos par les bookmakers)
- Fuzzy date matching (±1 jour) pour absorber le décalage UTC ↔ date locale de martj42

**Premières observations sur les divergences M2/M3 vs M9** :
- M9 systématiquement **plus confiant sur les favoris** : Portugal vs Uzbekistan M2 68% → M9 82% ; Bosnia vs Qatar 50% → 66% ; Tunisia vs Pays-Bas 70% → 85%
- Signature classique du marché vs modèles académiques : le marché intègre des signaux non disponibles dans nos features (blessures, suspensions, motivation, météo).

**Limites du free tier** :
- Pas d'odds historiques → impossible de backtest M9 sur les 6 tournois passés ni sur les 40 matchs WC 2026 déjà joués.
- Eval seulement live, au fil des matchs joués sur les 7 prochains jours.

**Prochaines étapes** :
- Configurer un refresh quotidien (`scripts/daily_refresh.bat` → odds + predict + snapshot)
- Évaluer M9 vs M2/M3 au fil des matchs joués
- Si M9 confirme son avance, construire ensemble M2/M3 + M9 ou désigner M9 champion W/D/L
- ADR de désignation champion à mettre à jour selon résultats

---

## ADR-021 — Plafond empirique atteint : M2/M3 sont à l'optimum extractible des données

**Date** : 2026-06-19
**Statut** : 📌 reconnaissance d'état

**Contexte** : Sprints S7 (vague quick wins) et S9 (M8 score-direct) cumulent **5 expériences modèles successives** qui toutes échouent à battre M2 / M3 sur 290 matchs :

| Modèle | Approche | Résultat |
|---|---|---|
| M5_nb | Negative Binomial pour over-dispersion | ❌ α optimal ≈ 0, dégénère en Poisson (ADR-015) |
| M6_stack | LogReg meta-learner sur M1-M4 | ❌ Sous M2/M3 sur 290 matchs (ADR-016) |
| M2_cal / M3_cal | Calibration isotonic out-of-sample | ❌ -4pp d'accuracy (perte de training, ADR-017) |
| M7 empirical Bayes | Ridge ∈ {0.005, 0.05, 0.5, 2} sur M2 | ❌ Toutes valeurs ≈ M2 actuel, optimum déjà atteint |
| M8 LGBM 49-classes | Classifier direct sur le score exact | ❌ Sous M2 sur exact-acc / top3-acc / score log-loss (ADR-020) |

**Interprétation** : avec **49 477 matchs internationaux historiques + Elo + forme + repos + confédération + lambdas Poisson**, la performance plafonne à :
- **log-loss ≈ 0.96** (M2 / M3)
- **accuracy W/D/L ≈ 57%** (M3)
- **exact-score acc ≈ 13.8%** (M2)
- **ECE ≈ 0.03**

Ces chiffres sont au niveau de la **littérature académique SOTA** en prédiction internationale (Constantinou et al. ~58% accuracy, Karlis-Ntzoufras ~14% exact-score). Le gain marginal possible avec nos données s'arrête là.

**Décision** : Acceptation du plafond. Les nouveaux modèles à essayer ne seront pas des variations sur (Poisson, hyperparam, ensemble) — il faudra changer la **source de signal** :

1. **Données joueurs** (Transfermarkt/FBref scraping) — absences, suspensions, market value squad. Gain attendu : +1-3pp accuracy. Coût : 2-3 jours d'ingénierie fragile.
2. **Cotes des bookmakers** — meilleure source de signal connue. Gain attendu : +5-10pp. Coût : scraping/légal incertain.
3. **xG par équipe sur fenêtre récente** — Understat/FBref. Sparse pour l'international, peu d'intérêt.
4. **Manager / formation tactique** — quasi inexploitable sans saisie manuelle.

**Conséquences** :
- Fin du sprint d'ajout de modèles (S7-S9 clos).
- Si nouveau modèle, il devra venir avec une nouvelle feature externe (pas une nouvelle architecture).
- Réorienter le projet vers : (a) monitoring live + métriques cumulées (déjà fait), (b) données joueurs si l'utilisateur veut investir, (c) acceptation du plafond et focus opérationnel.

---

## ADR-020 — M8 LightGBM 49-classes : résultat négatif

**Date** : 2026-06-19
**Statut** : ❌ rejeté

**Contexte** : Sprint S9 — dernier essai après l'abandon de PyMC. Approche fondamentalement différente : classification multinomiale à 49 classes ({0..6} × {0..6}) avec features standard + λ_H/λ_A de M2 en input. L'idée : LightGBM apprend des *corrections* sur la base de M2.

**Itérations testées** :

| Config | exact_acc | top3_acc | score_log_loss |
|---|---|---|---|
| Class-weighted (cap 5x) + n_est=500 | 6.25% | 31.25% | 3.31 |
| No weights, n_est=500 | 7.81% | 26.56% | 3.57 |
| No weights, n_est=200, reg=2.0 | 9.4% | 28.1% | 3.14 |
| **No weights, n_est=100, reg=3.0** | **9.4%** | **26.6%** | **3.04** |
| **M2 baseline** | **10.9%** | **32.8%** | **3.03** |

Même avec configuration la plus simple, M8 **fait à peine match nul** avec M2 sur le score-log-loss (3.04 vs 3.03) et perd sur exact-acc (-1.5pp) et top3-acc (-6.2pp).

**Diagnostic** :
- 49 classes × ~50k samples = parameters per class ≈ 1000. Trop fin pour les classes rares (5-1, 4-3...).
- Les features standard (form, rest, confederation) n'ajoutent pas de signal au-delà des λ M2.
- Le 49-class classification convexe sur cross-entropy ne reproduit pas la structure générative correcte (corrélation goals_h ⊥ goals_a sachant team strengths).

**Conséquences** :
- M8 rejeté du bench.
- Le `src/wc2026/models/lgbm_score.py` reste dans le codebase pour reproductibilité.
- Voir [ADR-021] pour la conclusion globale "plafond empirique atteint".

---

## ADR-019 — Skip de M7 PyMC bayésien hiérarchique : empirical Bayes négatif

**Date** : 2026-06-19
**Statut** : ⏸️ reporté (négatif)

**Contexte** : Sprint S8 — implémentation prévue d'un Dixon-Coles hiérarchique bayésien via PyMC (ADVI). Le bénéfice attendu : shrinkage adaptatif des forces d'équipe via un hyper-prior `att[t] ~ Normal(0, σ_att)` au lieu d'un ridge fixe.

**Quick check empirical Bayes** : avant d'investir 1-2 jours dans PyMC, on a tuné le ridge de M2 sur 4 valeurs {0.005, 0.05, 0.5, 2.0} via bench 6 tournois (290 matchs) :

| Ridge | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| 0.005 | 0.9606 | 0.5660 | 0.1889 | 56.90% | 0.0307 |
| 0.05 (actuel) | 0.9606 | 0.5661 | 0.1889 | 56.55% | 0.0324 |
| 0.5 | 0.9623 | 0.5685 | 0.1897 | 56.55% | 0.0276 |
| 2.0 | 0.9699 | 0.5754 | 0.1927 | 55.86% | 0.0318 |

**Résultat** : la régularisation optimale est dans la fourchette [0.005, 0.05] et **M2 est déjà à l'optimum**. Tuning du ridge ne change rien sur log-loss/Brier/RPS, marginal sur ECE (mais dégrade l'accuracy).

**Implication pour PyMC** : la valeur ajoutée du Bayesian hiérarchique = régularisation adaptative + shrinkage. Si la régularisation fixe optimale ne bouge pas la performance, la version adaptative non plus. Estimation du gain réel : 0% sur log-loss, peut-être 0-0.5% sur ECE.

**Décision** : Skip de l'implémentation PyMC. Coût (1-2j + dépendances + g++ manquant sur Windows + mode Python pur 5-10x plus lent) > bénéfice attendu (~0%). Le code `src/wc2026/models/bayesian.py` est laissé comme squelette pour usage futur si on change d'avis.

**Conséquences** :
- S8 fermé sans implémentation.
- PyMC 6.0.1 installé reste utilisable si on décide de re-investir.
- Voir [ADR-021] pour la conclusion sur le plafond empirique.

---

## ADR-018 — Champion dual : M3 W/D/L + M2 score-niveau

**Date** : 2026-06-19
**Statut** : ✅ accepté

**Contexte** : Avec le refactor 2-étapes de M3 (ADR-013) et le bench étendu post-S7, M3 Dixon-Coles dépasse maintenant M2 sur accuracy (57.24% vs 56.55%) et ECE (0.0259 vs 0.0324) tout en étant statistiquement à égalité sur log-loss / Brier / RPS. Mais sur le score-level (ADR-010), M2 garde l'avantage exact-score (13.79% vs 12.41%).

**Décision** : Champion dual selon la métrique :
- **M3 Dixon-Coles** = champion **W/D/L** (page Predictions, choix d'outcome)
- **M2 Poisson** = champion **distribution de scores** (page Score detail, heatmap + top-5)

**Conséquences** :
- `CHAMPION_MODEL` du dashboard pointe vers M3 sur Predictions (l'étoile dans le dropdown)
- La page Score detail garde M2 comme moteur (cohérent avec le bench score-niveau ADR-010)
- Documentation : noter clairement que les deux champions servent des questions différentes

---

## ADR-017 — Calibration isotonic : résultat négatif sur 290 matchs

**Date** : 2026-06-19
**Statut** : ❌ rejeté

**Contexte** : Sprint S7 vague 1 prévoyait d'appliquer une calibration isotonic out-of-sample à M2 et M3. L'idée : améliorer l'ECE de ~0.06 à <0.04 sans dégrader log-loss/accuracy.

**Approche** :
1. Split du training en pre-cutoff (base train) / post-cutoff (cal set, 365j)
2. Fit base sur pre-cutoff, prédire sur cal set out-of-sample
3. Fit isotonic par classe sur (proba_pred, outcome)
4. À predict-time, utiliser la base pré-cutoff + calibrators (pas de refit, sinon les calibrators ne s'appliquent plus)

**Résultats (bench 6 tournois)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M2_poisson | 0.9606 | 0.5661 | 0.1889 | 56.55% | 0.0324 |
| M2_cal | 0.9931 | 0.5778 | 0.1930 | 52.76% | 0.0423 |

**M2_cal est PIRE que M2 sur toutes les métriques**, y compris l'ECE qu'elle était censée améliorer. Pourquoi :
- Le base model perd la dernière année de données (utilisée pour calibrer). En foot international avec ~5-6k matchs/an, c'est ~5k matchs en moins → forces d'équipes plus anciennes, moins précises.
- Les calibrators apprennent un biais qui ne transfère pas bien parce que le base sous-jacent est sous-entraîné.
- Solution alternative théorique : K-fold cross-validation pour utiliser toutes les données, mais coût × K ≈ 10×.

**Conséquences** :
- Calibration isotonic rejetée pour ce projet. Wrapper `CalibratedModel` reste dans le code (utilisable si besoin futur) mais pas dans le bench officiel.
- Pour réduire l'ECE, préférer une approche structurelle : passer à M3 (ECE 0.0259 sans calibration) ou plus tard M7 bayésien.

---

## ADR-016 — M6_stack (LogReg meta-learner) : résultat négatif sur 290 matchs

**Date** : 2026-06-19
**Statut** : ❌ rejeté

**Contexte** : Sprint S7 vague 1 prévoyait un stacking ensemble M1-M4 → LogReg multinomial L2. Premier essai WC 2022 (64 matchs) prometteur : M6_stack avait la meilleure log-loss (1.029 vs M2 1.043). Mais 64 matchs = bruit.

**Bench étendu (290 matchs, 6 tournois)** :

| Modèle | log_loss | brier | rps | accuracy | ece |
|---|---|---|---|---|---|
| M2_poisson | 0.9606 | 0.5661 | 0.1889 | 56.55% | 0.0324 |
| M3_dixon_coles | 0.9607 | 0.5662 | 0.1889 | 57.24% | 0.0259 |
| **M6_stack** | 0.9638 | 0.5719 | 0.1913 | 55.52% | 0.0238 |

**M6_stack ne bat plus** M2/M3 — sauf sur l'ECE marginale (0.0238 vs 0.0259 pour M3). Sur les 4 autres métriques, M6_stack est légèrement en-dessous.

**Diagnostic** :
- M1, M2, M3 sont **trop corrélés** entre eux (tous basés sur le même squelette Elo/Poisson). Le stacking gagne quand les base models ont des erreurs décorrélées.
- M4 LightGBM est faible et apporte plus de bruit que de signal.
- La meta-training set (~365j × 15 matchs/j) est suffisante pour LogReg mais le meta-learner finit par sur-pondérer M2 ou M3 sans vraie diversité à exploiter.

**Conséquences** :
- M6_stack rejeté du bench officiel.
- Retiré du `_registered_models()` pour ne pas alourdir le pipeline quotidien (~14s/jour économisés).
- Pour qu'un stacking marche il faudrait des base models réellement différents (ex : M2 + M7 bayésien + M8 score-direct LightGBM). On y reviendra après S8.

---

## ADR-015 — M5_nb Negative Binomial : essai négatif (dégénère en Poisson)

**Date** : 2026-06-19
**Statut** : ❌ rejeté (NB non retenu dans le bench officiel)

**Contexte** : Sprint S7 vague 1 prévoyait d'ajouter une Negative Binomial pour gérer l'over-dispersion empirique des buts internationaux (var/mean = 1.80 sur 49k matchs, soit 80% au-dessus de Poisson). Réponse aussi au retour utilisateur sur "le modèle prédit rarement 3+ buts".

**Mesure** : Implémentation NB2 (Var = μ + α μ²) avec fit 2-étapes (Poisson MLE puis α via Brent 1D). Sur 49 405 matchs d'entraînement :

```
alpha=0.000   weighted NLL = 7,861.70   (Poisson)
alpha=0.050   weighted NLL = 7,878.40
alpha=0.100   weighted NLL = 7,905.97
alpha=0.200   weighted NLL = 7,978.13
alpha=0.500   weighted NLL = 8,240.27
```

L'optimum est à **α = 0** : la NB dégénère exactement en Poisson.

**Explication** : L'over-dispersion marginale (var=2.64, mean=1.47) est **entièrement expliquée par l'hétérogénéité des équipes** modélisée via les paramètres att/def. Conditionnellement à μ_match (la force du couple), la variance résiduelle = mean → Poisson est correcte. C'est un résultat statistique connu : la "mixture de Poissons" (un par couple d'équipes) produit une marginale over-dispersée même si chaque composant est Poisson pur.

**Conséquences** :
- M5_nb laissé dans le codebase (`src/wc2026/models/negbin.py`) pour reproductibilité de l'expérience.
- **Pas registered** dans le bench officiel — identique numériquement à M2 sur nos données.
- Le retour utilisateur "rarement 3+ buts" n'est donc pas un problème de dispersion mais un problème de MODE (mode de Poisson(λ=1.5) = 1). Réponse adéquate : affichage E[H]:E[A] + top-5 scores (déjà fait).
- La voie pour vraiment capturer des scores élevés serait soit (a) un **modèle de mélange à 2 régimes** ("tight" vs "open game"), soit (b) une **NB avec α dépendant de μ** (gamma-Poisson hiérarchique). Reporté en S8 ou plus tard.

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
