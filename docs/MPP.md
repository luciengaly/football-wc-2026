# Module MPP — optimiseur de pronostics "Mon Petit Prono"

Objectif : choisir, pour chaque match, le **score à pronostiquer qui maximise
l'espérance de points MPP**, en s'appuyant sur nos modèles de prédiction.

## Règles du jeu (MPP)

Tu donnes **un score** par match. Gains :
- mauvais résultat → **0**
- bon résultat, score inexact → **cote du résultat** (indexée sur les cotes bookmakers)
- bon résultat **et** score exact → cote **+ bonus de rareté**

Bonus de rareté (selon le % de joueurs au bon résultat ayant le score exact) :
`>30% → +20 · 20-30% → +30 · 5-20% → +50 · 0.5-5% → +70 · <0.5% → +100`.

Détails : bonus **x2** (un seul pour 104 matchs, double tout si le résultat est
bon) ; à partir des 16es, le prono vaut pour les **120 min** (pas les tirs au but).

## La fonction objectif

$$s^* = \arg\max_s \Big[\; \text{cote}(R_s)\cdot P(\text{résultat}=R_s) \;+\; \text{bonus}\cdot P(\text{score}=s) \;\Big]$$

C'est l'**espérance de points du match** (règle de décision de Bayes). Décomposition
par linéarité : on touche la cote dès que le résultat impliqué par `s` est correct
(proba `P(R_s)`), et le bonus en plus si le score exact tombe (proba `P(score=s)`).

**Pourquoi maximiser l'espérance suffit (pas de Kelly)** : les paris MPP sont à
poids unitaire, indépendants et sans downside (pas de bankroll, pas de ruine).
Maximiser l'espérance de chaque match maximise — par linéarité — l'espérance du
**total** sur le tournoi. La variance n'intervient jamais.

## Quels modèles, et pourquoi deux

| Terme | Modèle | Raison |
|---|---|---|
| `P(résultat)` | **M13_blend3** (champion W/D/L) | meilleure proba de résultat calibrée |
| `P(score=s)` | **M2 Poisson** (champion score) | distribution jointe `P(h,a)` via λ_H, λ_A |

La cote MPP étant **indexée sur les cotes bookmakers**, le terme de base
`cote(R)·P(R)` est de fait un *value bet* contre le marché : on ne gagne que là
où notre modèle diverge du marché (typiquement via M12i — valeur + blessures).

## Avantage du terrain (important)

L'avantage terrain (`γ` Poisson ≈ +0.3 log-λ, +100 Elo) est :
- **appris** sur les vrais matchs domicile/extérieur de l'historique,
- **appliqué uniquement** quand `neutral=False`.

Pour la WC 2026, `martj42` flague `neutral=True` partout **sauf les 9 matchs où
une nation hôte (USA/Canada/Mexique) joue à domicile**. On utilise ce flag faisant
autorité. → L'avantage terrain ne s'applique **qu'aux hôtes**.

## Pipeline d'entraînement / test (anti-leakage)

```
Pour chaque match-jour J :
  training = results[date < J]            # strictement antérieur
  M13.fit(training) ; M2.fit(training)    # walk-forward, refit par jour
  P(résultat) = M13.predict_proba(match)
  P(score)    = M2.predict_score_dist(match)
  s* = argmax_s [ cote(R_s)·P(R_s) + bonus·P(score=s) ]
```
Cotes, valeurs marchandes et blessures sont lues **as-of la date** (panels datés).
Aucune information post-match n'entre dans la prédiction.

## Backtest (validation)

`python -m wc2026.mpp.backtest --bonus 40` rejoue les matchs joués et compare les
**points réalisés** de :
- `ours` (notre stratégie EV)
- `user` (ton prono réel, colonne `my_pred_*`)
- `favorite` (toujours le favori marché, score modal)
- `mode` (score le plus probable M2)

Résultat sur 54 matchs (bonus moyen 40) :

| Stratégie | Points | Moy/match | Précision résultat | Scores exacts |
|---|---|---|---|---|
| **ours (EV)** | **2476** | **45.9** | 55.6% | 5 |
| favorite | 2135 | 39.5 | 64.8% | 5 |
| user | 1879 | 34.8 | 55.6% | 4 |
| mode | 1714 | 31.7 | 53.7% | 4 |

→ La stratégie EV marque **+32% vs le prono réel**, avec une précision résultat
*plus basse* : elle échange de la précision contre de la valeur (cotes élevées), et
ça paie sur la durée.

## Fichiers

- `src/wc2026/mpp/scoring.py` — barème MPP (`mpp_points`, `rarity_bonus`)
- `src/wc2026/mpp/optimizer.py` — `optimal_score`, `expected_points_table`, `favorite_score`
- `src/wc2026/mpp/backtest.py` — backtest vs baselines
- `data/mpp/mpp_cotes.csv` — saisie des cotes + pronos (cf. `data/mpp/README.md`)
- Dashboard : page **🃏 Conseil MPP** (slate du jour, score conseillé, EV, flags value)

## Limites

- Bonus rareté approximé par une **constante** (faute de popularité par score).
- **Calibration des nuls** = risque principal (la stratégie en joue beaucoup).
- Hôte en équipe extérieure (phase finale) : léger sous-crédit, marginal.
