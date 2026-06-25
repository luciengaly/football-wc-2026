# Saisie des cotes MPP — `mpp_cotes.csv`

Ce fichier liste les 72 matchs de phase de groupes WC 2026. Remplis les cotes
MPP et (optionnellement) ton prono, puis je m'en sers pour le backtest et
l'optimiseur.

## Colonnes à remplir

| Colonne | Quoi mettre |
|---|---|
| `cote_home_win` | points MPP si **`home_team`** gagne (l'équipe de la colonne `home_team`) |
| `cote_draw` | points MPP en cas de **nul** |
| `cote_away_win` | points MPP si **`away_team`** gagne |
| `my_pred_home` | (optionnel) ton score prédit pour `home_team` |
| `my_pred_away` | (optionnel) ton score prédit pour `away_team` |

⚠️ **Attention au sens** : la cote va par **équipe**, pas par position gauche/droite
de MPP. Mets la cote du gain de `home_team` dans `cote_home_win`, même si MPP
affiche cette équipe à droite. (Exemple déjà rempli : `United States,Turkey` →
`cote_home_win=93` car c'est la cote de victoire des USA, `cote_away_win=109`
pour la Turquie.)

## Priorités

- **Matchs `played`** : ce sont eux qui servent au **backtest** (combien de
  points notre stratégie aurait marqué). Le plus utile.
- **Matchs `scheduled`** : pour le conseil live à venir.
- **6 matchs déjà pré-remplis** (J3, ceux de ta capture).

## Noms d'équipes (anglais ici ↔ MPP en français)

Les noms sont en anglais. Correspondances non évidentes :

| Fichier (EN) | MPP (FR) |
|---|---|
| South Korea | Corée du Sud |
| North Macedonia | Macédoine du Nord |
| Czech Republic | Tchéquie |
| Bosnia and Herzegovina | Bosnie-Herzégovine |
| Ivory Coast | Côte d'Ivoire |
| Netherlands | Pays-Bas |
| Germany | Allemagne |
| Spain | Espagne |
| England | Angleterre |
| Wales | Pays de Galles |
| Saudi Arabia | Arabie Saoudite |
| South Africa | Afrique du Sud |
| Cape Verde | Cap-Vert |
| DR Congo | RD Congo |
| United States | États-Unis |
| Scotland | Écosse |
| Croatia | Croatie |
| Morocco | Maroc |
| Senegal | Sénégal |
| Norway | Norvège |
| Sweden | Suède |
| Turkey | Turquie |
| Japan | Japon |
| Australia | Australie |

## Notes

- Tu peux ne remplir que ce que tu as — même 1-2 journées suffisent pour un
  premier backtest.
- Ouvre le fichier dans Excel / Google Sheets (séparateur virgule, encodage
  UTF-8). Garde les colonnes telles quelles.
- Renvoie-moi le fichier rempli (ou colle le contenu) et je lance le backtest +
  l'optimiseur.
