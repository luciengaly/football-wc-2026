"""MPP (Mon Petit Prono) scoring rules.

You predict ONE score per match. Payoff:
  * wrong result (sign of predicted score ≠ actual)      → 0
  * right result, wrong exact score                       → cote(actual result)
  * right result AND exact score                          → cote + rarity bonus

Rarity bonus tiers (by % of correct-result players who picked the exact score):
   >30%      → +20   (commun)
   20-30%    → +30   (rare)
   5-20%     → +50   (très rare)
   0.5-5%    → +70   (mega rare)
   <0.5%     → +100  (ultra rare)

We usually don't have the per-score popularity, so the bonus is treated as a
configurable constant `avg_bonus` (default 0 = base only). When popularity is
known, pass the tier explicitly.
"""

from __future__ import annotations


def result_sign(h: int, a: int) -> str:
    return "H" if h > a else ("A" if a > h else "D")


def rarity_bonus(pct_players: float) -> int:
    """Bonus points given the % of correct-result players who had the exact score."""
    if pct_players > 30:
        return 20
    if pct_players > 20:
        return 30
    if pct_players > 5:
        return 50
    if pct_players > 0.5:
        return 70
    return 100


def mpp_points(
    pred_h: int, pred_a: int,
    actual_h: int, actual_a: int,
    cote_h: float, cote_d: float, cote_a: float,
    bonus: float = 0.0,
) -> float:
    """Points earned for predicting (pred_h, pred_a) given the actual score.

    `bonus` is added only if the exact score is correct (pass the rarity tier
    or an average constant). Base = cote of the actual result if the predicted
    result matches.
    """
    if result_sign(pred_h, pred_a) != result_sign(actual_h, actual_a):
        return 0.0
    cote = {"H": cote_h, "D": cote_d, "A": cote_a}[result_sign(actual_h, actual_a)]
    pts = float(cote)
    if pred_h == actual_h and pred_a == actual_a:
        pts += bonus
    return pts
