"""MPP optimal-score chooser.

Given the three result cotes, our model's P(result) and joint score
distribution, pick the score maximising expected MPP points:

    E[pts | score s=(h,a)] = cote(R_s) · P(result = R_s) + bonus · P(score = s)

The first term depends only on the implied result R_s; the second rewards
nailing the exact score. With bonus = 0 the optimiser maximises the base term
and breaks ties toward the most likely score of the chosen result.
"""

from __future__ import annotations

import numpy as np

from wc2026.mpp.scoring import result_sign


def expected_points_table(
    cote_h: float, cote_d: float, cote_a: float,
    p_result: tuple[float, float, float],   # (P_home, P_draw, P_away)
    score_dist: np.ndarray,                 # joint P(h, a), shape (G+1, G+1)
    bonus: float = 0.0,
) -> np.ndarray:
    """Return an EV matrix the same shape as score_dist."""
    cote = {"H": cote_h, "D": cote_d, "A": cote_a}
    p_r = {"H": p_result[0], "D": p_result[1], "A": p_result[2]}
    g = score_dist.shape[0]
    ev = np.zeros_like(score_dist)
    for h in range(g):
        for a in range(g):
            r = result_sign(h, a)
            ev[h, a] = cote[r] * p_r[r] + bonus * score_dist[h, a]
    return ev


def optimal_score(
    cote_h: float, cote_d: float, cote_a: float,
    p_result: tuple[float, float, float],
    score_dist: np.ndarray,
    bonus: float = 0.0,
) -> tuple[tuple[int, int], float]:
    """Return ((h, a), expected_points) maximising EV."""
    ev = expected_points_table(cote_h, cote_d, cote_a, p_result, score_dist, bonus)
    # Break ties (e.g. bonus=0, where all scores of a result share base EV)
    # toward the most likely score, so we keep exact-score upside for free.
    ev_tiebroken = ev + 1e-9 * score_dist
    h, a = np.unravel_index(ev_tiebroken.argmax(), ev.shape)
    return (int(h), int(a)), float(ev[h, a])


def favorite_score(
    cote_h: float, cote_d: float, cote_a: float,
    score_dist: np.ndarray,
) -> tuple[int, int]:
    """Baseline: back the market favourite (lowest cote), play its modal score."""
    fav = min([("H", cote_h), ("D", cote_d), ("A", cote_a)], key=lambda x: x[1])[0]
    g = score_dist.shape[0]
    best, best_p = (1, 0), -1.0
    for h in range(g):
        for a in range(g):
            if result_sign(h, a) == fav and score_dist[h, a] > best_p:
                best_p = score_dist[h, a]
                best = (h, a)
    return best
