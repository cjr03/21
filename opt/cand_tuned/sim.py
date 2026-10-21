"""Fast in-process simulator (same decks as opt/bench.py) used for tuning."""
import random
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import core

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
RV = {r: (1 if r == "a" else 10 if r in "jqk" else int(r)) for r in RANKS}


def deck(seed):
    d = [r for r in RANKS for _ in range(4)]
    random.Random(seed).shuffle(d)
    return [RV[r] for r in d]


def play(P, seed, chooser=None, stats=None):
    chooser = chooser or core.score_moves
    piles = [0, 0, 0, 0]
    b, H, score = 0, 2, 0
    D = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
    for v in deck(seed):
        D[v] -= 1
        _, idx, hold = chooser(P, piles, b, H, D, v)[0]
        s0 = piles[idx]
        if hold:
            H -= 1
            s0 = 0
        score += core.REW[s0][v]
        if core.BUST[s0][v]:
            b += 1
        piles[idx] = core.NXT[s0][v]
        if stats is not None:
            r = core.REW[s0][v]
            if r:
                stats[r] = stats.get(r, 0) + 1
            if hold:
                stats["hold"] = stats.get("hold", 0) + 1
        if b >= 3:
            break
    if stats is not None:
        stats["busts"] = stats.get("busts", 0) + b
        stats["dead"] = stats.get("dead", 0) + (b >= 3)
    return score
