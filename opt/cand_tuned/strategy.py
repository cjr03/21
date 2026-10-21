"""Tuned structural-heuristic strategy for the 21 solitaire calculator.

rank_moves(game, rank) -> [(score, idx, use_hold), ...] best first, for every legal move.
Scores are expected points: immediate reward + expected future points (relative pile values
plus an expected-future baseline of RHO points per remaining card), so gaps between moves are
point differences.  See core.py for the algorithm and params.json for the tuned weights.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import twentyone as _t  # noqa: E402
import core  # noqa: E402

_P = None


def params():
    global _P
    if _P is None:
        d = {}
        try:
            with open(os.path.join(_HERE, "params.json")) as f:
                d = json.load(f)
        except OSError:
            pass
        _P = core.Params(d)
    return _P


def rank_moves(game, rank):
    P = params()
    v = _t.RANK_VALUE[rank]
    rem = game.remaining_values()
    D = [0] * 11
    for val, n in rem.items():
        D[val] = n
    D[v] -= 1
    piles = [core.pile_index(p) for p in game.piles]
    moves = core.auto_moves(P, piles, game.busts, game.holds, D, v,
                            depth=int(getattr(P, "depth", 3)), beam=int(getattr(P, "beam", 2)))
    base = P.rho * sum(D)
    return [(s + base, i, h) for s, i, h in moves]
