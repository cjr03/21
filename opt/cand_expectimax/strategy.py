"""Expectimax strategy for the 21 solitaire calculator.

rank_moves(game, rank) -> [(expected_points, idx, use_hold), ...] best first.
The score of a move = immediate points + expected future points (exact expectimax over the known
remaining deck for `depth` more cards, then a fitted value function at the leaves).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import core  # noqa: E402
import twentyone as t  # noqa: E402

LEAF = core.Leaf(os.path.join(HERE, "weights.npz"))
BUDGET = float(os.environ.get("EXPECTI_BUDGET", "0.35"))  # seconds of search per move (soft)
MIN_DEPTH = int(os.environ.get("EXPECTI_MIN_DEPTH", "2"))
MAX_DEPTH = int(os.environ.get("EXPECTI_MAX_DEPTH", "60"))


def _root_moves(game, v):
    pids = [core.pile_id(p) for p in game.piles]
    legal = [(i, False) for i in range(4)]
    if game.holds > 0:
        legal += [(i, True) for i in range(4) if game.piles[i]]
    out = []
    for i, uh in legal:
        np_ = list(pids)
        if uh:
            np_[i] = core.SINGLE[v]
            pts, bust = 0, False
        else:
            np_[i], pts, bust = core.TRANS[pids[i]][v]
        np_.sort()
        out.append(((tuple(np_), game.busts + bust, game.holds - uh), pts, i, uh))
    return out


def evaluate(game, rank, budget=None, min_depth=None, max_depth=None, stats=None):
    budget = BUDGET if budget is None else budget
    min_depth = MIN_DEPTH if min_depth is None else min_depth
    max_depth = MAX_DEPTH if max_depth is None else max_depth
    v = t.RANK_VALUE[rank]
    rem = game.remaining_values()
    deck = [0] * 11
    for val, c in rem.items():
        deck[val] = c
    deck[v] -= 1
    n = sum(deck)
    code = core.deck_code(deck)
    moves = _root_moves(game, v)
    succ = {}
    for key, pts, i, uh in moves:
        succ.setdefault(key, pts)
    S = core.Searcher(LEAF)
    t0 = time.perf_counter()
    vals = {}
    depth = 0
    last = None
    while True:
        depth += 1
        ts = time.perf_counter()
        cur = {}
        for key, pts in succ.items():
            piles, b, h = key
            cur[key] = 0.0 if b >= 3 else pts + S.chance(piles, b, h, deck, code, n, depth)
        vals = cur
        tl = time.perf_counter() - ts
        if depth >= n or depth >= max_depth:  # exact to the end of the deck (or cap)
            break
        if depth >= min_depth:
            ratio = tl / last if last and last > 1e-4 else 30.0
            ratio = max(ratio, 3.0)
            if time.perf_counter() - t0 + tl * ratio > budget:
                break
        last = tl
    if stats is not None:
        stats.append((n, depth, time.perf_counter() - t0, S.nodes))
    return [(vals[key], i, uh) for key, pts, i, uh in moves]


def rank_moves(game, rank):
    ranked = evaluate(game, rank)
    # deterministic tie-break: prefer non-hold, then lower index
    ranked.sort(key=lambda m: (-round(m[0], 6), m[2], m[1]))
    return ranked
