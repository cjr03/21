"""Core tables + vectorised (numpy) simulator for the 21 solitaire game.

Pile states are compacted to the reachable set: (n_cards, hard_total, has_ace) with n<=4,
hard<=21 and best!=21 (those clear). State 0 is the empty pile.
Card values are coded 1..10 (index v-1 = 0..9).
"""
import numpy as np

MAXC = 5


def _resolve(n, h, a):
    """Return (new_state_tuple_or_None, reward, bust)."""
    if h > 21:
        return None, 0, 1
    best = h + 10 if a and h + 10 <= 21 else h
    if best == 21:
        return None, (1000 if n == MAXC else 400), 0
    if n == MAXC:
        return None, 800, 0
    return (n, h, a), 0, 0


def _build():
    states = [(0, 0, 0)]
    idx = {(0, 0, 0): 0}
    frontier = [(0, 0, 0)]
    trans = {}
    while frontier:
        nxt = []
        for st in frontier:
            n, h, a = st
            for v in range(1, 11):
                ns, r, b = _resolve(n + 1, h + v, a or v == 1)
                if ns is None:
                    ns = (0, 0, 0)
                if ns not in idx:
                    idx[ns] = len(states)
                    states.append(ns)
                    nxt.append(ns)
                trans[(st, v)] = (ns, r, b)
        frontier = nxt
    S = len(states)
    TN = np.zeros((S, 11), dtype=np.int16)
    TR = np.zeros((S, 11), dtype=np.float64)
    TB = np.zeros((S, 11), dtype=np.int8)
    for st in states:
        for v in range(1, 11):
            ns, r, b = trans[(st, v)]
            TN[idx[st], v] = idx[ns]
            TR[idx[st], v] = r
            TB[idx[st], v] = b
    return states, idx, TN, TR, TB


STATES, STATE_IDX, TN, TR, TB = _build()
NS = len(STATES)


def pile_state(pile):
    """Map a list of card values to the compact state index."""
    if not pile:
        return 0
    h = sum(pile)
    return STATE_IDX[(len(pile), h, 1 if 1 in pile else 0)]
