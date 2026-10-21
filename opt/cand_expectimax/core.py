"""Core tables + expectimax search for the 21 solitaire game.

Pile state = (hard total, has ace, card count) mapped to a small int id (0 = empty pile).
A position (after a card has been placed, i.e. an "afterstate") is
    (sorted tuple of 4 pile ids, busts, holds, remaining deck counts by value 1..10).
"""
import os

import numpy as np

MAXC = 5
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- pile tables
STATES = [(0, False, 0)]
SID = {(0, False, 0): 0}
TRANS = []  # TRANS[s][v] = (new_state, points, bust)


def _step(st, v):
    hard, ace, n = st
    h2, a2, n2 = hard + v, ace or v == 1, n + 1
    if h2 > 21:
        return None, 0, True
    best = h2 + 10 if a2 and h2 + 10 <= 21 else h2
    if best == 21:
        return None, (1000 if n2 == MAXC else 400), False
    if n2 == MAXC:
        return None, 800, False
    return (h2, a2, n2), 0, False


def _build():
    i = 0
    while i < len(STATES):
        st = STATES[i]
        row = [None]
        for v in range(1, 11):
            ns, pts, bust = _step(st, v)
            if ns is None:
                row.append((0, pts, bust))
            else:
                if ns not in SID:
                    SID[ns] = len(STATES)
                    STATES.append(ns)
                row.append((SID[ns], pts, bust))
        TRANS.append(row)
        i += 1


_build()
NS = len(STATES)
SINGLE = [None] + [TRANS[0][v][0] for v in range(1, 11)]  # pile holding just card v


def pile_id(pile):
    if not pile:
        return 0
    return SID[(sum(pile), 1 in pile, len(pile))]


# deck code: counts of value 1..9 in base 5, value 10 count on top
MULT = [0] + [5 ** (v - 1) for v in range(1, 10)] + [5 ** 9]


def deck_code(deck):
    return sum(deck[v] * MULT[v] for v in range(1, 11))


# ----------------------------------------------------------------------------- leaf evaluation
class Leaf:
    """V(afterstate) = G[n][bh] + U[n].p + sum_piles (F[n][bh][pile] + W[n][pile].p)
    (n = cards left, bh = busts*3+holds, p = remaining-deck value fractions)."""

    def __init__(self, path=None):
        path = path or os.path.join(HERE, "weights.npz")
        if os.path.exists(path):
            w = np.load(path)
            self.F, self.G = w["F"], w["G"]
            self.W = w["W"] if "W" in w else np.zeros((52, NS, 10))
            self.U = w["U"] if "U" in w else np.zeros((52, 10))
        else:
            self.F, self.G = np.zeros((52, 9, NS)), np.zeros((52, 9))
            self.W, self.U = np.zeros((52, NS, 10)), np.zeros((52, 10))
        self.cache = {}

    def tables(self, deck, code, n):
        t = self.cache.get(code)
        if t is None:
            p = np.array(deck[1:11], dtype=float) / n
            rows = (self.F[n] + (self.W[n] @ p)[None, :]).tolist()
            g = (self.G[n] + self.U[n] @ p).tolist()
            t = (rows, g)
            if len(self.cache) > 200000:
                self.cache.clear()
            self.cache[code] = t
        return t

    def __call__(self, piles, busts, holds, n, deck, code):
        if n <= 0:
            return 0.0
        rows, g = self.tables(deck, code, n)
        bh = busts * 3 + holds
        r = rows[bh]
        return g[bh] + r[piles[0]] + r[piles[1]] + r[piles[2]] + r[piles[3]]


# ----------------------------------------------------------------------------- search
class Searcher:
    def __init__(self, leaf):
        self.leaf = leaf
        self.memo = {}
        self.nodes = 0

    def moves(self, piles, holds, v):
        """Distinct successor configurations for placing value v: list of (newpiles, pts, bust, used_hold)."""
        out = []
        seen = set()
        for i in range(4):
            s = piles[i]
            if s in seen:
                continue
            seen.add(s)
            ns, pts, bust = TRANS[s][v]
            np_ = list(piles)
            np_[i] = ns
            np_.sort()
            out.append((tuple(np_), pts, bust, 0))
        if holds > 0:
            single = SINGLE[v]
            seen = set()
            for i in range(4):
                s = piles[i]
                if s == 0 or s in seen:
                    continue
                seen.add(s)
                np_ = list(piles)
                np_[i] = single
                np_.sort()
                out.append((tuple(np_), 0, False, 1))
        return out

    def chance1(self, piles, busts, holds, deck, code, n):
        """Depth-1 chance node specialised for the additive leaf (no canonicalisation needed)."""
        self.nodes += 1
        tables = self.leaf.tables
        p0, p1, p2, p3 = piles
        total = 0.0
        bh = busts * 3 + holds
        n1 = n - 1
        for v in range(1, 11):
            c = deck[v]
            if c == 0:
                continue
            deck[v] = c - 1
            rows, g = tables(deck, code - MULT[v], n1)
            deck[v] = c
            r0 = rows[bh]
            base0 = g[bh] + r0[p0] + r0[p1] + r0[p2] + r0[p3]
            if busts < 2:
                rb = rows[bh + 3]
                baseb = g[bh + 3] + rb[p0] + rb[p1] + rb[p2] + rb[p3] + rb[0]
            if holds:
                rh = rows[bh - 1]
                baseh = g[bh - 1] + rh[p0] + rh[p1] + rh[p2] + rh[p3] + rh[SINGLE[v]]
            best = -1e18
            last = -1
            for s in piles:
                if s == last:
                    continue
                last = s
                ns, pts, bust = TRANS[s][v]
                if bust:
                    val = baseb - rb[s] if busts < 2 else 0.0
                else:
                    val = pts + base0 - r0[s] + r0[ns]
                if val > best:
                    best = val
                if holds and s:
                    val = baseh - rh[s]
                    if val > best:
                        best = val
            total += c * best
        return total / n

    def chance(self, piles, busts, holds, deck, code, n, depth):
        """Expected future points from an afterstate (piles, busts, holds) with deck counts `deck`."""
        if n == 0:
            return 0.0
        if depth == 0:
            return self.leaf(piles, busts, holds, n, deck, code)
        key = (piles, busts, holds, code, depth)
        val = self.memo.get(key)
        if val is not None:
            return val
        if depth == 1 and n > 1:
            val = self.chance1(piles, busts, holds, deck, code, n)
            self.memo[key] = val
            return val
        self.nodes += 1
        total = 0.0
        n1 = n - 1
        d1 = depth - 1
        for v in range(1, 11):
            c = deck[v]
            if c == 0:
                continue
            deck[v] = c - 1
            code1 = code - MULT[v]
            best = -1e18
            for np_, pts, bust, uh in self.moves(piles, holds, v):
                b2 = busts + bust
                if b2 >= 3:
                    val2 = 0.0
                else:
                    val2 = pts + self.chance(np_, b2, holds - uh, deck, code1, n1, d1)
                if val2 > best:
                    best = val2
            deck[v] = c
            total += c * best
        val = total / n
        self.memo[key] = val
        return val
