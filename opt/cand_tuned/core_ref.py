"""Core engine for the tuned 21-solitaire strategy (pure Python + numpy).

Pile state = (hard total, has-ace, card count); only 98 non-terminal states exist, index 0 = empty.
Evaluation of a position = sum of per-pile values + a global "escape" term:

  * per-pile value PV[s]: relative value of a pile from a single-pile Markov decision process that
    is re-solved every move against the exact remaining deck.  Each future card is "offered" to the
    pile, which may take it (paying a per-card cost g, collecting points / paying a bust cost) or
    pass it (paying a pass cost c, because another pile must absorb it).  Finite horizon = cards
    left in the deck, so piles that cannot finish in time lose their value automatically.
  * global term G(b, H, N) = -f(N) * PHI[escapes] + f(N) * ETA * holds,  escapes = (2-b) + holds.
    A 3rd bust is worth -RHO * N (all future points lost).

Moves are scored by a 1-card expectimax: immediate reward + E_next[max_reply(reward + eval)].
"""

import numpy as np

MAXC = 5
INF = float("inf")

# ----------------------------------------------------------------------------- pile tables


def _build():
    states = [(0, 0, 0)]
    index = {(0, 0, 0): 0}
    nxt, rew, bust = [], [], []
    i = 0
    while i < len(states):
        h, a, n = states[i]
        row_n, row_r, row_b = [0] * 11, [0] * 11, [0] * 11
        for v in range(1, 11):
            hh, aa, nn = h + v, 1 if (a or v == 1) else 0, n + 1
            if hh > 21:
                row_b[v] = 1
                continue
            best = hh + 10 if aa and hh + 10 <= 21 else hh
            if best == 21:
                row_r[v] = 1000 if nn == MAXC else 400
                continue
            if nn == MAXC:
                row_r[v] = 800
                continue
            s = (hh, aa, nn)
            if s not in index:
                index[s] = len(states)
                states.append(s)
            row_n[v] = index[s]
        nxt.append(row_n)
        rew.append(row_r)
        bust.append(row_b)
        i += 1
    return states, index, nxt, rew, bust


STATES, INDEX, NXT, REW, BUST = _build()
NS = len(STATES)
NXT_NP = np.array([r[1:] for r in NXT], dtype=np.int64)
REW_NP = np.array([r[1:] for r in REW], dtype=np.float64)
BUST_NP = np.array([r[1:] for r in BUST], dtype=np.float64)
COUNT = [s[2] for s in STATES]
HARD = [s[0] for s in STATES]
SOFT = [s[1] for s in STATES]
BEST = [h + 10 if a and h + 10 <= 21 else h for h, a, n in STATES]
COUNT_NP = np.array(COUNT)
ROOM = [min(21 - h, 10) for h in HARD]


def pile_index(pile):
    """Map a list of card values to its state index."""
    if not pile:
        return 0
    return INDEX[(sum(pile), 1 if 1 in pile else 0, len(pile))]


# ----------------------------------------------------------------------------- parameters

DEFAULT = {
    "g": 60.0,       # per-card cost inside the pile MDP
    "c": 25.0,       # cost of passing a card (another pile has to take it)
    "B0": 150.0,     # base bust cost inside pile MDP (cards wasted etc.)
    "kB": 1.0,       # weight of the escape-loss cost on the pile-MDP bust cost
    "phi3": 150.0,   # loss when escapes drop 4->3 (scaled by f(N))
    "phi2": 450.0,
    "phi1": 1000.0,
    "phi0": 2000.0,
    "eta": 0.0,      # extra value of a hold beyond being an escape
    "rho": 110.0,    # expected points per remaining card (cost of game over)
    "alpha": 1.0,    # scale on pile values
    "gamma": 1.0,    # lookahead weight
    "fpow": 1.0,     # f(N) = (N/51)^fpow
    "K": 40,         # max DP horizon
    "n1": 0.0, "n2": 0.0, "n3": 0.0, "n4": 0.0,   # additive per-count pile corrections
    "soft": 0.0,     # bonus for a soft (usable-ace) pile
    "e0": 0.0,       # bonus per empty pile (flexibility), scaled by f(N)
    "hc": -1.0,      # hold floor inside pile MDP (pile value >= -hc when holds remain); <0 = off
    "omega": 0.0,    # weight of P(card after the lookahead has no safe pile) * escape cost
}

PARAM_ORDER = ["g", "c", "B0", "kB", "phi3", "phi2", "phi1", "phi0", "eta", "rho",
               "alpha", "gamma", "fpow", "n1", "n2", "n3", "n4", "soft", "omega"]


class Params:
    def __init__(self, d=None):
        p = dict(DEFAULT)
        if d:
            p.update(d)
        self.__dict__.update(p)
        self.phi = [p["phi0"], p["phi1"], p["phi2"], p["phi3"], 0.0]
        self.corr = np.array([0.0, p["n1"], p["n2"], p["n3"], p["n4"]])[COUNT_NP] + \
            p["soft"] * np.array([1.0 if (s[1] and s[0] + 10 <= 21) else 0.0 for s in STATES])
        self.corr[0] = 0.0


# ----------------------------------------------------------------------------- evaluation pieces


def gval(P, b, H, N):
    """Global value of busts/holds with N cards left (b < 3)."""
    if N <= 0:
        return 0.0
    f = (N / 51.0) ** P.fpow
    esc = 2 - b + H
    if esc > 4:
        esc = 4
    return -f * P.phi[esc] + f * P.eta * H


def death(P, N):
    return -P.rho * N


def pile_tables(P, D, N, b, H, levels=4):
    """Relative values of all pile states for the remaining deck D (list idx 1..10).
    Returns {n: PV list} for n = N, N-1, ..., N-levels (n >= 0): the value iteration's k-th iterate
    is the finite-horizon value with k cards left, so the shallower search levels get their own
    horizon for free.  n = 0 -> all zeros (unfinished piles are worthless at the end)."""
    out = {}
    if N <= 0:
        out[0] = [0.0] * NS
        return out
    p = np.array(D[1:], dtype=np.float64) / N
    # bust cost inside the MDP = base + marginal loss of the global term
    if b >= 2:
        gb = gval(P, b, H, N) - death(P, N)
    else:
        gb = gval(P, b, H, N) - gval(P, b + 1, H, N)
    B = P.B0 + P.kB * gb
    R0 = REW_NP - B * BUST_NP - P.g
    W = np.zeros(NS)
    k = min(N, int(P.K))
    c = P.c
    hfloor = P.hc if (H > 0 and P.hc >= 0) else None
    lo = max(0, N - levels)
    for j in range(1, k + 1):
        A = R0 + W[NXT_NP]
        W = (np.maximum(A, (W - c)[:, None]) * p).sum(1)
        if hfloor is not None:
            W = np.maximum(W, W[0] - hfloor)
        n = N - (k - j)
        if n >= lo and j >= 1:
            PV = P.alpha * (W - W[0]) + P.corr
            PV[0] = P.e0 * (n / 51.0) ** P.fpow
            out[n] = PV.tolist()
    for n in range(lo, N + 1):
        if n not in out:
            out[n] = [0.0] * NS if n == 0 else out[min(m for m in out if m > n)] if any(m > n for m in out) else out[N]
    out[0] = [0.0] * NS
    return out


def pile_values(P, D, N, b, H):
    return pile_tables(P, D, N, b, H, 0)[max(N, 0)]


class Ctx:
    """Per-decision context: pile values, tail probabilities and cached global terms."""

    def __init__(self, P, D, N, b, H):
        self.P = P
        self.PVT = pile_tables(P, D, N, b, H)
        self.PV = self.PVT[N]
        tail = [0.0] * 11
        if N > 0:
            acc = 0
            for r in range(10, -1, -1):
                tail[r] = acc / N
                if r >= 1:
                    acc += D[r]
        self.tail = tail
        self.G = {}

    def g(self, bb, hh, n):
        """(global value, escape cost weighted by omega) for b<3."""
        key = (bb, hh, n)
        r = self.G.get(key)
        if r is None:
            P = self.P
            gv = gval(P, bb, hh, n)
            if P.omega:
                if hh > 0:
                    esc = gv - gval(P, bb, hh - 1, n)
                elif bb < 2:
                    esc = gv - gval(P, bb + 1, hh, n)
                else:
                    esc = gv - death(P, n)
                r = (gv, P.omega * esc)
            else:
                r = (gv, 0.0)
            self.G[key] = r
        return r

    def pv(self, n):
        t = self.PVT.get(n)
        if t is None:
            t = self.PVT[min(self.PVT, key=lambda m: abs(m - n))]
        return t


def leaf(ctx, new, b1, H1, vals, N):
    """E over the next card (counts vals, total N) of the best placement scored by the static eval.
    `new` = 4 pile states, b1 < 3, N > 0."""
    tail = ctx.tail
    N2 = N - 1
    dN = -ctx.P.rho * N2
    if N2 == 0:
        # last card of the deck: only immediate points matter
        best = 0.0
        exp = 0.0
        for w, cnt in vals:
            best = 0.0
            for s in new:
                if REW[s][w] > best:
                    best = REW[s][w]
            if H1 > 0 and REW[0][w] > best:
                best = REW[0][w]
            exp += cnt * best
        return ctx.P.gamma * exp / N
    PV = ctx.pv(N2)
    base = PV[new[0]] + PV[new[1]] + PV[new[2]] + PV[new[3]]
    r0, r1_, r2_, r3_ = ROOM[new[0]], ROOM[new[1]], ROOM[new[2]], ROOM[new[3]]
    distinct = {}
    distinct[new[0]] = max(r1_, r2_, r3_)
    if new[1] not in distinct:
        distinct[new[1]] = max(r0, r2_, r3_)
    if new[2] not in distinct:
        distinct[new[2]] = max(r0, r1_, r3_)
    if new[3] not in distinct:
        distinct[new[3]] = max(r0, r1_, r2_)
    distinct = list(distinct.items())
    gb, rb = ctx.g(b1, H1, N2)
    if b1 + 1 < 3:
        gbb = ctx.g(b1 + 1, H1, N2)[0]
    else:
        gbb = None
    if H1 > 0:
        gh, rh = ctx.g(b1, H1 - 1, N2)
    exp = 0.0
    for w, cnt in vals:
        best = -INF
        for s, mo in distinct:
            if BUST[s][w]:
                x = dN if gbb is None else base - PV[s] + PV[0] + gbb
            else:
                t = NXT[s][w]
                rt = ROOM[t]
                x = REW[s][w] + base - PV[s] + PV[t] + gb - rb * tail[rt if rt > mo else mo]
            if x > best:
                best = x
            if H1 > 0 and s:
                t = NXT[0][w]
                rt = ROOM[t]
                x = REW[0][w] + base - PV[s] + PV[t] + gh - rh * tail[rt if rt > mo else mo]
                if x > best:
                    best = x
        exp += cnt * best
    return ctx.P.gamma * exp / N


def static_eval(ctx, new, b, H, N):
    if b >= 3:
        return -ctx.P.rho * N
    if N <= 0:
        return 0.0
    PV = ctx.pv(N)
    rooms = [ROOM[s] for s in new]
    gv, rk = ctx.g(b, H, N)
    return PV[new[0]] + PV[new[1]] + PV[new[2]] + PV[new[3]] + gv - rk * ctx.tail[max(rooms)]


def successors(piles, b, H, w):
    """Yield (reward, new_piles, b, H, idx, use_hold) for card value w; duplicates by state skipped."""
    seen = set()
    for idx in range(4):
        s = piles[idx]
        for use_hold in ((False, True) if (H > 0 and s) else (False,)):
            s0 = 0 if use_hold else s
            new = list(piles)
            new[idx] = NXT[s0][w]
            yield REW[s0][w], new, b + BUST[s0][w], (H - 1 if use_hold else H), idx, use_hold


def expand(ctx, piles, b, H, D, N, depth, cache, beam=0):
    """Expected value of the next `depth` cards (expectimax) followed by the static eval/leaf.
    beam > 0: at inner max nodes only the `beam` best replies by (reward + static eval) are searched
    deeper (the rest are dropped)."""
    if b >= 3:
        return -ctx.P.rho * N
    if N <= 0:
        return 0.0
    vals = [(w, D[w]) for w in range(1, 11) if D[w] > 0]
    if depth <= 1:
        return leaf(ctx, piles, b, H, vals, N)
    key = (tuple(sorted(piles)), b, H, tuple(D), depth)
    r = cache.get(key)
    if r is not None:
        return r
    exp = 0.0
    for w, cnt in vals:
        D[w] -= 1
        best = -INF
        seen = set()
        kids = []
        for rew, new, b2, H2, _, _ in successors(piles, b, H, w):
            k2 = (tuple(sorted(new)), b2, H2)
            if k2 in seen:
                continue
            seen.add(k2)
            kids.append((rew, new, b2, H2))
        if beam and len(kids) > beam:
            kids = sorted(kids, key=lambda k: -(k[0] + static_eval(ctx, k[1], k[2], k[3], N - 1)))[:beam]
        for rew, new, b2, H2 in kids:
            x = rew + expand(ctx, new, b2, H2, D, N - 1, depth - 1, cache, beam)
            if x > best:
                best = x
        D[w] += 1
        exp += cnt * best
    r = cache[key] = ctx.P.gamma * exp / N
    return r


def score_moves(P, piles, b, H, D, v, depth=1, beam=0):
    """piles: 4 state indices; D: remaining-deck counts by value AFTER removing current card v.
    Returns list of (score, idx, use_hold) for all legal moves, best first.  Score = immediate
    points + expected future value (relative to a pile-less position) from a `depth`-card
    expectimax with the static evaluation at the horizon."""
    N = sum(D)
    ctx = Ctx(P, D, N, b, H)
    D = list(D)
    out = []
    cache = {}
    ecache = {}
    for r1, new, b1, H1, idx, use_hold in successors(piles, b, H, v):
        key = (tuple(sorted(new)), b1, H1)
        val = cache.get(key)
        if val is None:
            val = cache[key] = expand(ctx, new, b1, H1, D, N, depth, ecache, beam)
        out.append((r1 + val, idx, use_hold))
    out.sort(key=lambda m: -m[0])
    return out


def auto_moves(P, piles, b, H, D, v, depth=3, beam=2):
    """Search schedule used by the live strategy: exact expectimax over the last <=4 cards,
    depth 4 (beam 2) for 5-7 cards left, otherwise `depth` with `beam`."""
    N = sum(D)
    if N <= 4:
        return score_moves(P, piles, b, H, D, v, depth=max(1, N), beam=0)
    if N <= 7 and depth >= 3:
        return score_moves(P, piles, b, H, D, v, depth=4, beam=2)
    return score_moves(P, piles, b, H, D, v, depth=depth, beam=beam)
