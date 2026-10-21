"""Vectorised play-out of many games / rollouts at once with a table-driven greedy policy.

Policy: choose the move maximising  r + Vtot(afterstate), where
  Vtot = sum_j V[ph, s_j] + W[ph, busts] + U[ph, holds] + G[ph, g]   (0 if the game is over)
  g    = (min(busts,2)*2 + (holds>0)) * 11 + room,  room = min(10, max_j (21 - hard_j))
         (room = largest card value that still fits on some pile -> congestion / bust risk)
and ph = phase bucket of L = number of cards still in the deck after the current card.
"""
import numpy as np

from core import NS, STATES, TN, TR, TB

TNf = TN.astype(np.int64).ravel()          # index s*11 + v
TRf = TR.ravel()
TBf = TB.astype(np.int64).ravel()
ROOM = np.array([min(10, 21 - st[1]) for st in STATES], dtype=np.int64)
NG = 66

# phase buckets by cards left after the current card (L). bucket 0 is L==0 (terminal).
EDGES = [0, 1, 4, 8, 13, 20, 28, 37, 52]   # bucket k covers EDGES[k] <= L < EDGES[k+1]
NB = len(EDGES) - 1
PHASE = np.zeros(53, dtype=np.int64)
for k in range(NB):
    PHASE[EDGES[k]:EDGES[k + 1]] = k


class Params:
    def __init__(self, V=None, W=None, U=None, G=None):
        self.V = np.zeros((NB, NS)) if V is None else V
        self.W = np.zeros((NB, 3)) if W is None else W   # busts 0,1,2
        self.U = np.zeros((NB, 3)) if U is None else U   # holds 0,1,2
        self.G = np.zeros((NB, NG)) if G is None else G
        for a in (self.V, self.W, self.U, self.G):
            a[0] = 0

    def copy(self):
        return Params(self.V.copy(), self.W.copy(), self.U.copy(), self.G.copy())

    def save(self, path):
        np.savez(path, V=self.V, W=self.W, U=self.U, G=self.G)

    @staticmethod
    def load(path):
        d = np.load(path)
        G = d["G"].copy() if "G" in d.files else None
        return Params(d["V"].copy(), d["W"].copy(), d["U"].copy(), G)


def gindex(P, b, h):
    room = ROOM[P].max(1)
    return (np.minimum(b, 2) * 2 + (h > 0)) * 11 + room


def vtot(params, L, P, b, h):
    """Afterstate value for arrays P[N,4], b[N], h[N] with L cards left (scalar)."""
    ph = PHASE[L]
    v = (params.V[ph][P].sum(1) + params.W[ph][np.minimum(b, 2)] + params.U[ph][h]
         + params.G[ph][gindex(P, b, h)])
    return np.where(b >= 3, 0.0, v)


def play(params, P, b, h, cards, horizon=None, eps=0.0, rng=None, record=False):
    """Play out cards[N, T] (values 1..10) from states P[N,4], busts b[N], holds h[N].

    Cards beyond the horizon are not played; instead the value estimate is added.
    Returns total reward[N] (plus value estimate if truncated). If record, also returns the
    per-step afterstates for regression.
    """
    P = P.astype(np.int64).copy()
    b = b.astype(np.int64).copy()
    h = h.astype(np.int64).copy()
    N, T = cards.shape
    steps = T if horizon is None else min(T, horizon)
    total = np.zeros(N)
    alive = b < 3
    ar = np.arange(N)
    V, W, U, G = params.V, params.W, params.U, params.G
    useG = bool(np.any(G))
    recs = []
    for t in range(steps):
        L = T - t - 1
        ph = PHASE[L]
        Vp, Wp, Up, Gp = V[ph], W[ph], U[ph], G[ph]
        v = cards[:, t].astype(np.int64)
        cur = Vp[P]                                   # N,4
        bc = np.minimum(b, 2)
        base = cur.sum(1) + Wp[bc] + Up[h]
        fi = P * 11 + v[:, None]
        ns = TNf[fi]
        r = TRf[fi]
        bu = TBf[fi]
        Wnext = np.append(Wp, 0.0)
        dW = Wnext[np.minimum(b + 1, 3)] - Wp[bc]
        qp = r + base[:, None] - cur + Vp[ns] + bu * dW[:, None]
        ns0 = TNf[v]
        r0 = TRf[v]
        Uprev = Up[np.maximum(h - 1, 0)]
        qh = (r0 + Vp[ns0] + Uprev - Up[h] + base)[:, None] - cur
        if useG:
            rp = ROOM[P]
            srt = np.sort(rp, 1)
            m1, m2 = srt[:, 3], srt[:, 2]
            am = rp.argmax(1)
            omax = np.where(np.arange(4)[None, :] == am[:, None], m2[:, None], m1[:, None])
            room_p = np.maximum(omax, ROOM[ns])
            gp = (np.minimum(b[:, None] + bu, 2) * 2 + (h[:, None] > 0)) * 11 + room_p
            qp = qp + Gp[gp]
            room_h = np.maximum(omax, ROOM[ns0][:, None])
            gh = (bc[:, None] * 2 + (h[:, None] > 1)) * 11 + room_h
            qh = qh + Gp[gh]
        dead_bust = (bu == 1) & (b[:, None] >= 2)
        qp = np.where(dead_bust, r, qp)
        can_hold = (P != 0) & (h[:, None] > 0)
        qh = np.where(can_hold, qh - 1e-6, -1e18)
        q = np.concatenate([qp, qh], axis=1)
        a = q.argmax(1)
        if eps > 0:
            explore = rng.random(N) < eps
            if explore.any():
                legal = np.concatenate([np.ones((N, 4), bool), can_hold], axis=1)
                rnd = rng.random((N, 8)) * legal
                a = np.where(explore, rnd.argmax(1), a)
        j = a & 3
        hold = (a >= 4).astype(np.int64)
        src = np.where(hold == 1, 0, P[ar, j])
        f = src * 11 + v
        rew = TRf[f] * alive
        bust = TBf[f] * alive
        P[ar, j] = TNf[f]
        total += rew
        b += bust
        h -= hold * alive
        alive = b < 3
        if record:
            recs.append((L, P.copy(), b.copy(), h.copy(), rew.copy(), alive.copy()))
    if steps < T:
        total += vtot(params, T - steps, P, b, h) * alive
    if record:
        return total, recs
    return total


def deal(n, rng, remaining=None):
    """n shuffled decks of the remaining values (default: full deck). Returns int [n, L]."""
    if remaining is None:
        remaining = [4] * 9 + [16]
    vals = np.repeat(np.arange(1, 11), remaining)
    keys = rng.random((n, len(vals)))
    order = keys.argsort(1)
    return vals[order]
