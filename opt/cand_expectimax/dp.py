"""Backward fitted dynamic programming for the leaf value tables.

1. Sample realistic afterstates by self-play with the current search policy (+ optional exploration).
2. For n = 1..51 (cards left), fit V_n(afterstate) = G[n][bh] + sum_piles F[n][bh][pile] by ridge
   regression to the exact one-card Bellman backup
       y = sum_v p(v) * max_move [ pts + V_{n-1}(next afterstate) ]      (V_0 = 0, 3rd bust = 0)
   computed with the already-fitted V_{n-1}.  Because n strictly decreases this is one backward pass.
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402
import train  # noqa: E402

NS = core.NS
# vectorised transition tables
T_NS = np.zeros((NS, 11), dtype=np.int64)
T_PTS = np.zeros((NS, 11))
T_BUST = np.zeros((NS, 11), dtype=np.int64)
for s in range(NS):
    for v in range(1, 11):
        a, b, c = core.TRANS[s][v]
        T_NS[s, v], T_PTS[s, v], T_BUST[s, v] = a, b, int(c)
SINGLE = np.array([0] + [core.SINGLE[v] for v in range(1, 11)])


def tables_for(n, F, G, W, U, D):
    """Per-sample effective tables at n cards left: Fe (N,9,NS), Ge (N,9)."""
    p = D[:, 1:] / max(n, 1)                      # (N,10)
    pw = p @ W[n].T                               # (N,NS)
    Fe = F[n][None, :, :] + pw[:, None, :]
    Ge = G[n][None, :] + (p @ U[n])[:, None]
    return Fe, Ge


def backup(P, B, H, D, n, F, G, W, U):
    """Vectorised Bellman backup from n cards left to n-1 using the fitted tables at n-1."""
    N = len(P)
    y = np.zeros(N)
    ar = np.arange(N)
    for v in range(1, 11):
        cnt = D[:, v]
        if not cnt.any():
            continue
        D1 = D.copy()
        D1[:, v] -= 1
        if n - 1 > 0:
            Fe, Ge = tables_for(n - 1, F, G, W, U, D1)
        else:
            Fe, Ge = np.zeros((N, 9, NS)), np.zeros((N, 9))
        best = np.full(N, -1e18)
        for i in range(4):
            s = P[:, i]
            ns = T_NS[s, v]
            pts = T_PTS[s, v]
            b2 = B + T_BUST[s, v]
            alive = b2 < 3
            bh = np.minimum(b2, 2) * 3 + H
            Fb = Fe[ar, bh]                       # (N,NS)
            tot = Ge[ar, bh] + Fb[ar[:, None], P].sum(1) - Fb[ar, s] + Fb[ar, ns]
            val = np.where(alive, pts + tot, 0.0)
            best = np.maximum(best, val)
            ok = (H > 0) & (s != 0)
            if ok.any():
                bh2 = B * 3 + np.maximum(H - 1, 0)
                Fb2 = Fe[ar, bh2]
                tot2 = Ge[ar, bh2] + Fb2[ar[:, None], P].sum(1) - Fb2[ar, s] + Fb2[ar, SINGLE[v]]
                best = np.where(ok, np.maximum(best, tot2), best)
        y += cnt / n * best
    return y


NPAR = NS + 9 * NS + NS * 10 + 9 + 10


def design(P, B, H, D, n):
    N = len(P)
    bh = B * 3 + H
    p = D[:, 1:] / n
    X = np.zeros((N, NPAR))
    ar = np.arange(N)
    o_dev, o_w, o_g, o_u = NS, 10 * NS, 20 * NS, 20 * NS + 9
    for i in range(4):
        s = P[:, i]
        np.add.at(X, (ar, s), 1.0)
        np.add.at(X, (ar, o_dev + bh * NS + s), 1.0)
        for v in range(10):
            np.add.at(X, (ar, o_w + s * 10 + v), p[:, v])
    X[ar, o_g + bh] = 1.0
    X[:, o_u:o_u + 10] = p
    return X


def fit_n(P, B, H, D, n, y, lam_base, lam_dev, lam_g, lam_w=30.0):
    lam = np.concatenate([np.full(NS, lam_base), np.full(9 * NS, lam_dev), np.full(10 * NS, lam_w),
                          np.full(9, lam_g), np.full(10, lam_w)])
    A = np.diag(lam)
    b = np.zeros(NPAR)
    for c in range(0, len(P), 3000):
        X = design(P[c:c + 3000], B[c:c + 3000], H[c:c + 3000], D[c:c + 3000], n)
        A += X.T @ X
        b += X.T @ y[c:c + 3000]
    w = np.linalg.solve(A, b)
    se = 0.0
    for c in range(0, len(P), 3000):
        X = design(P[c:c + 3000], B[c:c + 3000], H[c:c + 3000], D[c:c + 3000], n)
        se += ((X @ w - y[c:c + 3000]) ** 2).sum()
    base = w[:NS]
    dev = w[NS:10 * NS].reshape(9, NS)
    Wn = w[10 * NS:20 * NS].reshape(NS, 10)
    g = w[20 * NS:20 * NS + 9]
    u = w[20 * NS + 9:]
    return base[None, :] + dev, g, Wn, u, np.sqrt(se / len(P))


def successors(L, rng, k):
    """k random one-step successors (random card from the deck, uniformly random legal move) per state."""
    out = []
    sr = core.Searcher(None)
    for p, b, h, dk in L:
        n = sum(dk)
        for _ in range(k):
            r = rng.integers(0, n)
            v = 1
            acc = dk[1]
            while acc <= r:
                v += 1
                acc += dk[v]
            mv = sr.moves(p, h, v)
            np_, pts, bust, uh = mv[rng.integers(0, len(mv))]
            if b + bust >= 3:
                continue
            d2 = list(dk)
            d2[v] -= 1
            out.append((np_, b + bust, h - uh, tuple(d2)))
    return out


def collect(paths):
    S = []
    for p in paths:
        with open(p, "rb") as f:
            S += pickle.load(f)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--win", default="none.npz")
    ap.add_argument("--wout", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--randbh", type=float, default=0.5, help="fraction of samples with random busts/holds")
    ap.add_argument("--succ", type=int, default=1, help="random successors per state of n+1")
    ap.add_argument("--lam", type=float, nargs=3, default=[1.0, 20.0, 1.0])
    a = ap.parse_args()
    t0 = time.time()
    S = []
    if a.games > 0:
        seeds = list(range(a.seed0, a.seed0 + a.games))
        with Pool(a.workers, initializer=train._init, initargs=(a.win, a.depth, a.eps)) as pool:
            res = pool.map(train.play, seeds, chunksize=20)
        scores = [r[0] for r in res]
        S = [s for r in res for s in r[1]]
        print(f"selfplay depth={a.depth} eps={a.eps} games={a.games} mean={np.mean(scores):.1f} "
              f"se={np.std(scores) / np.sqrt(len(scores)):.1f} samples={len(S)} t={time.time() - t0:.0f}s",
              flush=True)
        if a.data:
            with open(a.data, "wb") as f:
                pickle.dump(S, f)
    S += collect(a.extra)
    rng = np.random.default_rng(1)
    byn = {}
    for p, b, h, n, _t, dk in S:
        byn.setdefault(n, []).append((p, b, h, dk))
    F = np.zeros((52, 9, NS))
    G = np.zeros((52, 9))
    W = np.zeros((52, NS, 10))
    U = np.zeros((52, 10))
    for n in range(1, 52):
        L = byn.get(n, []) + successors(byn.get(n + 1, []), rng, a.succ)
        P = np.array([x[0] for x in L], dtype=np.int64)
        B = np.array([x[1] for x in L], dtype=np.int64)
        H = np.array([x[2] for x in L], dtype=np.int64)
        D = np.array([x[3] for x in L], dtype=np.float64)
        # augment with random busts/holds so every (b,h) cell is covered
        m = int(len(L) * a.randbh)
        if m:
            idx = rng.integers(0, len(L), m)
            P = np.concatenate([P, P[idx]])
            D = np.concatenate([D, D[idx]])
            B = np.concatenate([B, rng.integers(0, 3, m)])
            H = np.concatenate([H, rng.integers(0, 3, m)])
        y = backup(P, B, H, D, n, F, G, W, U)
        F[n], G[n], W[n], U[n], rmse = fit_n(P, B, H, D, n, y, *a.lam)
        if n in (1, 2, 5, 10, 20, 30, 40, 51):
            print(f"n={n:2d} N={len(P)} sd(y)={y.std():.1f} rmse={rmse:.1f} "
                  f"V(empty,b0,h2)={G[n][2] + 4 * F[n][2][0] + (W[n][0] * 4 + U[n]) @ (D[0, 1:] / n):.0f}", flush=True)
    np.savez(a.wout, F=F, G=G, W=W, U=U)
    print(f"saved {a.wout} t={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
