"""Fit the leaf value tables by approximate policy iteration (self-play + Monte Carlo returns).

usage: python3 train.py ITER --games N --depth D --seed0 S [--workers W]
Games use decks random.Random(seed).shuffle for seeds in [seed0, seed0+N) (keep these < 50000 and
away from the evaluation range).
"""
import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
RV = {r: (1 if r == "a" else 10 if r in "jqk" else int(r)) for r in RANKS}

_leaf = None
_depth = 1
_eps = 0.0


def _init(wpath, depth, eps):
    global _leaf, _depth, _eps
    _leaf = core.Leaf(wpath)
    _depth = depth
    _eps = eps


def choose(searcher, piles, busts, holds, deck, n, v, depth, rng=None, eps=0.0):
    """deck/n exclude v. Returns (newpiles, pts, bust, uh, value)."""
    code = core.deck_code(deck)
    best = None
    cands = []
    for np_, pts, bust, uh in searcher.moves(piles, holds, v):
        b2 = busts + bust
        if b2 >= 3:
            val = 0.0
        else:
            val = pts + searcher.chance(np_, b2, holds - uh, deck, code, n, depth)
        cands.append((val, np_, pts, bust, uh))
        if best is None or val > best[0]:
            best = (val, np_, pts, bust, uh)
    if rng is not None and eps > 0 and rng.random() < eps and len(cands) > 1:
        cands.sort(key=lambda c: -c[0])
        best = cands[1]
    return best


def play(seed):
    d = [r for r in RANKS for _ in range(4)]
    rng = random.Random(seed)
    rng.shuffle(d)
    erng = random.Random(seed + 7777777)
    piles = (0, 0, 0, 0)
    busts = holds = 0
    holds = 2
    deck = [0] + [4] * 9 + [16]
    n = 52
    score = 0
    recs = []
    s = core.Searcher(_leaf)
    for r in d:
        v = RV[r]
        deck[v] -= 1
        n -= 1
        s.memo.clear()
        val, np_, pts, bust, uh = choose(s, piles, busts, holds, deck, n, v, _depth, erng, _eps)
        piles, busts, holds = np_, busts + bust, holds - uh
        score += pts
        if busts >= 3:
            break
        if n > 0:
            recs.append((piles, busts, holds, n, score, tuple(deck)))
    out = [(p, b, h, nn, score - sc, dk) for p, b, h, nn, sc, dk in recs]
    return score, out


KNOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 51]


def knot_weights(n):
    """Return (k0, w0, k1, w1) linear interpolation of n between knots."""
    for k in range(len(KNOTS) - 1):
        a, b = KNOTS[k], KNOTS[k + 1]
        if a <= n <= b:
            t = (n - a) / (b - a)
            return k, 1 - t, k + 1, t
    raise ValueError(n)


def fit(samples, lam_base=1.0, lam_dev=30.0, lam_g=1.0, iters=400):
    NS = core.NS
    K = len(KNOTS)
    # parameter layout per knot: base[NS], dev[9*NS], g[9]
    per = NS + 9 * NS + 9
    P = K * per
    N = len(samples)
    idx = np.zeros((N, 18), dtype=np.int64)
    W = np.zeros((N, 18))
    y = np.zeros(N)
    kw = [None] + [knot_weights(n) for n in range(1, 52)]
    for i, (piles, b, h, n, tgt, _dk) in enumerate(samples):
        bh = b * 3 + h
        k0, w0, k1, w1 = kw[n]
        j = 0
        for k, w in ((k0, w0), (k1, w1)):
            off = k * per
            for s in piles:
                idx[i, j] = off + s
                W[i, j] = w
                j += 1
                idx[i, j] = off + NS + bh * NS + s
                W[i, j] = w
                j += 1
            idx[i, j] = off + 10 * NS + bh
            W[i, j] = w
            j += 1
        y[i] = tgt
    lam = np.zeros(P)
    for k in range(K):
        off = k * per
        lam[off:off + NS] = lam_base
        lam[off + NS:off + 10 * NS] = lam_dev
        lam[off + 10 * NS:off + per] = lam_g
    flat = idx.ravel()

    def A(w):
        r = (W * w[idx]).sum(1)
        return np.bincount(flat, weights=(W * r[:, None]).ravel(), minlength=P) + lam * w

    bvec = np.bincount(flat, weights=(W * y[:, None]).ravel(), minlength=P)
    # Jacobi-preconditioned CG
    diag = np.bincount(flat, weights=(W * W).ravel(), minlength=P) + lam
    w = np.zeros(P)
    r = bvec - A(w)
    z = r / diag
    p = z.copy()
    rz = r @ z
    for it in range(iters):
        Ap = A(p)
        alpha = rz / (p @ Ap)
        w += alpha * p
        r -= alpha * Ap
        z = r / diag
        rz2 = r @ z
        if np.sqrt(r @ r) < 1e-6 * np.sqrt(bvec @ bvec):
            break
        p = z + (rz2 / rz) * p
        rz = rz2
    pred = (W * w[idx]).sum(1)
    resid = y - pred
    print(f"fit: N={N} rmse={np.sqrt((resid ** 2).mean()):.1f} sd(y)={y.std():.1f} iters={it}")
    # expand to per-n tables
    F = np.zeros((52, 9, NS))
    G = np.zeros((52, 9))
    for n in range(1, 52):
        k0, w0, k1, w1 = kw[n]
        for k, wt in ((k0, w0), (k1, w1)):
            off = k * per
            base = w[off:off + NS]
            dev = w[off + NS:off + 10 * NS].reshape(9, NS)
            g = w[off + 10 * NS:off + per]
            F[n] += wt * (base[None, :] + dev)
            G[n] += wt * g
    return F, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--win", default=os.path.join(HERE, "weights.npz"))
    ap.add_argument("--wout", default=os.path.join(HERE, "weights.npz"))
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--data", default=None, help="also save samples (pickle)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra pickled sample files to include")
    a = ap.parse_args()
    t0 = time.time()
    seeds = list(range(a.seed0, a.seed0 + a.games))
    with Pool(a.workers, initializer=_init, initargs=(a.win, a.depth, a.eps)) as pool:
        res = pool.map(play, seeds, chunksize=20)
    scores = [r[0] for r in res]
    samples = [s for r in res for s in r[1]]
    print(f"selfplay depth={a.depth} games={a.games} mean={np.mean(scores):.1f} "
          f"se={np.std(scores) / np.sqrt(len(scores)):.1f} samples={len(samples)} t={time.time() - t0:.0f}s")
    import pickle
    if a.data:
        with open(a.data, "wb") as f:
            pickle.dump(samples, f)
    for e in a.extra:
        with open(e, "rb") as f:
            samples += pickle.load(f)
    F, G = fit(samples)
    np.savez(a.wout, F=F, G=G)
    print(f"saved {a.wout}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
