"""Fit the base-policy value tables by approximate policy iteration (Monte Carlo regression)."""
import sys
import time

import numpy as np

from core import NS
from sim import NB, NG, PHASE, Params, deal, gindex, play

D = NS + 3 + 3 + NG


def regress(params, n_games, rng, eps, ridge=1.0):
    N = n_games
    cards = deal(N, rng)
    P0 = np.zeros((N, 4), np.int64)
    total, recs = play(params, P0, np.zeros(N, np.int64), np.full(N, 2), cards,
                       eps=eps, rng=rng, record=True)
    # return-to-go after each step
    rews = np.stack([r[4] for r in recs])          # T, N
    rtg = np.cumsum(rews[::-1], 0)[::-1]
    rtg = np.concatenate([rtg[1:], np.zeros((1, N))], 0)  # future after step t
    XtX = np.zeros((NB, D, D))
    Xty = np.zeros((NB, D))
    for t, (L, P, b, h, rew, alive) in enumerate(recs):
        if L == 0:
            continue
        ph = PHASE[L]
        m = alive
        if not m.any():
            continue
        F = np.concatenate([P[m], NS + b[m, None], NS + 3 + h[m, None],
                            NS + 6 + gindex(P[m], b[m], h[m])[:, None]], 1)  # n,7
        y = rtg[t][m]
        n = F.shape[0]
        pairs = (F[:, :, None] * D + F[:, None, :]).ravel()
        XtX[ph] += np.bincount(pairs, minlength=D * D).reshape(D, D)
        Xty[ph] += np.bincount(F.ravel(), weights=np.repeat(y, 7), minlength=D)
    newp = Params()
    for ph in range(1, NB):
        A = XtX[ph] + ridge * np.eye(D)
        w = np.linalg.solve(A, Xty[ph])
        newp.V[ph] = w[:NS]
        newp.W[ph] = w[NS:NS + 3]
        newp.U[ph] = w[NS + 3:NS + 6]
        newp.G[ph] = w[NS + 6:]
        # normalise: empty pile value 0, shift into W
        e = newp.V[ph][0]
        newp.V[ph] -= e
        newp.W[ph] += 4 * e
    return newp, total.mean()


def evaluate(params, n, seed=12345):
    rng = np.random.default_rng(seed)
    cards = deal(n, rng)
    tot = play(params, np.zeros((n, 4), np.int64), np.zeros(n, np.int64), np.full(n, 2), cards)
    return tot.mean(), tot.std() / np.sqrt(n)


NG = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
EPS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    p = Params.load(sys.argv[6]) if len(sys.argv) > 6 else Params()
    if len(sys.argv) <= 6:
        for ph in range(1, NB):
            p.W[ph] = [0, -300, -700]
            p.U[ph] = [0, 150, 250]
    best = (evaluate(p, 20000)[0], p)
    print("init", best[0])
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for it in range(iters):
        t0 = time.time()
        eps = EPS
        newp, m = regress(p, NG, rng, eps)
        # damped update
        alpha = ALPHA
        p = Params(alpha * newp.V + (1 - alpha) * p.V, alpha * newp.W + (1 - alpha) * p.W,
                   alpha * newp.U + (1 - alpha) * p.U, alpha * newp.G + (1 - alpha) * p.G)
        ev, se = evaluate(p, 20000)
        print(f"it {it} train-mean {m:.0f} eval {ev:.1f} se {se:.1f} ({time.time() - t0:.1f}s)",
              flush=True)
        if ev > best[0]:
            best = (ev, p.copy())
    print("best", best[0])
    best[1].save(sys.argv[2] if len(sys.argv) > 2 else "base_params.npz")
