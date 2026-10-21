"""CMA-ES tuning of the base policy tables (direct maximisation of mean game score)."""
import sys
import time

import numpy as np

from core import NS
from sim import EDGES, NB, Params, deal, play

# x(ph) = mean L of bucket / 51
XPH = np.array([0.0] + [((EDGES[k] + EDGES[k + 1] - 1) / 2) / 51 for k in range(1, NB)])
NV = NS - 1
K = NB - 1
DIM = 2 * NV + 2 * K + 3 * K + 2 * K


def to_params(th):
    th = th * 100.0
    V0, V1 = th[:NV], th[NV:2 * NV]
    o = 2 * NV
    f = th[o:o + K] / 100.0; o += K
    g = th[o:o + K] / 100.0; o += K
    w = th[o:o + 3 * K].reshape(K, 3); o += 3 * K
    u = th[o:o + 2 * K].reshape(K, 2)
    p = Params()
    for ph in range(1, NB):
        p.V[ph, 1:] = V0 * f[ph - 1] + V1 * g[ph - 1]
        p.W[ph] = w[ph - 1]
        p.U[ph, 1:] = u[ph - 1]
    return p


def from_cma1(th1):
    th1 = th1 * 100.0
    V0, V1 = th1[:NV], th1[NV:2 * NV]
    w = th1[2 * NV:2 * NV + 6].reshape(3, 2)
    u = th1[2 * NV + 6:].reshape(2, 2)
    x = XPH[1:]
    W = w[:, 0][None] + w[:, 1][None] * x[:, None]
    U = u[:, 0][None] + u[:, 1][None] * x[:, None]
    th = np.concatenate([V0, V1, np.ones(K) * 100, x * 100, W.ravel(), U.ravel()])
    return th / 100.0


def from_params(p):
    """Least-squares projection of tables onto the parametrisation."""
    X = np.stack([np.ones(NB - 1), XPH[1:]], 1)
    pinv = np.linalg.pinv(X)
    V = pinv @ p.V[1:, 1:]          # 2, NV
    W = pinv @ p.W[1:]              # 2, 3
    U = pinv @ p.U[1:, 1:]          # 2, 2
    th = np.concatenate([V[0], V[1], W.T.ravel(), U.T.ravel()]) / 100.0
    return th


def score(p, cards):
    n = cards.shape[0]
    return play(p, np.zeros((n, 4), np.int64), np.zeros(n, np.int64), np.full(n, 2), cards).mean()


def main():
    init = sys.argv[1]
    out = sys.argv[2]
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    ngames = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    rng = np.random.default_rng(7)
    if init.endswith("_theta.npy"):
        m = np.load(init)
        if len(m) != DIM:
            m = from_cma1(m)
    else:
        m = from_params(Params.load(init))
    test = deal(40000, np.random.default_rng(999))
    print("init test", score(to_params(m), test), flush=True)
    n = DIM
    lam = 24
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1 / (wts ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    pc = np.zeros(n)
    ps = np.zeros(n)
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    best_test = -1
    t0 = time.time()
    for g in range(gens):
        cards = deal(ngames, rng)
        z = rng.standard_normal((lam, n))
        y = (z * D) @ B.T
        xs = m + sigma * y
        f = np.array([score(to_params(x), cards) for x in xs])
        order = np.argsort(-f)
        ysel = y[order[:mu]]
        yw = wts @ ysel
        m = m + sigma * yw
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinvsqrt @ yw
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (ysel.T * wts) @ ysel)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if g % 5 == 0:
            C = np.triu(C) + np.triu(C, 1).T
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-20))
        if g % 20 == 19 or g == gens - 1:
            ts = score(to_params(m), test)
            if ts > best_test:
                best_test = ts
                to_params(m).save(out)
                np.save(out.replace(".npz", "_theta.npy"), m)
            print(f"gen {g} best-in-pop {f.max():.0f} mean-pop {f.mean():.0f} sigma {sigma:.3f} "
                  f"test {ts:.1f} (best {best_test:.1f}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
