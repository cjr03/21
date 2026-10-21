"""Calibrate the DP leaf tables against on-policy Monte Carlo returns.

The backward-DP fit is optimistic (max over approximate values compounds), and the optimism differs by
(busts, holds), which distorts how the search values holds / bust "lives".  Here we fit a correction
    y_MC - V_DP(s) ~ c[n][bh] + sum_piles e[n][pile]          (ridge, smoothed over n with knots)
and fold it into G / F.
usage: python3 calibrate.py WIN WOUT data1.pkl [data2.pkl ...] [--pile]
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

KNOTS = [1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45, 51]


def kw(n):
    for k in range(len(KNOTS) - 1):
        a, b = KNOTS[k], KNOTS[k + 1]
        if a <= n <= b:
            t = (n - a) / (b - a)
            return k, 1 - t, k + 1, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("win")
    ap.add_argument("wout")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--pile", action="store_true", help="also fit per-pile-state corrections")
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--lampile", type=float, default=200.0)
    ap.add_argument("--shrink", type=float, default=1.0, help="fraction of the correction applied")
    a = ap.parse_args()
    L = core.Leaf(a.win)
    S = []
    for p in a.data:
        with open(p, "rb") as f:
            S += pickle.load(f)
    NS = core.NS
    K = len(KNOTS)
    nb = 10  # 9 (busts,holds) cells + 1 common offset per knot
    npar = K * nb + (K * NS if a.pile else 0)
    A = np.zeros((npar, npar))
    bvec = np.zeros(npar)
    resid = []
    for p, b, h, n, y, dk in S:
        r = y - L(p, b, h, n, list(dk), core.deck_code(dk))
        resid.append(r)
        k0, w0, k1, w1 = kw(n)
        idx, val = [], []
        for k, w in ((k0, w0), (k1, w1)):
            idx.append(k * nb + b * 3 + h)
            val.append(w)
            idx.append(k * nb + 9)
            val.append(w)
            if a.pile:
                for s in p:
                    idx.append(K * nb + k * NS + s)
                    val.append(w)
        idx = np.array(idx)
        val = np.array(val)
        np.add.at(A, (idx[:, None], idx[None, :]), val[:, None] * val[None, :])
        np.add.at(bvec, idx, val * r)
    lam = np.full(npar, a.lam)
    lam[9:K * nb:nb] = 1.0
    if a.pile:
        lam[K * nb:] = a.lampile
    w = np.linalg.solve(A + np.diag(lam), bvec) * a.shrink
    resid = np.array(resid)
    print(f"N={len(S)} mean resid={resid.mean():.1f} rmse={np.sqrt((resid ** 2).mean()):.1f}")
    c = w[:K * nb].reshape(K, nb)
    e = w[K * nb:].reshape(K, NS) if a.pile else np.zeros((K, NS))
    F, G = L.F.copy(), L.G.copy()
    for n in range(1, 52):
        k0, w0, k1, w1 = kw(n)
        cc = w0 * c[k0] + w1 * c[k1]
        G[n] += cc[:9] + cc[9]
        F[n] += (w0 * e[k0] + w1 * e[k1])[None, :]
    for n in (5, 15, 25, 35, 45):
        k0, w0, k1, w1 = kw(n)
        cc = w0 * c[k0] + w1 * c[k1]
        print(f"n={n}: corr by bh " + " ".join(f"{x + cc[9]:6.0f}" for x in cc[:9]))
    np.savez(a.wout, F=F, G=G, W=L.W, U=L.U)


if __name__ == "__main__":
    main()
