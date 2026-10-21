#!/usr/bin/env python3
"""Train the afterstate value network V(s) by self-play with Watkins-style TD(lambda) targets.

Decks are generated with numpy's PCG64 from TRAIN_SEED (never the bench's random.Random seeds).

Usage: python3 train.py [--iters N] [--games G] [--out weights.npz] [--init weights.npz]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

# train the installed strategy's core (strategy/vf_core.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "strategy"))
import vf_core as C  # noqa: E402

BASE_DECK = np.repeat(np.arange(1, 11, dtype=np.int16), [4, 4, 4, 4, 4, 4, 4, 4, 4, 16])


def selfplay(net, G, rng, eps, lam, expected=False):
    decks = rng.permuted(np.broadcast_to(BASE_DECK, (G, 52)), axis=1)
    pn = np.zeros((G, 4), np.int16)
    ph = np.zeros((G, 4), np.int16)
    pa = np.zeros((G, 4), np.int16)
    bu = np.zeros(G, np.int16)
    ho = np.full(G, 2, np.int16)
    st = np.zeros(G, np.int16)
    dk = np.tile(np.array([4] * 9 + [16], np.int16), (G, 1))
    score = np.zeros(G, np.float64)
    alive = np.ones(G, bool)
    T = 52
    rec = {k: [] for k in ("pn", "ph", "pa", "bu", "ho", "st", "dk")}
    qmax = np.zeros((T, G), np.float32)
    rch = np.zeros((T, G), np.float32)
    greedy = np.zeros((T, G), bool)
    term = np.zeros((T, G), bool)
    was_alive = np.zeros((T, G), bool)
    enext = np.zeros((T, G), np.float32)  # E_c[max Q] of the chosen afterstate (expected backup)
    for t in range(T):
        idx = np.nonzero(alive)[0]
        was_alive[t] = alive
        c = decks[idx, t]
        dk[idx, c - 1] -= 1
        n, h, a, bu2, ho2, st2, r, legal = C.expand(pn[idx], ph[idx], pa[idx], bu[idx], ho[idx], st[idx], c)
        A = len(idx)
        dkr = np.repeat(dk[idx], 8, axis=0)
        V = C.value(net, n.reshape(A * 8, 4), h.reshape(A * 8, 4), a.reshape(A * 8, 4),
                    bu2.reshape(-1), ho2.reshape(-1), st2.reshape(-1), dkr).reshape(A, 8)
        Q = r + V
        Q[~legal] = -np.inf
        best = np.argmax(Q, axis=1)
        choice = best.copy()
        if eps > 0:
            explore = rng.random(A) < eps
            if explore.any():
                rnd = rng.random((A, 8)) * legal
                choice[explore] = np.argmax(rnd[explore], axis=1)
        ar = np.arange(A)
        qmax[t, idx] = Q[ar, best]
        rch[t, idx] = r[ar, choice]
        greedy[t, idx] = (Q[ar, choice] >= Q[ar, best] - 1e-6)
        pn[idx] = n[ar, choice]
        ph[idx] = h[ar, choice]
        pa[idx] = a[ar, choice]
        bu[idx] = bu2[ar, choice]
        ho[idx] = ho2[ar, choice]
        st[idx] = st2[ar, choice]
        score[idx] += r[ar, choice]
        tm = (bu[idx] >= 3) | (t == T - 1)
        term[t, idx] = tm
        for k, arr in (("pn", pn), ("ph", ph), ("pa", pa), ("bu", bu), ("ho", ho), ("st", st), ("dk", dk)):
            rec[k].append(arr.copy())
        if expected:
            enext[t, idx] = C.expected_next(net, pn[idx], ph[idx], pa[idx], bu[idx], ho[idx], st[idx], dk[idx])
        alive[idx[tm]] = False
    # lambda returns, backwards
    Y = np.zeros((T, G), np.float32)
    valid = np.zeros((T, G), bool)
    Gnext = np.zeros(G, np.float32)
    for t in range(T - 1, -1, -1):
        v = was_alive[t] & ~term[t]
        # a game that ends here is worth its end-of-game bonus (whole deck played, no busts)
        tv = C.PERFECT_BONUS * ((t == T - 1) & (rec["bu"][t] == 0))
        if t < T - 1:
            boot = enext[t] if expected else qmax[t + 1]
            y = np.where(greedy[t + 1], (1 - lam) * boot + lam * (rch[t + 1] + Gnext), boot)
        else:
            y = np.zeros(G, np.float32)
        Y[t] = np.where(v, y, np.where(term[t], tv, 0))
        valid[t] = v
        Gnext = Y[t]
    m = valid.reshape(-1)
    data = {k: np.stack(rec[k]).reshape(T * G, *rec[k][0].shape[1:])[m] for k in rec}
    data["y"] = Y.reshape(-1)[m]
    data["X"] = np.concatenate([
        C.features(data["pn"][i:i + 65536], data["ph"][i:i + 65536], data["pa"][i:i + 65536],
                   data["bu"][i:i + 65536], data["ho"][i:i + 65536], data["st"][i:i + 65536],
                   data["dk"][i:i + 65536]).astype(np.float16)
        for i in range(0, len(data["y"]), 65536)])
    score += C.PERFECT_BONUS * (bu == 0)  # games with no busts played the whole deck
    return data, score


def fit(net, opt, data, epochs, bs, rng):
    N = len(data["y"])
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(N)
        for s in range(0, N - bs + 1, bs):
            b = perm[s:s + bs]
            X = data["X"][b].astype(np.float32)
            y = data["y"][b] / C.SCALE
            out, acts = net.forward_train(X)
            err = out - y
            # Huber-ish: clip gradient
            g = np.clip(err, -1.0, 1.0) / bs
            gW, gb = net.grads(acts, g.astype(np.float32))
            opt.step(gW, gb)
            losses.append(float((err ** 2).mean()))
    return float(np.mean(losses[-200:])) * C.SCALE ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--keep", type=int, default=3, help="iterations of data kept in buffer")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--expected", action="store_true", help="expected (all next cards) bootstrap")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    D = C.N_FEATURES
    if a.init:
        net = C.load_net(a.init)
    else:
        net = C.MLP([D] + [int(x) for x in a.hidden.split(",")] + [1], seed=a.seed)
        net.fast = C.FastNet(net)
    opt = Adam = C.Adam(net, lr=a.lr)
    buf = []
    print(f"features D={D}", flush=True)
    for it in range(a.iters):
        t0 = time.time()
        data, score = selfplay(net, a.games, rng, a.eps, a.lam, a.expected)
        t1 = time.time()
        buf.append(data)
        buf = buf[-a.keep:]
        merged = {k: np.concatenate([d[k] for d in buf]) for k in data}
        Adam.lr = a.lr * (0.5 if it >= a.iters * 0.6 else 1.0) * (0.5 if it >= a.iters * 0.85 else 1.0)
        net.fast = None
        mse = fit(net, opt, merged, a.epochs, a.bs, rng)
        net.fast = C.FastNet(net)
        t2 = time.time()
        np.savez(a.out, **net.params())
        print(f"it {it:3d}  selfplay mean {score.mean():7.1f} (eps={a.eps})  n={len(data['y'])}  "
              f"rmse {np.sqrt(mse):6.1f}  V0~{data['y'][:1]}  t_play {t1 - t0:5.1f}s t_fit {t2 - t1:5.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
