"""Monte Carlo rollout strategy for the 21 solitaire calculator.

rank_moves(game, rank) -> [(score, idx, use_hold), ...] best first, where score is the
estimated expected number of future points (this card's points included) after the move.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import twentyone as tw  # noqa: E402
from core import pile_state, TN, TR, TB  # noqa: E402
from sim import Params, play, vtot  # noqa: E402

CFG = {
    "params": os.path.join(HERE, "base_api.npz"),
    "k0": 256,          # first-round samples per candidate
    "kmax": 4096,       # max samples per candidate
    "budget": 0.8,      # seconds per move (safety cap)
    "z": 2.5,           # racing elimination threshold (in paired SEs)
    "horizon": None,    # truncate rollouts after this many cards (None = play out)
    "seed": None,
}

_params = None
_rng = None


def _get_params():
    global _params
    if _params is None:
        _params = Params.load(CFG["params"])
    return _params


def _get_rng():
    global _rng
    if _rng is None:
        _rng = np.random.default_rng(CFG["seed"])
    return _rng


def _sample_decks(rng, remaining, k):
    vals = np.repeat(np.arange(1, 11), remaining)
    if len(vals) == 0:
        return np.zeros((k, 0), np.int64)
    keys = rng.random((k, len(vals)))
    return vals[keys.argsort(1)]


def rank_moves(game, rank):
    params = _get_params()
    v = tw.RANK_VALUE[rank]
    rem = game.remaining_values()
    rem[v] -= 1
    remaining = [max(0, rem[x]) for x in range(1, 11)]
    L = sum(remaining)

    P0 = [pile_state(p) for p in game.piles]
    moves = tw.options(game.piles, game.holds)
    # afterstates
    cands = {}   # canonical key -> (reward, P, b, h, [moves])
    for idx, uh in moves:
        P = list(P0)
        h = game.holds
        src = P[idx]
        if uh:
            src = 0
            h -= 1
        r = float(TR[src, v])
        b = game.busts + int(TB[src, v])
        P[idx] = int(TN[src, v])
        key = (r, tuple(sorted(P)), b, h)
        if key in cands:
            cands[key][4].append((idx, uh))
        else:
            cands[key] = [r, P, b, h, [(idx, uh)]]
    keys = list(cands)
    C = len(keys)
    R = np.array([cands[k][0] for k in keys])
    Pa = np.array([cands[k][1] for k in keys], np.int64)
    Ba = np.array([cands[k][2] for k in keys], np.int64)
    Ha = np.array([cands[k][3] for k in keys], np.int64)

    if C == 1 or L == 0:
        est = R + vtot(params, L, Pa, Ba, Ha)
        means = est
    else:
        means = _race(params, R, Pa, Ba, Ha, remaining, L)

    out = []
    for ci, k in enumerate(keys):
        for idx, uh in cands[k][4]:
            out.append((float(means[ci]), idx, uh))
    # stable: best score first, then no-hold before hold, then lower pile index
    out.sort(key=lambda m: (-m[0], m[2], m[1]))
    return out


def _race(params, R, Pa, Ba, Ha, remaining, L):
    rng = _get_rng()
    t0 = time.perf_counter()
    C = len(R)
    sums = np.zeros(C)
    samples = [[] for _ in range(C)]
    active = list(range(C))
    k = CFG["k0"]
    n_done = 0
    hz = CFG["horizon"]
    while True:
        decks = _sample_decks(rng, remaining, k)
        A = len(active)
        P = np.repeat(Pa[active], k, axis=0)
        b = np.repeat(Ba[active], k)
        h = np.repeat(Ha[active], k)
        cards = np.tile(decks, (A, 1))
        tot = play(params, P, b, h, cards, horizon=hz)
        tot = tot.reshape(A, k)
        for ai, c in enumerate(active):
            samples[c].append(tot[ai])
        n_done += k
        # racing elimination with paired differences vs current leader
        X = {c: np.concatenate(samples[c])[-n_done:] for c in active}
        m = {c: R[c] + X[c].mean() for c in active}
        lead = max(active, key=lambda c: m[c])
        keep = [lead]
        for c in active:
            if c == lead:
                continue
            d = X[lead] - X[c] + (R[lead] - R[c])
            se = d.std() / np.sqrt(len(d)) + 1e-9
            if d.mean() > CFG["z"] * se:
                continue
            keep.append(c)
        active = keep
        elapsed = time.perf_counter() - t0
        if len(active) == 1 or n_done >= CFG["kmax"]:
            break
        # time safety: estimate cost of the next round (doubling)
        per = elapsed / max(1, n_done)
        k = min(n_done, CFG["kmax"] - n_done)  # double the sample size
        if elapsed + per * k * 1.2 > CFG["budget"]:
            break
    # estimate for every candidate: the final leader's full mean; others are expressed relative
    # to it through the paired difference on the decks they shared (common prefix of rounds).
    lead = active[0]
    full = np.concatenate(samples[lead])
    means = np.zeros(C)
    means[lead] = R[lead] + full.mean()
    for c in range(C):
        if c == lead:
            continue
        xc = np.concatenate(samples[c])
        n = len(xc)
        diff = (R[lead] + full[:n].mean()) - (R[c] + xc.mean())
        means[c] = means[lead] - max(diff, 1e-3)
    return means
