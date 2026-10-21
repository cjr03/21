"""CMA-ES tuner (numpy) with common random numbers.

Every generation draws a fresh batch of tuning seeds from [SEED_LO, SEED_HI) and evaluates all
lambda candidates on the SAME decks, so ranking inside a generation is paired.  The mean of the
search distribution is written to tuned_<tag>.json after each generation.

usage: python3 tune.py TAG GENERATIONS GAMES_PER_CAND WORKERS [param ...]
env TUNE_DEPTH: search used while tuning, e.g. "1" (default), "2b2" (depth 2, beam 2),
"a2b2" (auto schedule: exact endgame + depth 2 beam 2).  TUNE_SIGMA: initial step (default 0.6).
"""
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402
import sim  # noqa: E402

SEED_LO, SEED_HI = 10000, 40000

SCALE = {"g": 30, "c": 15, "B0": 100, "kB": 0.5, "phi3": 100, "phi2": 200, "phi1": 400,
         "phi0": 800, "eta": 100, "rho": 40, "alpha": 0.3, "gamma": 0.15, "fpow": 0.5,
         "n1": 50, "n2": 50, "n3": 50, "n4": 50, "soft": 50, "omega": 0.4}

_P = {}
_DEPTH = os.environ.get("TUNE_DEPTH", "1")


def _chooser():
    d = _DEPTH.replace("a", "-").split("b")
    depth, beam = int(d[0]), (int(d[1]) if len(d) > 1 else 0)
    if depth < 0:
        return lambda P, p, b, H, D, v: core.auto_moves(P, p, b, H, D, v, depth=-depth, beam=beam)
    return lambda P, p, b, H, D, v: core.score_moves(P, p, b, H, D, v, depth=depth, beam=beam)


_CH = _chooser()


def _work(task):
    ci, pd, seed = task
    key = json.dumps(pd, sort_keys=True)
    P = _P.get(key)
    if P is None:
        _P.clear()
        P = _P[key] = core.Params(pd)
    return ci, seed, sim.play(P, seed, chooser=_CH)


def load_base():
    fn = os.path.join(HERE, "params.json")
    return json.load(open(fn)) if os.path.exists(fn) else {}


def main():
    tag, gens, games, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    names = sys.argv[5:] or list(SCALE)
    base = dict(core.DEFAULT)
    base.update(load_base())
    for k in list(base):
        if k not in core.DEFAULT and k not in SCALE:
            pass
    x0 = np.array([float(base[k]) for k in names])
    sc = np.array([float(SCALE.get(k, 1.0)) for k in names])
    n = len(names)
    lam = 4 + int(3 * math.log(n))
    lam = max(lam, 10)
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m = np.zeros(n)
    sigma = float(os.environ.get("TUNE_SIGMA", 0.6))
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    rng = np.random.default_rng(hash(tag) & 0xffffffff)
    srng = random.Random(tag)
    log = open(os.path.join(HERE, f"tune_{tag}.log"), "a")

    def todict(z):
        d = dict(base)
        for k, val in zip(names, x0 + sc * z):
            d[k] = float(val)
        return d

    with Pool(workers) as pool:
        for gen in range(gens):
            t0 = time.time()
            evals, B = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-20)
            D = np.sqrt(evals)
            arz = rng.standard_normal((lam, n))
            ary = arz * D @ B.T
            arx = m + sigma * ary
            cands = [todict(z) for z in arx] + [todict(m)]
            seeds = srng.sample(range(SEED_LO, SEED_HI), games)
            tasks = [(ci, cands[ci], s) for s in seeds for ci in range(len(cands))]
            res = np.zeros((len(cands), games))
            sidx = {s: j for j, s in enumerate(seeds)}
            for ci, s, val in pool.imap_unordered(_work, tasks, chunksize=8):
                res[ci, sidx[s]] = val
            fit = res[:lam].mean(1)
            order = np.argsort(-fit)
            y = ary[order[:mu]]
            ymean = w @ y
            m = m + sigma * ymean
            Cinvsqrt = B @ np.diag(1 / D) @ B.T
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ ymean)
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * ymean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * (y.T * w) @ y
            sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 1.5)
            md = todict(m)
            json.dump({k: md[k] for k in md}, open(os.path.join(HERE, f"tuned_{tag}.json"), "w"), indent=1)
            msg = (f"gen {gen} {time.time() - t0:.0f}s best {fit.max():.0f} meanfit {fit.mean():.0f} "
                   f"prevmean {res[lam].mean():.0f} sigma {sigma:.3f} | "
                   + " ".join(f"{k}={md[k]:.3g}" for k in names))
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()


if __name__ == "__main__":
    main()
