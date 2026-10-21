#!/usr/bin/env python3
"""Benchmark 21-calculator strategies on identical shuffled decks (common random numbers).

A strategy is a .py file that defines  rank_moves(game, rank) -> [(score, idx, use_hold), ...]
sorted best first (only element [0] is used to play). `game` is a twentyone.Game.

Usage:
  python3 opt/bench.py STRATEGY.py [BASELINE.py] [--games N] [--start S] [--workers W]

Deck i is random.Random(i).shuffle of the 52 ranks, so every strategy sees the same decks.
Reports mean score, standard error, finish rate, ms per move and, when a baseline is given,
the paired mean difference with its standard error.
"""

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import twentyone as t  # noqa: E402

_strategy = None


def load(path):
    spec = importlib.util.spec_from_file_location(f"strat_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def deck(seed):
    d = [r for r in t.RANKS for _ in range(t.COPIES_PER_RANK)]
    random.Random(seed).shuffle(d)
    return d


def play(mod, seed):
    g = t.Game()
    moves, elapsed = 0, 0.0
    for rank in deck(seed):
        if g.over():
            break
        t0 = time.perf_counter()
        _, idx, use_hold = mod.rank_moves(g, rank)[0]
        elapsed += time.perf_counter() - t0
        moves += 1
        if use_hold:
            assert g.holds > 0 and g.piles[idx], "illegal hold"
        g.place(rank, idx, use_hold)
    return g.score, g.busts < t.MAX_BUSTS, moves, elapsed


def _init(path):
    global _strategy
    _strategy = load(path)


def _work(seed):
    return seed, play(_strategy, seed)


def run(path, seeds, workers):
    with Pool(workers, initializer=_init, initargs=(path,)) as pool:
        out = dict(pool.imap_unordered(_work, seeds, chunksize=1))
    return [out[s] for s in seeds]


def summary(name, res):
    scores = [r[0] for r in res]
    n = len(scores)
    mean = sum(scores) / n
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(1, n - 1))
    fin = sum(r[1] for r in res) / n
    ms = 1000 * sum(r[3] for r in res) / max(1, sum(r[2] for r in res))
    print(f"{name:<40} mean {mean:8.1f}  se {sd / math.sqrt(n):6.1f}  "
          f"finished {fin * 100:5.1f}%  {ms:8.2f} ms/move  (n={n})")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("baseline", nargs="?")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    seeds = list(range(a.start, a.start + a.games))
    s1 = summary(os.path.basename(a.strategy), run(a.strategy, seeds, a.workers))
    if a.baseline:
        s0 = summary(os.path.basename(a.baseline), run(a.baseline, seeds, a.workers))
        d = [x - y for x, y in zip(s1, s0)]
        md = sum(d) / len(d)
        sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, len(d) - 1))
        print(f"paired diff (strategy - baseline): {md:+.1f}  se {sd / math.sqrt(len(d)):.1f}")


if __name__ == "__main__":
    main()
