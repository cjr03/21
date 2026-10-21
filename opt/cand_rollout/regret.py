"""Decision-level noise regret of the racing allocator vs a large reference rollout."""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
import strategy as S  # noqa: E402
import twentyone as tw  # noqa: E402


def collect(n_games, start, every=3):
    pos = []
    for seed in range(start, start + n_games):
        g = tw.Game()
        for i, rank in enumerate(bench.deck(seed)):
            if g.over():
                break
            if i % every == 0:
                pos.append((pickle.dumps(g), rank))
            _, idx, uh = S.rank_moves(g, rank)[0]
            g.place(rank, idx, uh)
    return pos


def reference(pos, k=40000):
    out = []
    old = dict(S.CFG)
    S.CFG.update({"k0": k, "kmax": k, "z": 1e9, "budget": 1e9})
    for p, rank in pos:
        g = pickle.loads(p)
        out.append({(i, h): s for s, i, h in S.rank_moves(g, rank)})
    S.CFG.clear(); S.CFG.update(old)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        pos = collect(int(sys.argv[2]), int(sys.argv[3]))
        t0 = time.time()
        ref = reference(pos)
        print(len(pos), "positions", time.time() - t0)
        pickle.dump((pos, ref), open(sys.argv[4], "wb"))
    else:
        pos, ref = pickle.load(open(sys.argv[2], "rb"))
        cfg = json.loads(sys.argv[3])
        S.CFG.update(cfg)
        reg = []
        t0 = time.time()
        for (p, rank), r in zip(pos, ref):
            g = pickle.loads(p)
            _, i, h = S.rank_moves(g, rank)[0]
            reg.append(max(r.values()) - r[(i, h)])
        reg = np.array(reg)
        print(cfg, f"mean regret/move {reg.mean():.2f} se {reg.std() / np.sqrt(len(reg)):.2f} "
              f"P(regret>0) {np.mean(reg > 1e-6):.3f}  {1000 * (time.time() - t0) / len(pos):.0f} ms/pos")
