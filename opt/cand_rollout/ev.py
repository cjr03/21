"""Run a strategy over seeds, save per-seed scores (npz) for paired comparisons.
usage: ev.py STRATEGY OUT.npz --games N --start S --workers W [--cfg key=val ...]"""
import argparse, json, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import bench

_cfg = {}

def _init(path, cfg):
    bench._init(path)
    if cfg:
        bench._strategy.CFG.update(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy'); ap.add_argument('out')
    ap.add_argument('--games', type=int, default=200); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--cfg', default='{}')
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    seeds = list(range(a.start, a.start + a.games))
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.strategy, cfg)) as pool:
        out = dict(pool.imap_unordered(bench._work, seeds, chunksize=1))
    res = [out[s] for s in seeds]
    sc = np.array([r[0] for r in res], float)
    fin = np.array([r[1] for r in res])
    ms = 1000 * sum(r[3] for r in res) / sum(r[2] for r in res)
    np.savez(a.out, seeds=np.array(seeds), scores=sc, fin=fin, ms=ms)
    print(f"{a.out}: mean {sc.mean():.1f} se {sc.std()/math.sqrt(len(sc)):.1f} fin {fin.mean()*100:.1f}% {ms:.1f} ms/move wall {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
