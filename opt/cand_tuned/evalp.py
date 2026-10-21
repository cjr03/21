"""Evaluate parameter sets on a seed range with the fast simulator (paired comparison).
usage: python3 evalp.py START COUNT WORKERS spec1 [spec2 ...]
spec = params.json[:depth][:key=val,...]   e.g. tuned_v1.json:2:omega=0.7
Prints mean/se for each spec and paired diffs vs the first spec (and vs cached baseline)."""
import json, math, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core, sim

_c = {}
def parse(spec):
    parts = spec.split(":")
    d = json.load(open(os.path.join(HERE, parts[0]))) if parts[0] not in ("", "default") else {}
    dp = parts[1].replace("a", "-").split("b") if len(parts) > 1 and parts[1] else ["1"]
    depth = (int(dp[0]), int(dp[1]) if len(dp) > 1 else 0)
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            d[k] = float(v)
    return d, depth

def work(t):
    spec, seed = t
    if spec not in _c:
        d, depth = parse(spec)
        _c[spec] = (core.Params(d), depth)
    P, depth = _c[spec]
    if depth[0] < 0:
        ch = (lambda P_, p, b, H, D, v: core.auto_moves(P_, p, b, H, D, v, depth=-depth[0], beam=depth[1]))
    else:
        ch = (lambda P_, p, b, H, D, v: core.score_moves(P_, p, b, H, D, v, depth=depth[0], beam=depth[1]))
    t0 = time.perf_counter()
    sc = sim.play(P, seed, chooser=ch)
    return spec, seed, sc, time.perf_counter() - t0

def stats(x):
    m = sum(x) / len(x)
    sd = math.sqrt(sum((a - m) ** 2 for a in x) / max(1, len(x) - 1))
    return m, sd / math.sqrt(len(x))

if __name__ == "__main__":
    start, count, workers = map(int, sys.argv[1:4])
    specs = sys.argv[4:]
    seeds = list(range(start, start + count))
    res = {s: {} for s in specs}; tm = {s: 0.0 for s in specs}
    with Pool(workers) as p:
        for spec, seed, sc, dt in p.imap_unordered(work, [(s, x) for x in seeds for s in specs], chunksize=2):
            res[spec][seed] = sc; tm[spec] += dt
    base = {}
    bf = os.path.join(HERE, "base_scores.json")
    if os.path.exists(bf):
        base = {int(k): v for k, v in json.load(open(bf)).items()}
    ref = [res[specs[0]][x] for x in seeds]
    for s in specs:
        xs = [res[s][x] for x in seeds]
        m, se = stats(xs)
        line = f"{s:<45} mean {m:7.1f} se {se:5.1f}  {tm[s]/count:6.3f}s/game"
        if s != specs[0]:
            dm, dse = stats([a - b for a, b in zip(xs, ref)])
            line += f"  vs first {dm:+6.1f} ±{dse:4.1f}"
        if all(x in base for x in seeds):
            dm, dse = stats([res[s][x] - base[x] for x in seeds])
            line += f"  vs baseline {dm:+6.1f} ±{dse:4.1f}"
        print(line, flush=True)
