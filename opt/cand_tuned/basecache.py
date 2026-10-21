"""Cache baseline (twentyone.rank_moves) scores per seed for paired comparisons."""
import sys, os, json
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opt")
import bench, twentyone as t


class M:
    rank_moves = staticmethod(t.rank_moves)


def work(s):
    return s, bench.play(M, s)[0]


if __name__ == "__main__":
    a, b, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_scores.json")
    d = json.load(open(fn)) if os.path.exists(fn) else {}
    with Pool(w) as p:
        for s, sc in p.imap_unordered(work, [s for s in range(a, b) if str(s) not in d], chunksize=4):
            d[str(s)] = sc
    json.dump(d, open(fn, "w"))
    print(len(d))
