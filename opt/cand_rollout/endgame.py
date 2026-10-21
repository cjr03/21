"""Exact expectimax for the last few cards (remaining deck composition is known exactly)."""
from core import TN, TR, TB

_TN = TN.tolist()
_TR = TR.tolist()
_TB = TB.tolist()


def solve_moves(P, b, h, v, rem):
    """Exact values of every distinct afterstate for placing value v.

    P: list of 4 pile states, rem: tuple of 10 counts (values 1..10) AFTER removing v.
    Returns list of (value, key) where key=(src_pile_idx, use_hold) of a representative.
    """
    memo = {}
    out = []
    for idx in range(4):
        for uh in (False, True):
            if uh and (h == 0 or P[idx] == 0):
                continue
            src = 0 if uh else P[idx]
            r = _TR[src][v]
            nb = b + _TB[src][v]
            nh = h - (1 if uh else 0)
            if nb >= 3:
                out.append((r, (idx, uh)))
                continue
            NP = list(P)
            NP[idx] = _TN[src][v]
            out.append((r + _value(tuple(sorted(NP)), nb, nh, rem, memo), (idx, uh)))
    return out


def _value(P, b, h, rem, memo):
    key = (P, b, h, rem)
    got = memo.get(key)
    if got is not None:
        return got
    L = sum(rem)
    if L == 0:
        return 0.0
    tot = 0.0
    rl = list(rem)
    for vi in range(10):
        c = rem[vi]
        if c == 0:
            continue
        v = vi + 1
        rl[vi] -= 1
        nrem = tuple(rl)
        rl[vi] += 1
        best = -1.0
        seen = set()
        for idx in range(4):
            s = P[idx]
            for uh in (False, True):
                if uh and (h == 0 or s == 0):
                    continue
                src = 0 if uh else s
                if (src, uh) in seen:
                    continue
                seen.add((src, uh))
                r = _TR[src][v]
                nb = b + _TB[src][v]
                if nb >= 3:
                    val = r
                else:
                    NP = list(P)
                    NP[idx] = _TN[src][v]
                    val = r + _value(tuple(sorted(NP)), nb, h - (1 if uh else 0), nrem, memo)
                if val > best:
                    best = val
        tot += c * best
    res = tot / L
    memo[key] = res
    return res
