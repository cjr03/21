"""Learned-value-function strategy for the 21 solitaire calculator.

rank_moves(game, rank) -> [(score, idx, use_hold), ...] best first, where score is the expected
total of FUTURE points (including this card's immediate reward) under the search policy.

Algorithm: expectimax over the exact remaining-deck distribution, with an afterstate value
network V (numpy MLP trained by self-play TD(lambda), see train.py) at the leaves.
  depth 1: score(move) = r(move) + V(afterstate)
  depth 2: score(move) = r(move) + E_next card[ max_move2 r2 + V(afterstate2) ]   (default)
  depth 3: one more card/move layer, children pruned to the top-K by depth-1 value.
"""
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
import vf_core as C  # noqa: E402
import twentyone as T  # noqa: E402

WEIGHTS = os.environ.get("VF_WEIGHTS", os.path.join(_HERE, "weights.npz"))
DEPTH = int(os.environ.get("VF_DEPTH", "2"))
TOPK = int(os.environ.get("VF_TOPK", "3"))
DEEP_DEPTH = int(os.environ.get("VF_DEEP", "0"))  # re-search close root moves at this depth
MARGIN = float(os.environ.get("VF_MARGIN", "200"))
_NET = None


def net():
    global _NET
    if _NET is None:
        _NET = C.load_net(WEIGHTS)
    return _NET


def _dedup(n, h, a, bu2, ho2):
    """Unique canonical children among the 8 moves of each row. Returns (uniq_idx, inverse)."""
    M = n.shape[0]
    t = n.astype(np.int64) * 42 + h.astype(np.int64) * 2 + a.astype(np.int64)  # [M,8,4]
    t = -np.sort(-t, axis=2)
    key = ((t[:, :, 0] << 24) | (t[:, :, 1] << 16) | (t[:, :, 2] << 8) | t[:, :, 3]
           | (bu2.astype(np.int64) << 32) | (ho2.astype(np.int64) << 34)
           | (np.arange(M, dtype=np.int64)[:, None] << 36))
    _, uidx, inv = np.unique(key.reshape(-1), return_index=True, return_inverse=True)
    return uidx, inv.reshape(-1)


def _eval_after(pn, ph, pa, bu, ho, dk, depth):
    """Expected future points of a flat batch of afterstates searching `depth` more card layers."""
    nt = net()
    if depth == 0:
        return C.value(nt, pn, ph, pa, bu, ho, dk)
    B = pn.shape[0]
    out = np.zeros(B, np.float32)
    live = np.nonzero((bu < 3) & (dk.sum(1) > 0))[0]
    if len(live) == 0:
        return out
    # children: (state, card value k) for k with count>0
    sdk = dk[live]
    si, kk = np.nonzero(sdk > 0)
    w = sdk[si, kk].astype(np.float32) / sdk.sum(1)[si].astype(np.float32)
    src = live[si]
    dk2 = dk[src].copy()
    dk2[np.arange(len(src)), kk] -= 1
    n, h, a, bu2, ho2, r, legal = C.expand(pn[src], ph[src], pa[src], bu[src], ho[src], kk + 1)
    M = len(src)
    uidx, inv = _dedup(n, h, a, bu2, ho2)
    fn, fh, fa = n.reshape(M * 8, 4)[uidx], h.reshape(M * 8, 4)[uidx], a.reshape(M * 8, 4)[uidx]
    fb, fo = bu2.reshape(-1)[uidx], ho2.reshape(-1)[uidx]
    fd = dk2[uidx // 8]
    if depth == 1 or TOPK >= 8:
        v = _eval_after(fn, fh, fa, fb, fo, fd, depth - 1)[inv].reshape(M, 8)
    else:
        v0 = C.value(nt, fn, fh, fa, fb, fo, fd)[inv].reshape(M, 8)
        q0 = np.where(legal, r + v0, -np.inf)
        top = np.argsort(-q0, axis=1)[:, :TOPK]
        mask = np.zeros((M, 8), bool)
        np.put_along_axis(mask, top, True, axis=1)
        mask &= legal
        # which unique children are needed
        need = np.zeros(len(uidx), bool)
        need[inv[mask.reshape(-1)]] = True
        sel = np.nonzero(need)[0]
        vu = np.full(len(uidx), -np.inf, np.float32)
        vu[sel] = _eval_after(fn[sel], fh[sel], fa[sel], fb[sel], fo[sel], fd[sel], depth - 1)
        v = vu[inv].reshape(M, 8)
        legal = mask
    q = np.where(legal, r + v, -np.inf).max(1)
    np.add.at(out, src, w * q)
    return out


def rank_moves(game, rank):
    value = T.RANK_VALUE[rank]
    rv = game.remaining_values()
    dk = np.array([[rv[v] for v in range(1, 11)]], np.int16)
    dk[0, value - 1] -= 1
    pn = np.array([[len(p) for p in game.piles]], np.int16)
    ph = np.array([[sum(p) for p in game.piles]], np.int16)
    pa = np.array([[1 if 1 in p else 0 for p in game.piles]], np.int16)
    bu = np.array([game.busts], np.int16)
    ho = np.array([game.holds], np.int16)
    n, h, a, bu2, ho2, r, legal = C.expand(pn, ph, pa, bu, ho, np.array([value]))
    idx = np.nonzero(legal[0])[0]
    # identical canonical afterstates (e.g. any of several empty piles) are searched once
    t = n[0].astype(np.int64) * 42 + h[0].astype(np.int64) * 2 + a[0].astype(np.int64)
    t = -np.sort(-t, axis=1)
    key = ((t[:, 0] << 24) | (t[:, 1] << 16) | (t[:, 2] << 8) | t[:, 3]
           | (bu2[0].astype(np.int64) << 32) | (ho2[0].astype(np.int64) << 34)
           | (r[0].astype(np.int64) // 200 << 36))[idx]
    _, uidx, inv = np.unique(key, return_index=True, return_inverse=True)
    um = idx[uidx]
    ur = r[0, um]
    us = (n[0, um], h[0, um], a[0, um], bu2[0, um], ho2[0, um], np.repeat(dk, len(um), axis=0))
    usc = ur + _eval_after(*us, max(0, DEPTH - 1))
    if DEEP_DEPTH > DEPTH and len(um) > 1:
        # re-search the contenders one card layer deeper
        cand = np.nonzero(usc >= usc.max() - MARGIN)[0]
        if len(cand) > 1:
            sub = tuple(x[cand] for x in us)
            usc[cand] = ur[cand] + _eval_after(*sub, DEEP_DEPTH - 1)
    scores = usc[inv.reshape(-1)]
    ranked = [(float(s), int(C.MOVE_IDX[m]), bool(C.MOVE_HOLD[m])) for s, m in zip(scores, idx)]
    # stable tie-break: plain placement before hold, lower pile index first
    ranked.sort(key=lambda x: (-round(x[0], 3), x[2], x[1]))
    return ranked
