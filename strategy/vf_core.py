"""Shared core for the learned-value-function 21 strategy.

State representation (vectorised over a batch B):
    pn[B,4]  cards in each pile (0..4 after resolution)
    ph[B,4]  hard total of each pile (aces count 1)
    pa[B,4]  1 if the pile contains an ace
    bu[B]    busts so far (0..3)
    ho[B]    holds left (0..2)
    st[B]    scoring streak (0..3): consecutive cards that scored; the Nth pays STREAK_BONUS[N]
    dk[B,10] remaining deck counts by value (index k <-> value k+1)

An *afterstate* is the state right after a card has been placed and resolved; its deck is the
distribution of the NEXT card.  V(afterstate) = expected future points.  When the game is over
V is the end-of-game bonus: PERFECT_BONUS if the whole deck was played with no busts, else 0.
"""

import os

import numpy as np

NP = 4
VALS = np.arange(1, 11, dtype=np.int16)  # value of deck index k
MOVE_IDX = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
MOVE_HOLD = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
STREAK_LEN = 4
STREAK_BONUS = np.array([0, 0, 250, 500, 1000], np.float32)  # bonus for the Nth scoring card in a row
# finishing the deck with no busts pays this (the phone game's "perfect bonus"); VF_PERFECT=0 plays
# as if it didn't exist, for comparisons
PERFECT_BONUS = float(os.environ.get("VF_PERFECT", "1000"))


def terminal_value(bu, dk):
    """End-of-game bonus for afterstates (0 for states where the game goes on)."""
    return np.where((bu == 0) & (dk.sum(1) == 0), PERFECT_BONUS, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- transitions

def expand(pn, ph, pa, bu, ho, st, v):
    """Apply all 8 moves (4 piles x hold?) for card values v[B].

    Returns arrays of shape [B,8,...] for the afterstates plus reward[B,8] and legal[B,8].
    The deck is unchanged (caller already removed the current card).
    """
    B = pn.shape[0]
    n = np.broadcast_to(pn[:, None, :], (B, 8, NP)).copy()
    h = np.broadcast_to(ph[:, None, :], (B, 8, NP)).copy()
    a = np.broadcast_to(pa[:, None, :], (B, 8, NP)).copy()
    hold_mask = np.zeros((8, NP), dtype=bool)
    hold_mask[np.arange(4, 8), np.arange(4)] = True
    # legality: hold requires holds>0 and the pile non-empty
    pile_nonempty = pn > 0  # [B,4]
    legal = np.ones((B, 8), dtype=bool)
    legal[:, 4:] = pile_nonempty & (ho[:, None] > 0)
    # apply hold
    hm = np.broadcast_to(hold_mask[None], (B, 8, NP))
    n[hm] = 0
    h[hm] = 0
    a[hm] = 0
    ho2 = np.broadcast_to(ho[:, None], (B, 8)).astype(np.int16) - MOVE_HOLD[None, :]
    # place card on selected pile
    vv = v.astype(np.int16)[:, None]
    nn = n[:, np.arange(8), MOVE_IDX]  # [B,8]
    hh = h[:, np.arange(8), MOVE_IDX] + vv
    aa = a[:, np.arange(8), MOVE_IDX] | (vv == 1)
    nn = nn + 1
    bust = hh > 21
    best = np.where(aa.astype(bool) & (hh + 10 <= 21), hh + 10, hh)
    is21 = (~bust) & (best == 21)
    five = (~bust) & (~is21) & (nn == 5)
    reward = np.where(is21, np.where(nn == 5, 1000, 400), 0) + np.where(five, 800, 0)
    clear = bust | is21 | five
    scored = is21 | five
    st2 = np.where(scored, np.where(MOVE_HOLD[None, :], 0, st[:, None]) + 1, 0)
    reward = reward + STREAK_BONUS[st2]
    st2 = np.where(st2 == STREAK_LEN, 0, st2).astype(np.int16)
    nn = np.where(clear, 0, nn)
    hh = np.where(clear, 0, hh)
    aa = np.where(clear, 0, aa)
    n[:, np.arange(8), MOVE_IDX] = nn
    h[:, np.arange(8), MOVE_IDX] = hh
    a[:, np.arange(8), MOVE_IDX] = aa
    bu2 = np.broadcast_to(bu[:, None], (B, 8)).astype(np.int16) + bust
    return n, h, a, bu2, ho2, st2, reward.astype(np.float32), legal


# --------------------------------------------------------------------------- features

N_HARD = 21  # hard totals 0..20 one-hot


def features(pn, ph, pa, bu, ho, st, dk):
    """Feature matrix float32 [B, D] for afterstates (flattened batch)."""
    B = pn.shape[0]
    pn = pn.astype(np.int16)
    ph = ph.astype(np.int16)
    pa = pa.astype(np.int16)
    dkf = dk.astype(np.float32)
    L = dkf.sum(1)  # cards left
    Ls = np.maximum(L, 1.0)
    p = dkf / Ls[:, None]  # next-card distribution [B,10]

    # canonical pile order: sort by key descending
    key = pn * 64 + ph * 2 + pa
    order = np.argsort(-key, axis=1, kind="stable")
    pn = np.take_along_axis(pn, order, 1)
    ph = np.take_along_axis(ph, order, 1)
    pa = np.take_along_axis(pa, order, 1)

    # outcome of each value on each pile [B,4,10]
    h2 = ph[:, :, None] + VALS[None, None, :]
    a2 = (pa[:, :, None] > 0) | (VALS[None, None, :] == 1)
    n2 = pn[:, :, None] + 1
    bust = h2 > 21
    best = np.where(a2 & (h2 + 10 <= 21), h2 + 10, h2)
    is21 = (~bust) & (best == 21)
    five = (~bust) & (~is21) & (n2 == 5)
    pts = np.where(is21, np.where(n2 == 5, 1000, 400), 0) + np.where(five, 800, 0)
    pp = p[:, None, :]
    pbust = (bust * pp).sum(2)
    p21 = (is21 * pp).sum(2)
    p5 = (five * pp).sum(2)
    ept = (pts * pp).sum(2) / 1000.0
    empty = pn == 0
    pbust = np.where(empty, 0, pbust)

    cols = []
    # per pile block
    for i in range(NP):
        oh_n = np.zeros((B, 5), np.float32)
        oh_n[np.arange(B), pn[:, i]] = 1
        oh_h = np.zeros((B, N_HARD), np.float32)
        oh_h[np.arange(B), np.minimum(ph[:, i], N_HARD - 1)] = 1
        soft = ((pa[:, i] > 0) & (ph[:, i] + 10 < 21)).astype(np.float32)
        cols += [oh_n, oh_h, (pa[:, i] > 0).astype(np.float32)[:, None], soft[:, None],
                 pbust[:, i:i + 1], p21[:, i:i + 1], p5[:, i:i + 1], ept[:, i:i + 1],
                 ((21 - ph[:, i]) / 21.0).astype(np.float32)[:, None]]
    # cross-pile features
    fits = (~bust | empty[:, :, None]).any(1)  # value fits somewhere without bust [B,10]
    pforced = ((~fits) * p).sum(1)
    scores_any = ((is21 | five) & ~empty[:, :, None]).any(1)
    pscore = (scores_any * p).sum(1)
    nempty = empty.sum(1).astype(np.float32)
    oh_b = np.zeros((B, 3), np.float32)
    oh_b[np.arange(B), np.minimum(bu, 2)] = 1
    oh_o = np.zeros((B, 3), np.float32)
    oh_o[np.arange(B), ho] = 1
    lf = L / 52.0
    oh_l = np.zeros((B, 13), np.float32)
    oh_l[np.arange(B), np.minimum((L // 4).astype(np.int64), 12)] = 1
    cols += [pforced[:, None], pscore[:, None], (nempty / 4.0)[:, None], oh_b, oh_o,
             lf[:, None], (lf * lf)[:, None], oh_l, p, dkf / 16.0,
             (oh_b * lf[:, None]), (oh_o * lf[:, None]), _streak_onehot(st)]
    X = np.concatenate([c.astype(np.float32) for c in cols], axis=1)
    return X


def _streak_onehot(st):
    oh = np.zeros((st.shape[0], STREAK_LEN), np.float32)
    oh[np.arange(st.shape[0]), st] = 1
    return oh


# --------------------------------------------------------------------------- MLP

class MLP:
    def __init__(self, sizes=None, params=None, seed=0):
        if params is not None:
            self.W = [params[f"W{i}"] for i in range(len(params) // 2)]
            self.b = [params[f"b{i}"] for i in range(len(params) // 2)]
            return
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            w = rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32)
            if i == len(sizes) - 2:
                w *= 0.1
            self.W.append(w)
            self.b.append(np.zeros(sizes[i + 1], np.float32))

    def params(self):
        d = {}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"] = w
            d[f"b{i}"] = b
        return d

    def forward(self, X):
        h = X
        for i in range(len(self.W) - 1):
            h = h @ self.W[i] + self.b[i]
            np.maximum(h, 0, out=h)
        return (h @ self.W[-1] + self.b[-1])[:, 0]

    def forward_train(self, X):
        acts = [X]
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0)
            acts.append(h)
        out = (h @ self.W[-1] + self.b[-1])[:, 0]
        return out, acts

    def grads(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = dout[:, None]
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        return gW, gb


class Adam:
    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.mW = [np.zeros_like(w) for w in net.W]
        self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]
        self.vb = [np.zeros_like(b) for b in net.b]
        self.t = 0

    def step(self, gW, gb):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for P, G, M, V in ((self.net.W, gW, self.mW, self.vW), (self.net.b, gb, self.mb, self.vb)):
            for i in range(len(P)):
                M[i] = self.b1 * M[i] + (1 - self.b1) * G[i]
                V[i] = self.b2 * V[i] + (1 - self.b2) * G[i] * G[i]
                P[i] -= self.lr * (M[i] / c1) / (np.sqrt(V[i] / c2) + self.eps)


SCALE = 1000.0  # network output is points / SCALE


def value(net, pn, ph, pa, bu, ho, st, dk):
    """V for a flat batch of afterstates (points). Terminal states -> their end-of-game bonus."""
    out = terminal_value(bu, dk)
    term = (bu >= 3) | (dk.sum(1) == 0)
    live = ~term
    if live.any():
        fast = getattr(net, "fast", None)
        if fast is not None:
            out[live] = fast.forward_state(pn[live], ph[live], pa[live], bu[live], ho[live], st[live],
                                           dk[live]) * SCALE
        else:
            X = features(pn[live], ph[live], pa[live], bu[live], ho[live], st[live], dk[live])
            out[live] = net.forward(X) * SCALE
    return out


def load_net(path, fast=True):
    with np.load(path) as z:
        net = MLP(params={k: z[k].astype(np.float32) for k in z.files})
    if net.W[0].shape[0] == N_FEATURES - STREAK_LEN:
        # weights from before the streak rule: streak inputs start with zero weight
        net.W[0] = np.concatenate([net.W[0], np.zeros((STREAK_LEN, net.W[0].shape[1]), np.float32)])
    if fast:
        net.fast = FastNet(net)
    return net


def expected_next(net, pn, ph, pa, bu, ho, st, dk, chunk=4096):
    """E over the next card of max over moves (r + V(child)) for a flat batch of afterstates."""
    out = terminal_value(bu, dk)
    live_all = np.nonzero((bu < 3) & (dk.sum(1) > 0))[0]
    for c0 in range(0, len(live_all), chunk):
        live = live_all[c0:c0 + chunk]
        sdk = dk[live]
        si, kk = np.nonzero(sdk > 0)
        w = sdk[si, kk].astype(np.float32) / sdk.sum(1)[si].astype(np.float32)
        src = live[si]
        dk2 = dk[src].copy()
        dk2[np.arange(len(src)), kk] -= 1
        n, h, a, bu2, ho2, st2, r, legal = expand(pn[src], ph[src], pa[src], bu[src], ho[src], st[src], kk + 1)
        M = len(src)
        v = value(net, n.reshape(M * 8, NP), h.reshape(M * 8, NP), a.reshape(M * 8, NP),
                  bu2.reshape(-1), ho2.reshape(-1), st2.reshape(-1), np.repeat(dk2, 8, axis=0)).reshape(M, 8)
        q = np.where(legal, r + v, -np.inf).max(1)
        np.add.at(out, src, w * q)
    return out


# --------------------------------------------------------------------------- fast inference
# Exact re-parametrisation of  MLP(features(...))  in which every one-hot block of the first
# layer becomes a table lookup and the pile/next-card outcome statistics come from precomputed
# per-pile-type tables.  Pile type = n*42 + hard*2 + ace  (n 0..4, hard 0..20, ace 0/1).

NT = 5 * 42


def _type_tables():
    n = np.arange(NT) // 42
    h = (np.arange(NT) % 42) // 2
    a = np.arange(NT) % 2
    h2 = h[:, None] + VALS[None, :].astype(np.int64)
    a2 = (a[:, None] > 0) | (VALS[None, :] == 1)
    n2 = n[:, None] + 1
    bust = h2 > 21
    best = np.where(a2 & (h2 + 10 <= 21), h2 + 10, h2)
    is21 = (~bust) & (best == 21)
    five = (~bust) & (~is21) & (n2 == 5)
    pts = np.where(is21, np.where(n2 == 5, 1000, 400), 0) + np.where(five, 800, 0)
    empty = n == 0
    fit = ~bust | empty[:, None]
    score = (is21 | five) & ~empty[:, None]
    # outcome stats table [10, NT*4]: columns (type, stat) for stat in pbust,p21,p5,ept
    stats = np.stack([bust & ~empty[:, None], is21, five, pts / 1000.0], axis=2)  # [NT,10,4]
    tab = stats.transpose(1, 0, 2).reshape(10, NT * 4).astype(np.float32)
    return n, h, a, tab, fit, score


_TN, _TH, _TA, _TAB, _FIT, _SCORE = _type_tables()
_TT = np.ascontiguousarray(_TAB.reshape(10, NT, 4).transpose(1, 2, 0))  # [NT, stat, value]


class FastNet:
    def __init__(self, mlp):
        W1, b1 = mlp.W[0], mlp.b[0]
        H = W1.shape[1]
        PB = 5 + N_HARD + 7
        self.E = np.zeros((4, NT, H), np.float32)
        self.Wd = np.zeros((16, H), np.float32)
        soft = ((_TA > 0) & (_TH + 10 < 21)).astype(np.float32)
        for i in range(4):
            o = i * PB
            self.E[i] = (W1[o + _TN] + W1[o + 5 + np.minimum(_TH, N_HARD - 1)]
                         + _TA[:, None] * W1[o + 5 + N_HARD] + soft[:, None] * W1[o + 6 + N_HARD]
                         + ((21 - _TH) / 21.0)[:, None] * W1[o + 11 + N_HARD])
            self.Wd[i * 4:(i + 1) * 4] = W1[o + 7 + N_HARD:o + 11 + N_HARD]
        g = 4 * PB
        self.Wg = np.concatenate([W1[g:g + 3], W1[g + 9:g + 11], W1[g + 24:g + 44]])  # 25 rows
        self.Wb = W1[g + 3:g + 6] + b1[None, :]
        self.Wo = W1[g + 6:g + 9].copy()
        self.WL = W1[g + 11:g + 24].copy()
        self.Wbl = W1[g + 44:g + 47].copy()
        self.Wol = W1[g + 47:g + 50].copy()
        self.Ws = W1[g + 50:g + 50 + STREAK_LEN].copy()
        assert g + 50 + STREAK_LEN == W1.shape[0]
        self.W = mlp.W[1:]
        self.b = mlp.b[1:]

    def forward_state(self, pn, ph, pa, bu, ho, st, dk):
        B = pn.shape[0]
        t = pn.astype(np.int64) * 42 + ph.astype(np.int64) * 2 + pa.astype(np.int64)
        t = -np.sort(-t, axis=1)  # canonical order (same as features())
        dkf = dk.astype(np.float32)
        L = dkf.sum(1)
        p = dkf / np.maximum(L, 1.0)[:, None]
        Q = (_TT[t].reshape(B, 16, 10) @ p[:, :, None])[:, :, 0]  # [B, slot*4 + stat]
        fitany = _FIT[t].any(1)  # [B,10]
        scany = _SCORE[t].any(1)
        pforced = ((~fitany) * p).sum(1)
        pscore = (scany * p).sum(1)
        nempty = (t == 0).sum(1) / 4.0
        lf = L / 52.0
        G = np.concatenate([pforced[:, None], pscore[:, None], nempty[:, None], lf[:, None],
                            (lf * lf)[:, None], p, dkf / 16.0], axis=1).astype(np.float32)
        h = Q @ self.Wd + G @ self.Wg
        h += self.E[0][t[:, 0]] + self.E[1][t[:, 1]] + self.E[2][t[:, 2]] + self.E[3][t[:, 3]]
        bi = np.minimum(bu, 2).astype(np.int64)
        hi = ho.astype(np.int64)
        h += self.Wb[bi] + self.Wo[hi] + self.WL[np.minimum((L // 4).astype(np.int64), 12)]
        h += self.Ws[st.astype(np.int64)]
        h += lf[:, None] * (self.Wbl[bi] + self.Wol[hi])
        np.maximum(h, 0, out=h)
        for i in range(len(self.W) - 1):
            h = h @ self.W[i] + self.b[i]
            np.maximum(h, 0, out=h)
        return (h @ self.W[-1] + self.b[-1])[:, 0]


def fast_value(fnet, pn, ph, pa, bu, ho, st, dk):
    B = pn.shape[0]
    out = np.zeros(B, np.float32)
    live = (bu < 3) & (dk.sum(1) > 0)
    if live.all():
        return fnet.forward_state(pn, ph, pa, bu, ho, st, dk) * SCALE
    if live.any():
        out[live] = fnet.forward_state(pn[live], ph[live], pa[live], bu[live], ho[live], st[live],
                                       dk[live]) * SCALE
    return out


N_FEATURES = features(np.zeros((1, NP), np.int16), np.zeros((1, NP), np.int16), np.zeros((1, NP), np.int16),
                      np.zeros(1, np.int16), np.full(1, 2, np.int16), np.zeros(1, np.int16),
                      np.array([[4] * 9 + [16]], np.int16)).shape[1]
