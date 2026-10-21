"""Exact endgame solver: memoized expectimax over the known remaining deck.

A pile is (cards, hard_total, has_ace); piles are kept sorted so symmetric states share a
memo entry. value(state) is the exact expected future points under optimal play.
"""

import os

STREAK_LEN = 4
STREAK_BONUS = (0, 0, 250, 500, 1000)  # bonus for the Nth scoring card in a row
PERFECT_BONUS = float(os.environ.get("VF_PERFECT", "1000"))  # whole deck played with no busts
_MEMO = {}


def _place(pile, v):
    """Card value v onto pile -> (new_pile, points, busted)."""
    n, h, a = pile
    n, h, a = n + 1, h + v, a or v == 1
    if h > 21:
        return (0, 0, False), 0, True
    best = h + 10 if a and h + 10 <= 21 else h
    if best == 21:
        return (0, 0, False), (1000 if n == 5 else 400), False
    if n == 5:
        return (0, 0, False), 800, False
    return (n, h, a), 0, False


def moves(piles, busts, holds, streak, v):
    """Yield (points, piles', busts', holds', streak', pile_idx, use_hold) for every legal move.

    Points include the streak bonus: the Nth scoring card in a row pays STREAK_BONUS[N] and the
    streak restarts after STREAK_LEN; a non-scoring card or a hold resets it.
    """
    for i, p in enumerate(piles):
        for use_hold in ((False, True) if holds and p[0] else (False,)):
            q, pts, bust = _place((0, 0, False) if use_hold else p, v)
            np_ = piles[:i] + (q,) + piles[i + 1:]
            s2 = (0 if use_hold else streak) + 1 if pts else 0
            pts += STREAK_BONUS[s2]
            if s2 == STREAK_LEN:
                s2 = 0
            yield pts, tuple(sorted(np_)), busts + bust, holds - use_hold, s2, i, use_hold


_CHILDREN = {}


def _children(piles, busts, holds, streak, v):
    """Distinct (points, piles', busts', holds', streak') outcomes of placing v; cached."""
    key = (piles, busts, holds, streak, v)
    got = _CHILDREN.get(key)
    if got is None:
        got = _CHILDREN[key] = list({m[:5] for m in moves(piles, busts, holds, streak, v)})
    return got


def value(piles, busts, holds, streak, deck):
    """Expected future points from an afterstate; deck is a 10-tuple of counts by value."""
    key = (piles, busts, holds, streak, deck)
    got = _MEMO.get(key)
    if got is not None:
        return got
    left = sum(deck)
    if busts >= 3 or left == 0:
        return PERFECT_BONUS if busts == 0 and left == 0 else 0.0
    total = 0.0
    for k, c in enumerate(deck):
        if not c:
            continue
        d2 = deck[:k] + (c - 1,) + deck[k + 1:]
        best = -1.0
        for pts, p2, b2, h2, s2 in _children(piles, busts, holds, streak, k + 1):
            if b2 >= 3:
                q = pts
            else:
                q = _MEMO.get((p2, b2, h2, s2, d2))
                q = pts + (q if q is not None else value(p2, b2, h2, s2, d2))
            if q > best:
                best = q
        total += c * best
    total /= left
    _MEMO[key] = total
    return total


def rank(game, card_value):
    """[(score, pile_idx, use_hold), ...] best first, exact."""
    piles = tuple((len(p), sum(p), 1 in p) for p in game.piles)
    rv = game.remaining_values()
    deck = [rv[v] for v in range(1, 11)]
    deck[card_value - 1] -= 1
    deck = tuple(deck)
    if len(_MEMO) > 2_000_000:
        _MEMO.clear()
        _CHILDREN.clear()
    out = [(pts + value(p2, b2, h2, s2, deck), i, uh)
           for pts, p2, b2, h2, s2, i, uh in moves(piles, game.busts, game.holds, game.streak, card_value)]
    out.sort(key=lambda x: (-round(x[0], 3), x[2], x[1]))
    return out
