#!/usr/bin/env python3
"""Play bench decks with a strategy file and print where the points come from / go.

Usage: python3 analyze.py STRATEGY.py --games N --start S
"""
import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import twentyone as t  # noqa: E402
from bench import load, deck  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    mod = load(a.strategy)
    ev = Counter()
    bust_at = Counter()
    hold_at = []
    end3 = []
    left_cards = []
    unused_holds = Counter()
    final_busts = Counter()
    scores = []
    for seed in range(a.start, a.start + a.games):
        g = t.Game()
        for i, rank in enumerate(deck(seed)):
            if g.over():
                break
            _, idx, hold = mod.rank_moves(g, rank)[0]
            if hold:
                hold_at.append(i)
                ev["hold_cards_discarded"] += len(g.piles[idx])
            before = g.score
            b0 = g.busts
            n_before = len(g.piles[idx]) if not hold else 0
            g.place(rank, idx, hold)
            pts = g.score - before
            if g.busts > b0:
                ev["bust"] += 1
                bust_at[i // 13] += 1
            elif pts == 400:
                ev[f"21 in {n_before + 1}"] += 1
            elif pts == 1000:
                ev["5-card 21"] += 1
            elif pts == 800:
                ev["5-card stack"] += 1
        if g.busts >= 3:
            end3.append(g.cards_seen())
        left_cards.append(sum(len(p) for p in g.piles))
        unused_holds[g.holds] += 1
        final_busts[g.busts] += 1
        scores.append(g.score)
    n = a.games
    print(f"mean {sum(scores) / n:.1f}  n={n}")
    print("events per game:", {k: round(v / n, 3) for k, v in sorted(ev.items())})
    print("busts by quarter of deck:", dict(sorted(bust_at.items())))
    print("final busts:", dict(sorted(final_busts.items())), " unused holds:", dict(sorted(unused_holds.items())))
    print(f"3-bust endings: {len(end3)}  at cards seen: {sorted(end3)}")
    if hold_at:
        hold_at.sort()
        print(f"hold used at card index: median {hold_at[len(hold_at) // 2]}  quartiles "
              f"{hold_at[len(hold_at) // 4]} {hold_at[3 * len(hold_at) // 4]}")
    print(f"cards stranded in piles at end: {sum(left_cards) / n:.2f}")


if __name__ == "__main__":
    main()
