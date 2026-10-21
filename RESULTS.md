# 21 Calculator — strategy optimization results

## Current rules: + perfect bonus (the phone game's end-of-game scoring)
The phone game adds a **perfect bonus of +1000** when all 52 cards are played without a bust
(final screen: TOTAL = SCORE + PERFECT BONUS). The engine had never been told, so it valued a
first bust at almost nothing when it really costs 1000. The rule is now in the game
(`twentyone.py`), the network's terminal value and training targets (`strategy/vf_core.py`,
`opt/cand_valuefn/train.py`), the search (`strategy/strategy.py`) and the exact endgame solver
(`strategy/endgame.py`); `VF_PERFECT=0` restores the old bonus-blind play for comparisons.

Scored by the phone's rules, 500 held-out decks (`opt/heldout_500_perfect.txt`):

| Strategy | Mean | SE | Finished deck | ms/move |
|---|---|---|---|---|
| Previous app (bonus-blind network and search) | 8563 | 51 | 98.8% | ~60 |
| **Bonus-aware search + network retrained with the bonus (in the app)** | **9008** | 52 | 99.6% | ~60 |

**+445 ± 56 per game.** Steps, on tuning decks 0–399: making only the search and endgame aware
of the bonus was +39 ± 10; retraining the network (60 self-play TD(λ) iterations warm-started
from the streak network, self-play mean 7534 → ~7950; `train_perfect.log`, `train_perfect2.log`)
was +484 ± 60 on top (30 iterations: +433 ± 66). New weights: `opt/cand_valuefn/weights_perfect.npz`
(installed as `strategy/weights.npz`; previous: `weights_streak.npz`). With the new network more
search no longer pays (decks 400–799): re-searching every move at depth 3 +4 ± 4 at 3x the time,
5 kept replies +0.6 ± 5, so the search settings are unchanged.

## Current rules: scoring streaks (2nd +250, 3rd +500, 4th +1000 in a row)
Consecutive cards that each score (21 or 5-card stack) build a streak; a non-scoring card or a
stay resets it, and it restarts after the 4th (+1750 total). The game (`twentyone.py`), the
network's game model (`strategy/vf_core.py`, which also gets the streak as an input), the search
and the exact endgame solver all apply this rule; a test confirmed they score identically.

The network was retrained (warm-started from the previous one: 20 self-play iterations with a
flat 4-in-a-row +1000 bonus, then 25 under these tiered bonuses, where self-play mean rose
7478 → ~7630). 500 held-out decks, same decks for both:

| Strategy | Mean | SE | Finished deck | ms/move |
|---|---|---|---|---|
| Original heuristic (scores streak bonuses, doesn't plan for them) | 6587 | 50 | 95.2% | 10 |
| Retrained value network + exact endgame | 8247 | 49 | 95.8% | 29 |
| **+ deeper search on close moves (in the app)** | **8383** | 43 | 98.8% | 70 |

**Deeper search on close moves: +136 ± 44 per game** over the previous app strategy on the same
500 held-out decks (`opt/heldout_500_deep3.txt`), and more games reach the end of the deck
(98.8% vs 95.8%). Every move is still scored by the 2-ply search; the moves within 200 points
of the best are then re-searched one card deeper (`VF_DEEP=3`, `VF_MARGIN=200`, now the
defaults). Under the old rules this had not paid off (60 decks), but streak bonuses reward
planning several cards ahead. On tuning decks it was +80 ± 44 (seeds 0–399) and +91 ± 52
(seeds 400–799); margin 100 (+83, 77 ms/move) and 5 kept replies (+86) were indistinguishable.
Compared with that (tuning decks, paired, same rules), these did not help:
- **Exact endgame from 6 cards left**: +12 ± 11, 10x slower. A packed-integer rewrite of the
  solver gave identical values but was only 1.2x faster (the cost is the number of states, not
  their representation), so the endgame stays at 5.
- **Fine-tuning the network on expected 2-ply targets** (20 iterations of 2048 self-play games
  from the installed weights, `opt/cand_valuefn/weights_expected.npz`, log in
  `train_expected.log`): −4 ± 39; self-play stayed flat at ~7450. The network has plateaued.

Paired difference **+1660 ± 66** (`opt/heldout_500_streak.txt`). Retraining alone was worth
+420 ± 59 over the old streak-blind network (300 tuning decks). The exact endgame now starts at
5 cards left (`VF_ENDGAME`, default 5): 6 was +15 ± 13, not significant, and caused pauses
of up to ~4 s. Old weights: `opt/cand_valuefn/weights.npz`; new: `weights_streak.npz`.

Everything below was measured under the original rules (no streak bonuses).

All numbers: mean final score on **held-out decks** (seeds from 100000, never used for tuning),
same decks for every strategy. Per-game sd ≈ 400–900, so differences under ~60 are noise.

| Strategy | Mean (500 decks) | SE | Finished deck (no 3rd bust) | ms/move |
|---|---|---|---|---|
| Original 1-card lookahead heuristic | 5865 | 39 | 94.8% | 11 |
| Tuned structural heuristic | 6618 | 26 | 96.6% | 45 |
| Monte Carlo rollout | 6630 | 29 | 96.6% | 66 |
| Expectimax + fitted leaf values (200 decks) | 6304 | 55 | 87.0% | 315 |
| Learned value function | 6804 | 19 | 98.0% | 31 |
| **Learned value function + exact endgame (in the app)** | **6826** | 19 | 98.4% | ~90 avg |

Head-to-head on the same 200 decks: value function beats tuned by **+212 ± 34**;
expectimax loses to tuned by −322 ± 60. Exact endgame vs none, same 500 held-out decks:
**+21.6 ± 3.8** (`opt/heldout_500_endgame.txt`).

**Result: +961 points per game (+16%) over the original strategy**, with fewer game-ending busts
(98.4% of games reach the end of the deck vs 94.8%).

## How the winning strategy works
Expectimax over the exact remaining-deck distribution (2 card plies), with a small numpy
neural network V(state) at the leaves that estimates expected future points. V was trained by
self-play with TD(λ), iterating policy improvement. Inputs describe piles, busts and stays left,
and the composition of the remaining deck.

Once 6 or fewer cards remain after the current one, `strategy/endgame.py` replaces the network
with a memoized exact expectimax over the known remaining cards, so endgame moves are provably
optimal. `VF_ENDGAME=7` solves one card earlier (about +7 more points per game, but some moves
take several seconds).

## What was tried and did not help
- Deeper search (3 plies on close moves): 6807 vs 6857 on 60 tuning decks, 3x slower.
- More network training (8 more iterations, round-2 recipe): +20 ± 25, not significant;
  self-play plateaued near 6400 (1-ply). Round-3 partial weights: −39 ± 27.
These suggest the remaining gap to perfect play is small. Exact play over the whole game is
out of reach (the state space is far too large to solve), so this can't be proven optimal.

## Where things live
- `strategy/` — the strategy the app uses (`strategy.py`, `vf_core.py`, `endgame.py`, `weights.npz`).
  `twentyone.py` loads it and falls back to its built-in heuristic if it is missing.
- `opt/baseline.py` — the original heuristic; `opt/vf_noend.py`, `opt/vf_weights.py` — variants
  of the installed strategy for paired comparisons; `opt/vf_variant.py` — the installed strategy
  with any settings overridden, e.g. `VARIANT="DEEP=0" python3 opt/bench.py opt/vf_variant.py
  strategy/strategy.py`.
- `opt/bench.py` — paired benchmark harness (`python3 opt/bench.py A.py [B.py] --games N --start S`).
- `opt/cand_*/` — each agent's full work: strategy, training/tuning scripts, weights, logs.
  `opt/cand_valuefn/train.py` retrains the value network.
- `opt/heldout_500.txt`, `opt/heldout_200_paired.txt` — raw held-out benchmark output.

## Caveats
- The agents were stopped before finishing (usage limit), so each strategy is a snapshot of
  in-progress work. All numbers above were measured on held-out decks after they stopped.
- The value network weights are the agent's round-2 result; round-3 training was cut off
  (`opt/cand_valuefn/weights_r3_partial.npz`). More training may improve it further.
- Exact endgame solving is too slow for interactive use beyond ~7 cards left.
