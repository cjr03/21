#!/usr/bin/env python3
"""21 calculator: enter cards one at a time and it places each one in the best pile.

Rules
  - 4 piles, up to 5 cards each. Standard 52-card deck (4 of each rank).
  - J/Q/K = 10, A = 1 or 11 (whichever helps).
  - Pile hits exactly 21:   400 pts (1000 pts if it's the 5th card). Pile clears.
  - Pile reaches 5 cards under 21: 800 pts. Pile clears.
  - Pile goes over 21: bust, pile clears. The 3rd bust ends the game.
  - 2 stays per game: a stay clears one pile before the card lands.
  - Streak: cards in a row that each score (21 or 5-card stack): 2nd +250, 3rd +500, 4th +1000
    (+1750 total), then the streak restarts.
    A card that doesn't score (including a bust) or a stay resets the streak.
  - Game ends after all 52 cards or on the 3rd bust.
  - Perfect bonus: finish all 52 cards without a single bust: +1000.
"""

import os
import re
import select
import shutil
import sys
import time
from collections import Counter

NUM_PILES = 4
MAX_CARDS = 5
MAX_BUSTS = 3
START_STAYS = 2
DECK_SIZE = 52
COPIES_PER_RANK = 4
STREAK_LEN = 4
STREAK_BONUS = (0, 0, 250, 500, 1000)  # bonus for the Nth scoring card in a row (+1750 for all 4)
PERFECT_BONUS = 1000   # finishing the deck with no busts

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
RANK_VALUE = {r: (1 if r == "a" else 10 if r in "jqk" else int(r)) for r in RANKS}
VALUE_LABEL = {1: "A", 10: "10/J/Q/K", **{v: str(v) for v in range(2, 10)}}

# Heuristic weights used by the placement search.
BUST_COST = 450        # cost of using up one of the two "free" busts
FINAL_BUST_COST = 100000  # third bust ends the game
STAY_COST = 200        # stays are a limited resource, don't spend them casually
LOOKAHEAD_WEIGHT = 0.9


# --------------------------------------------------------------------------- piles

def totals(pile):
    """Return (hard, best) totals. Hard counts every ace as 1; best counts one ace as 11 if it fits."""
    hard = sum(pile)
    best = hard + 10 if 1 in pile and hard + 10 <= 21 else hard
    return hard, best


def resolve(pile):
    """Score a pile after a card lands. Returns (points, busted, cleared, event_text)."""
    hard, best = totals(pile)
    if hard > 21:
        return 0, True, True, f"BUST ({hard})"
    if best == 21:
        pts = 1000 if len(pile) == MAX_CARDS else 400
        label = "5-card 21!" if len(pile) == MAX_CARDS else "21!"
        return pts, False, True, f"{label} +{pts}"
    if len(pile) == MAX_CARDS:
        return 800, False, True, f"5-card stack ({best}) +800"
    return 0, False, False, ""


def streak_step(streak, pts):
    """Advance the scoring streak after a card. Returns (new_streak, bonus)."""
    if not pts:
        return 0, 0
    streak += 1
    return (0 if streak == STREAK_LEN else streak), STREAK_BONUS[streak]


# --------------------------------------------------------------------------- game state

class Game:
    def __init__(self):
        self.piles = [[] for _ in range(NUM_PILES)]
        self.pile_cards = [[] for _ in range(NUM_PILES)]  # rank labels for display
        self.score = 0
        self.busts = 0
        # Stays left. The engine keeps the `holds` / `use_hold` names because strategy/ and opt/
        # (and the trained networks' inputs) depend on them; the player only ever sees "stay".
        self.holds = START_STAYS
        self.streak = 0  # consecutive scoring cards (0..STREAK_LEN-1)
        self.seen = Counter()  # rank -> times seen this game
        self.history = []  # snapshots for undo

    # ---- helpers
    def cards_seen(self):
        return sum(self.seen.values())

    def remaining_ranks(self):
        """Counter of ranks still left in the deck."""
        return Counter({r: COPIES_PER_RANK - self.seen[r] for r in RANKS})

    def remaining_values(self):
        """Counter of card values still left in the deck (J/Q/K/10 all count as 10)."""
        c = Counter()
        for r, n in self.remaining_ranks().items():
            c[RANK_VALUE[r]] += n
        return c

    def over(self):
        return self.busts >= MAX_BUSTS or self.cards_seen() >= DECK_SIZE

    def snapshot(self):
        self.history.append((
            [p[:] for p in self.piles], [p[:] for p in self.pile_cards],
            self.score, self.busts, self.holds, self.streak, self.seen.copy(),
        ))

    def undo(self):
        if not self.history:
            return False
        (self.piles, self.pile_cards, self.score,
         self.busts, self.holds, self.streak, self.seen) = self.history.pop()
        return True

    # ---- actions
    def stay(self, idx):
        self.piles[idx] = []
        self.pile_cards[idx] = []
        self.holds -= 1
        self.streak = 0

    def place(self, rank, idx, use_hold=False):
        """Place a card. Returns a list of event strings."""
        events = []
        if use_hold:
            self.stay(idx)
            events.append(f"Stay on pile {idx + 1}: pile cleared first")
        self.seen[rank] += 1
        self.piles[idx].append(RANK_VALUE[rank])
        self.pile_cards[idx].append(rank)
        pts, busted, cleared, text = resolve(self.piles[idx])
        self.score += pts
        if busted:
            self.busts += 1
        if text:
            events.append(f"Pile {idx + 1}: {text}")
        self.streak, bonus = streak_step(self.streak, pts)
        if bonus:
            self.score += bonus
            events.append(f"STREAK! {self.streak or STREAK_LEN} scoring cards in a row +{bonus}")
        if cleared:
            self.piles[idx] = []
            self.pile_cards[idx] = []
        if self.busts == 0 and self.cards_seen() == DECK_SIZE:
            self.score += PERFECT_BONUS
            events.append(f"PERFECT GAME! No busts +{PERFECT_BONUS}")
        return events


# --------------------------------------------------------------------------- strategy

def options(piles, holds):
    """All legal moves: (pile index, use_hold)."""
    opts = [(i, False) for i in range(NUM_PILES)]
    if holds > 0:
        opts += [(i, True) for i in range(NUM_PILES) if piles[i]]
    return opts


def apply(piles, busts, holds, streak, value, idx, use_hold):
    """Pure simulation of one move. Returns (new_piles, new_busts, new_holds, new_streak, reward)."""
    piles = [p[:] for p in piles]
    reward = 0.0
    if use_hold:
        piles[idx] = []
        holds -= 1
        streak = 0
        reward -= STAY_COST
    piles[idx].append(value)
    pts, busted, cleared, _ = resolve(piles[idx])
    streak, bonus = streak_step(streak, pts)
    reward += pts + bonus
    if busted:
        busts += 1
        reward -= FINAL_BUST_COST if busts >= MAX_BUSTS else BUST_COST
    if cleared:
        piles[idx] = []
    return piles, busts, holds, streak, reward


def pile_potential(pile, dist, total_left):
    """Rough value of a pile's position: expected payoff of the next card landing on it."""
    if not pile or total_left == 0:
        return 0.0
    hard, best = totals(pile)
    n = len(pile)
    ev = 0.0
    for v, cnt in dist.items():
        if cnt <= 0:
            continue
        pts, busted, _, _ = resolve(pile + [v])
        ev += cnt / total_left * (pts - (150 if busted else 0))
    # Small penalty for high, inflexible totals that clog up a pile.
    if best >= 17 and 1 not in pile:
        ev -= (best - 16) * 15
    # Small bonus for piles that are building toward a 5-card stack with room to spare.
    if n >= 3 and hard <= 21 - (MAX_CARDS - n):
        ev += 60 * (n - 2)
    return ev


def static_value(piles, dist):
    total_left = sum(dist.values())
    return sum(pile_potential(p, dist, total_left) for p in piles)


def rank_moves(game, rank):
    """Score every legal move for this card, best first.

    Each move is scored as immediate reward + expected value of the best reply to the next card,
    averaged over the cards actually left in the deck. Returns [(score, idx, use_hold), ...].
    """
    value = RANK_VALUE[rank]
    dist = game.remaining_values()
    dist[value] -= 1  # this card is no longer in the deck
    total_left = sum(dist.values())

    ranked = []
    for idx, use_hold in options(game.piles, game.holds):
        piles, busts, holds, streak, reward = apply(game.piles, game.busts, game.holds, game.streak,
                                                    value, idx, use_hold)
        score = reward
        if busts < MAX_BUSTS and total_left > 0:
            exp = 0.0
            for v, cnt in dist.items():
                if cnt <= 0:
                    continue
                d2 = dist.copy()
                d2[v] -= 1
                nxt = max(
                    r2 + static_value(p2, d2)
                    for p2, b2, h2, s2, r2 in (apply(piles, busts, holds, streak, v, i, h)
                                           for i, h in options(piles, holds))
                )
                exp += cnt / total_left * nxt
            score += LOOKAHEAD_WEIGHT * exp
        ranked.append((score, idx, use_hold))
    ranked.sort(key=lambda m: -m[0])
    return ranked


_heuristic_rank_moves = rank_moves


def rank_moves(game, rank):
    """Use the learned value-function strategy in ./strategy; fall back to the heuristic above."""
    global _tuned
    if _tuned is None:
        try:
            import importlib.util
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy", "strategy.py")
            spec = importlib.util.spec_from_file_location("tuned_strategy", path)
            _tuned = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_tuned)
        except Exception:
            _tuned = False
    return _tuned.rank_moves(game, rank) if _tuned else _heuristic_rank_moves(game, rank)


_tuned = None


def best_move(game, rank):
    _, idx, use_hold = rank_moves(game, rank)[0]
    return idx, use_hold


def pile_odds(pile, dist, total_left):
    """Chance the next card makes 21 / a 5-card stack / busts, if placed on this pile."""
    if total_left == 0:
        return 0.0, 0.0, 0.0
    p21 = p5 = pbust = 0
    for v, cnt in dist.items():
        if cnt <= 0:
            continue
        pts, busted, _, _ = resolve(pile + [v])
        if busted:
            pbust += cnt
        elif pts in (400, 1000):
            p21 += cnt
        elif pts == 800:
            p5 += cnt
    return p21 / total_left, p5 / total_left, pbust / total_left


def forced_bust_chance(piles, dist, total_left):
    """Chance the next card busts on EVERY pile (a stay or a bust would be forced)."""
    if total_left == 0:
        return 0.0
    bad = sum(cnt for v, cnt in dist.items()
              if cnt > 0 and all(totals(p + [v])[0] > 21 for p in piles))
    return bad / total_left




# --------------------------------------------------------------------------- CLI

TTY = sys.stdout.isatty() and sys.stdin.isatty()
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
ANIM = TTY and os.environ.get("TWENTYONE_ANIM", "1") != "0"  # toggled in-game with 'anim'
UNICODE = (sys.stdout.encoding or "").lower().replace("-", "").startswith("utf")
ANSI = re.compile(r"\033\[[0-9;]*m")
MIN_W, MIN_H = 78, 21  # the dashboard fills the terminal, but never gets smaller than this

# Casino palette (xterm-256), used as dashboard roles. Gold is the accent (the recommendation,
# the data, the meters), cream is text, red is danger and orange is caution. The page is the
# table's felt, panels are darker felt wells, and the header and footer are its leather rail.
PAGE, PANEL, RAIL, TRACK = 28, 22, 52, 28
TEXT, SUB, MUTED, LABEL = 230, 187, 108, 178
GOLD, ORANGE, RED = 220, 214, 210
CREAM, INK, CARD_RED = 231, 16, 160

G = dict(bar=" ▁▂▃▄▅▆▇█", meter="━", here="▸", none="·", play="▶", stay="◆", ok="✓", bust="✕",
         up="▲", warn="⚠", live="●", end="■", spark="▁▂▃▄▅▆▇█", arrow="›", cursor="▏",
         suits="♠♥♦♣") \
    if UNICODE else \
    dict(bar=" .:-=+*#%", meter="=", here=">", none="-", play=">", stay="*", ok="+", bust="x",
         up="^", warn="!", live="*", end="#", spark=".:-=+*#%", arrow=">", cursor="_",
         suits="SHDC")


# ---- ink: inline codes restore the surrounding background, so text can sit on any surface

def ink(text, color):
    return f"\033[38;5;{color}m{text}\033[38;5;{TEXT}m" if COLOR else text


def bold(text):
    return f"\033[1m{text}\033[22m" if COLOR else text


def sub(text):
    return ink(text, SUB)


def muted(text):
    return ink(text, MUTED)


def gold(text):
    return ink(text, GOLD)


def on(text, bg, fg=TEXT, back=PANEL):
    """Text on its own background (a chip, a highlight), then back to the surrounding one."""
    if not COLOR:
        return text
    return f"\033[48;5;{bg};38;5;{fg}m{text}\033[48;5;{back};38;5;{TEXT}m"


def visible(text):
    """Length as shown on screen (color codes take no space)."""
    return len(ANSI.sub("", text))


def pad(text, width, align="<"):
    gap = max(0, width - visible(text))
    if align == "^":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return " " * gap + text if align == ">" else text + " " * gap


def fit(lines, height):
    """Exactly `height` lines: extra lines dropped, short sections padded with blanks."""
    return list(lines)[:height] + [""] * max(0, height - len(lines))


def wrap(lines, width):
    """Word-wrap styled lines to a visible width; continuation lines keep the line's indent."""
    out = []
    for line in lines:
        if visible(line) <= width:
            out.append(line)
            continue
        plain = ANSI.sub("", line)
        indent = " " * (len(plain) - len(plain.lstrip(" ")))
        cur = ""
        for word in line.split(" "):
            if cur and visible(cur) + 1 + visible(word) > width:
                out.append(cur)
                cur = indent + word
            else:
                cur = word if not cur else cur + " " + word
        out.append(cur)
    return out


def chip(rank, style="face", back=PANEL):
    """A card as a chip. face: white card stock, aces and court cards in red ink; hot: glowing
    gold as it lands; ghost: the advisor's preview of where it would go; gone: just the rank."""
    label = pretty(rank)
    if not COLOR:
        return f"{label:>2} " if style != "ghost" else f"<{label}>"
    if style == "gone":
        return muted(f"{label:^3}")
    paper, fg = {"face": (CREAM, CARD_RED if rank in "ajqk" else INK), "hot": (GOLD, INK),
                 "ghost": (RAIL, GOLD), "ghost2": (PANEL, GOLD), "bust": (RED, INK)}[style]
    return on(bold(f"{label:^3}"), paper, fg, back)


def meter(fraction, width, fill=GOLD):
    """A bar meter: the filled part in the fill color, the rest as a dim felt track."""
    n = max(0, min(width, round(fraction * width)))
    if not COLOR:
        return "#" * n + "-" * (width - n)
    return ink(G["meter"] * n, fill) + ink(G["meter"] * (width - n), TRACK)


def sparkline(values, width):
    """The last `width` values as a sparkline: history in sage, the latest point in gold."""
    vals = values[-width:]
    if len(vals) < 2:
        return muted(G["none"])
    lo, hi = min(vals), max(vals)
    steps = G["spark"]
    marks = [steps[0 if hi == lo else round((v - lo) / (hi - lo) * (len(steps) - 1))] for v in vals]
    return muted("".join(marks[:-1])) + gold(marks[-1])


# ---- layout: the dashboard fills the terminal

class Layout:
    def __init__(self, width, height):
        self.w, self.h = max(MIN_W, width), max(MIN_H, height)
        self.left = max(46, (self.w - 2) * 3 // 5)
        self.right = self.w - 2 - self.left
        rows = self.h - 9  # header, gap, 3 tile rows, gap, gap, gap, footer
        self.top = rows // 2
        self.bottom = rows - self.top
        self.tile = (self.w - 8) // 5


def layout():
    if not TTY:
        return Layout(MIN_W, MIN_H)
    size = shutil.get_terminal_size((MIN_W, MIN_H + 1))
    return Layout(size.columns, size.lines)


def panel(title, lines, width, height, note=""):
    """A dashboard panel: a dark felt well with a gold title row (and a note on its right)."""
    inner = width - 2
    head = bold(gold(title))
    if note:
        head = pad(head, inner - visible(note)) + note
    body = [head] + fit(wrap(lines, inner), height - 1)
    return [paint(f" {pad(line, inner)} ", PANEL) for line in body]


def paint(text, bg):
    return f"\033[48;5;{bg};38;5;{TEXT}m{text}\033[48;5;{PAGE}m" if COLOR else text


def side_by_side(left, right):
    return [a + paint("  ", PAGE) + b for a, b in zip(left, right)]


def bar_line(text, width):
    """A full-width line on the leather rail (header and footer)."""
    return paint(pad(text, width), RAIL)


# ---- screen: drawn in place, only the lines that changed

class Screen:
    def __init__(self):
        self.last = []

    def show(self, frame, force=False):
        lines = frame.split("\n")
        if not sys.stdout.isatty():
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            return
        out = []
        if force or len(lines) != len(self.last):
            out.append("\033[H\033[2J")
            self.last = [None] * len(lines)
        for i, line in enumerate(lines):
            if line != self.last[i]:
                out.append(f"\033[{i + 1};1H{line}\033[0m\033[K")
        if out:
            sys.stdout.write("".join(out))
            sys.stdout.flush()
        self.last = lines


SCREEN = Screen()


def play(frames):
    """Run an animation: an iterable of (frame, seconds). Skipped when animation is off."""
    if not ANIM:
        return
    for frame, delay in frames:
        SCREEN.show(frame)
        time.sleep(delay)


def ease(t):
    return 1 - (1 - t) ** 3


# ---- message lines (the recommendation panel)

def say(text):
    return text


def detail(text):
    return f"  {text}"


def error(text):
    return f"{ink(G['warn'], ORANGE)} {text}"


def parse_card(text):
    t = text.strip().lower()
    if t == "1":
        t = "a"
    return t if t in RANKS else None


def pretty(rank):
    return rank.upper()


def pct(x):
    return f"{x * 100:.0f}%" if x >= 0.005 else G["none"]


def describe_move(idx, use_hold):
    return f"stay+pile {idx + 1}" if use_hold else f"pile {idx + 1}"


PILE_EVENT = re.compile(r"Pile (\d): (.*)")
STREAK_EVENT = re.compile(r"STREAK! (\d) .*\+(\d+)")


def event_line(text):
    """An engine event as a status line: icon + label, points in bold gold."""
    m = STREAK_EVENT.match(text)
    if m:
        return f"{gold(G['up'])} Streak x{m[1]}  {bold(gold('+' + m[2]))}"
    m = PILE_EVENT.match(text)
    if not m:
        return text
    pile, what = m[1], m[2]
    if what.startswith("BUST"):
        return f"{ink(G['bust'], RED)} Bust {what[5:]} on pile {pile}"
    pts = bold(gold("+" + what.rsplit("+", 1)[-1]))
    if what.startswith("5-card 21"):
        return f"{gold(G['ok'])} 5-card 21 on pile {pile}  {pts}"
    if what.startswith("21"):
        return f"{gold(G['ok'])} 21 on pile {pile}  {pts}"
    return f"{gold(G['ok'])} 5-card stack, pile {pile}  {pts}"


# ---- dashboard sections

def header(game, L, fx):
    """The top rail: the title (a glint drifts across it) and the live status."""
    title = "TWENTY-ONE"
    k = fx.get("tick")
    glint = None if k is None else (k // 2) % (len(title) + 12)
    lit = "".join(ink(ch, CREAM if i == glint else GOLD) for i, ch in enumerate(title))
    suits = G["suits"]
    left = (f" {gold(suits[0])} {ink(suits[1], RED)} {bold(lit)} {ink(suits[2], RED)} {gold(suits[3])}"
            f"   {sub('placement advisor')}")
    if game.over():
        status = sub(f"{G['end']} Game over")
    else:
        status = f"{gold(G['live'])} {sub('Live')}"
    return bar_line(left + " " * max(1, L.w - visible(left) - visible(status) - 1) + status, L.w)


def tiles(game, L, fx):
    """The KPI row: five stat tiles, label over value over a meter or sparkline."""
    score = fx.get("score", game.score)
    left = fx.get("shoe", DECK_SIZE - game.cards_seen())
    flash = fx.get("flash")
    history = [h[2] for h in game.history] + [game.score]
    last = game.score - game.history[-1][2] if game.history else 0
    delta = gold(f"+{last:,}") if last > 0 and score == game.score else ""
    bust_fill = (GOLD, ORANGE, RED, RED)[game.busts]
    nxt = muted(f"next +{STREAK_BONUS[game.streak + 1]}") if game.streak else muted("no streak")
    widths = [L.tile] * 4 + [L.w - 8 - 4 * L.tile]

    def streak_foot(w):
        if not COLOR:
            return nxt
        bars = gold(G["bar"][-1] * game.streak) + ink(G["bar"][-1] * (STREAK_LEN - game.streak), TRACK)
        if game.streak and 5 + visible(nxt) > w:  # narrow tile: just the bonus
            return bars + " " + muted(f"+{STREAK_BONUS[game.streak + 1]}")
        return bars + " " + nxt
    cells = [
        ("Score", lambda w: pad(bold(gold(f"{score:,}")), w - visible(delta)) + delta,
         lambda w: sparkline(history, w)),
        ("Cards left", lambda w: bold(str(left)) + muted(f" of {DECK_SIZE}"),
         lambda w: meter(left / DECK_SIZE, w)),
        ("Busts", lambda w: bold(str(game.busts)) + muted(f" of {MAX_BUSTS}")
         + (" " + ink(G["warn"], RED) if game.busts == MAX_BUSTS - 1 else ""),
         lambda w: meter(game.busts / MAX_BUSTS, w, bust_fill)),
        ("Stays", lambda w: bold(str(game.holds)) + muted(f" of {START_STAYS}"),
         lambda w: meter(game.holds / START_STAYS, w)),
        ("Streak", lambda w: bold(str(game.streak)) + muted(f" of {STREAK_LEN}"), lambda w: streak_foot(w)),
    ]
    rows = [[], [], []]
    for (name, value, foot), w in zip(cells, widths):
        bg = RAIL if flash == name else PANEL
        if flash == "Busts" and name == "Busts":
            bg = RED if fx.get("pulse", 0) % 2 == 0 else RAIL
        inner = w - 2
        for r, text in enumerate((ink(name, LABEL), value(inner), foot(inner))):
            rows[r].append(f"\033[48;5;{bg};38;5;{TEXT}m {pad(text, inner)} \033[48;5;{PAGE}m" if COLOR
                           else f" {pad(text, inner)} ")
    gap = paint("  ", PAGE)
    return [gap.join(r) for r in rows]


def pile_lines(game, L, height, fx, focus):
    """The piles as a table: cards, total, the chance the next card scores, and the bust risk."""
    inner = L.left - 2
    extra = inner - 44
    cols = ((3, "<"), (20 + extra // 2, "<"), (7, ">"), (6, ">"), (8 + extra - extra // 2, ">"))
    risk_w = cols[4][0] - 5

    def row(cells):
        return "".join(pad(c, w, a) for c, (w, a) in zip(cells, cols))

    dist = game.remaining_values()
    total_left = sum(dist.values())
    landing, cleared, ghost, sweep = fx.get("landing"), fx.get("cleared"), fx.get("ghost"), fx.get("sweep")
    out = [muted(row(["", "Cards", "Total", "Score", "Bust"]))]
    for i, pile in enumerate(game.piles):
        mark = gold(G["here"]) if i == focus else " "
        label = f"{mark}{bold(gold(str(i + 1))) if i == focus else sub(str(i + 1))} "
        if landing and landing[0] == i:
            _, rank, use_hold, slide = landing
            ranks = [] if use_hold else game.pile_cards[i]
            values = [RANK_VALUE[c] for c in ranks] + [RANK_VALUE[rank]]
            hard, best = totals(values)
            tot = "" if slide else (ink("BUST", RED) if hard > 21 else gold("21") if best == 21
                                    else str(best) if best == hard else f"soft {best}")
            base = " ".join(chip(c) for c in ranks)
            room = cols[1][0] - visible(base) - 4
            cards = base + (" " if ranks else "") + " " * max(0, min(slide, room)) + chip(rank, "hot")
            out.append(row([label, cards, bold(tot), "", ""]))
            continue
        if sweep is not None and sweep == i:
            out.append(row([label, " ".join(chip(c, "gone") for c in game.pile_cards[i]), "",
                            "", f"{gold(G['stay'])} {sub('stay')}"]))
            continue
        if cleared and cleared[0] == i:
            _, ranks, kind = cleared
            if kind == "bust":
                cards, note = " ".join(chip(c, "bust") for c in ranks), f"{ink(G['bust'], RED)} {sub('bust')}"
            else:
                cards, note = " ".join(chip(c, "hot") for c in ranks), f"{gold(G['ok'])} {sub('scored')}"
            out.append(row([label, cards, "", "", note]))
            continue
        hard, best = totals(pile)
        tot = "" if not pile else str(best) if best == hard else f"soft {best}"
        p21, p5, pb = pile_odds(pile, dist, total_left)
        ps = p21 + (p5 if len(pile) == MAX_CARDS - 1 else 0)
        score_cell = bold(gold(pct(ps))) if ps >= 0.25 else sub(pct(ps)) if pile else muted(G["none"])
        if pile:
            fill = RED if pb >= 0.5 else ORANGE if pb >= 0.25 else GOLD
            risk = " " + meter(pb, risk_w, fill) + " " + pad(sub(pct(pb)), 3, ">")
        else:
            risk = muted(G["none"])
        cards = " ".join(chip(c) for c in game.pile_cards[i])
        if ghost and ghost[0] == i:  # the preview: where the card being typed would go
            _, rank, use_hold, k = ghost
            style = "ghost" if k % 4 < 2 else "ghost2"
            cards = (" ".join(chip(c, "gone") for c in game.pile_cards[i]) + "  " if use_hold and cards
                     else cards + (" " if cards else "")) + chip(rank, style)
        elif not cards:
            cards = muted(G["none"])
        out.append(row([label, cards, tot, score_cell, risk]))
    # spread the rows out when the panel has room
    spare = height - len(out)
    if spare >= len(game.piles):
        spaced = [out[0]]
        for line in out[1:]:
            spaced += [line, ""]
        return spaced
    return out


VALUE_AXIS = {1: "A", 10: "10s", **{v: str(v) for v in range(2, 10)}}


def shoe_probs(game):
    dist = game.remaining_values()
    n = sum(dist.values())
    return [dist[v] / n if n else 0.0 for v in range(1, 11)]


def shoe_lines(game, L, height, fx):
    """The next-card distribution as a column chart by value, with the odds and cards left of each.
    Mid-animation the bars tween from the previous distribution (fx tween = (old, t))."""
    dist = game.remaining_values()
    if not sum(dist.values()):
        return [muted("The shoe is empty.")]
    probs = shoe_probs(game)
    if fx.get("tween"):
        old, t = fx["tween"]
        probs = [a + (b - a) * t for a, b in zip(old, probs)]
    grow = fx.get("grow", 1.0)
    inner = L.left - 2
    cw = max(4, (inner - 4) // 10)
    lw = inner - 10 * cw
    bw = max(2, cw - 2)
    rows = max(2, height - 1 - 3)
    top = max(probs) or 1
    levels = [round(p / max(top, 0.0001) * 8 * rows * grow) for p in probs]
    bars = G["bar"]
    out = []
    for r in range(rows - 1, -1, -1):
        cells = []
        for lvl in levels:
            part = max(0, min(8, lvl - 8 * r))
            cells.append(pad(gold(bars[part] * bw) if part else (muted(G["none"]) if r == 0 and not lvl else ""),
                             cw, ">"))
        out.append(" " * lw + "".join(cells))
    shown = max(probs)
    out.append(" " * lw + "".join(pad(sub(VALUE_AXIS[v]), cw, ">") for v in range(1, 11)))
    out.append(pad(muted("Odds"), lw) + "".join(
        pad(bold(gold(pct(p))) if p == shown else sub(pct(p)), cw, ">") for p in probs))
    out.append(pad(muted("Left"), lw) + "".join(
        pad(sub(str(dist[v])) if dist[v] else muted(G["none"]), cw, ">") for v in range(1, 11)))
    return out


def shoe_note(game):
    dist = game.remaining_values()
    total_left = sum(dist.values())
    fb = forced_bust_chance(game.piles, dist, total_left) if total_left else 0
    if fb > 0:
        return f"{ink(G['warn'], ORANGE)} {sub(f'{fb * 100:.0f}% fits no pile')}"
    return muted("10s = 10 J Q K")


def ranking_lines(ranking, L, height, grow=1.0):
    """The move ranking as an emphasis bar chart: the best move in gold, the rest in sage.
    The player's own pick, when it isn't the best, is marked with a pointer and drawn in cream.
    The bars grow in (grow 0..1) when a new ranking arrives."""
    if not ranking:
        return [muted("How the other moves compare"), muted("appears here after each card.")]
    rows, hidden, chosen = ranking
    inner = L.right - 2
    bw = max(8, inner - 21)
    shown = rows[:max(1, height - 2)]
    best, worst = shown[0][1], shown[-1][1]
    lo = worst - max(1.0, (best - worst) * 0.2)
    out = []
    for label, score, is_best, moves in shown:
        frac = max(1 / bw, (score - lo) / max(1e-9, best - lo)) * grow
        mark = gold(G["here"]) if chosen in moves else " "
        if is_best:
            out.append(mark + pad(bold(label), 12) + meter(frac, bw) + pad(bold(gold("best")), 8, ">"))
        elif chosen in moves:
            out.append(mark + pad(bold(label), 12) + meter(frac, bw, TEXT)
                       + pad(bold(f"{score - best:+.0f}"), 8, ">"))
        else:
            out.append(mark + pad(sub(label), 12) + meter(frac, bw, MUTED)
                       + pad(muted(f"{score - best:+.0f}"), 8, ">"))
    more = hidden + len(rows) - len(shown)
    if more:
        out.append(muted(f"+{more} worse options"))
    return out


def help_lines():
    def cmd(name, text):
        return pad(bold(gold(name)), 16) + sub(text)
    return [
        cmd("a 2-10 j q k", "Type the card you drew (1 = A): a preview shows where it goes."),
        cmd("", "Arrow keys pick another pile; Enter places it, Esc clears."),
        cmd("stay N", f"Clear pile N now (N = 1-{NUM_PILES}). Uses one of your {START_STAYS} stays."),
        cmd("undo", "Take back the last card or stay."),
        cmd("new", "Start a new game."),
        cmd("rules", "Show the scoring rules."),
        cmd("anim", f"Turn animations {'off' if ANIM else 'on'}."),
        cmd("help, ?", "Show this list."),
        cmd("quit", "End the game."),
    ]


def rules_lines():
    def rule(name, text, icon=None):
        return f"{icon or gold(G['ok'])} " + pad(bold(gold(name)), 14) + sub(text)
    return [
        sub(f"{NUM_PILES} piles, up to {MAX_CARDS} cards each. J/Q/K = 10, A = 1 or 11."),
        "",
        rule("21", "Exactly 21 pays +400 (+1000 on the 5th card). Pile clears."),
        rule("5-card stack", f"{MAX_CARDS} cards under 21 pays +800. The pile clears."),
        rule("Streak", "Scoring cards in a row pay 2nd +250, 3rd +500, 4th +1000,", gold(G["up"])),
        rule("", "then restart. A non-scoring card or a stay resets it.", " "),
        rule("Bust", "Over 21 clears the pile. The 3rd bust ends the game.", ink(G["bust"], RED)),
        rule("Stay", f"{START_STAYS} per game: clear a pile yourself, or the advisor does.", gold(G["stay"])),
        "",
        sub(f"The game ends when all {DECK_SIZE} cards are dealt or on the 3rd bust."),
    ]


def footer(game, L, fx):
    """The bottom rail: the input field (when typing goes straight to the dashboard) and the keys."""
    def key(text):
        return on(f" {text} ", CREAM, INK, RAIL) if COLOR else f"[{text}]"
    if game.over():
        keys = [key("new"), key("undo"), key("Enter") + sub(" finish")]
    elif fx.get("choosing"):
        arrows = "↑↓" if UNICODE else "up/dn"
        keys = [key(arrows) + sub(" choose pile"), key("Enter") + sub(" place"), key("Esc") + sub(" clear")]
    else:
        keys = [key("stay N")] if game.holds and any(game.piles) else []
        if game.history:
            keys.append(key("undo"))
        keys += [key("new"), key("help"), key("rules"), key("quit")]
    right = "  ".join(keys) + " "
    left = ""
    if "typed" in fx:
        k = fx.get("tick", 0)
        cursor = gold(G["cursor"]) if k % 6 < 4 else " "
        label = "> " if game.over() else f"Card {G['arrow']} "
        left = f" {bold(gold(label))}{bold(fx['typed'])}{cursor}"
    return bar_line(left + " " * max(1, L.w - visible(left) - visible(right)) + right, L.w)


def recommendation(game, message_lines, fx, L, height):
    msg = list(message_lines or [sub("Type the card you drew."), muted("Type help for commands.")])
    if game.over():
        msg = [f"{muted(G['end'])} {bold('Game over')} {muted(end_reason(game))}",
               sub("Final score ") + bold(gold(f"{fx.get('score', game.score):,}")), ""] + msg[:2]
    msg = wrap(msg, L.right - 2)
    if fx.get("toast") and msg and COLOR:  # highlight the whole first line, chips included
        msg[0] = on(pad(msg[0], L.right - 2).replace(f"48;5;{PANEL};", f"48;5;{RAIL};"), RAIL)
    return msg


def render(game, message_lines, ranking, focus=None, fx=None):
    """The whole dashboard as one string, filling the terminal.

    message_lines may be a (title, lines) pair: a panel (help, rules) that spans both panel rows.
    fx overrides what's drawn mid-animation or while typing: score / shoe (counters), grow (charts
    0..1), tween (old next-card odds, t), flash / pulse (a KPI tile to light), landing / cleared /
    sweep (a pile mid-deal, just cleared, or being stayed), ghost (the typed card's preview),
    toast (highlight the first message line), typed (the input field), tick (idle time)."""
    fx = fx or {}
    L = layout()
    lines = [header(game, L, fx), paint(" " * L.w, PAGE)]
    lines += tiles(game, L, fx)
    lines.append(paint(" " * L.w, PAGE))
    if isinstance(message_lines, tuple):
        title, body = message_lines
        lines += panel(title, body, L.w, L.top + 1 + L.bottom)
    else:
        piles = panel("Piles", pile_lines(game, L, L.top - 1, fx, focus), L.left, L.top)
        advice = panel("Recommendation", recommendation(game, message_lines, fx, L, L.top - 1), L.right, L.top)
        shoe = panel("Next card", shoe_lines(game, L, L.bottom, fx), L.left, L.bottom, shoe_note(game))
        rank = panel("Move ranking", ranking_lines(ranking, L, L.bottom - 1, fx.get("rgrow", 1.0)),
                     L.right, L.bottom, muted("vs. best"))
        lines += side_by_side(piles, advice)
        lines.append(paint(" " * L.w, PAGE))
        lines += side_by_side(shoe, rank)
    lines += [paint(" " * L.w, PAGE), footer(game, L, fx)]
    return "\n".join(f"\033[48;5;{PAGE};38;5;{TEXT}m{pad(line, L.w)}\033[0m" if COLOR else pad(line, L.w)
                     for line in lines)


def end_reason(game):
    if game.busts >= MAX_BUSTS:
        return f"{MAX_BUSTS} busts"
    return "all cards dealt" if game.cards_seen() >= DECK_SIZE else "ended early"


def analysis_for(ranked, chosen=None):
    """The move ranking: ([(label, score, is_best, moves)], how many more moves were left out,
    the player's chosen move (pile index, use_hold) or None).

    Moves with the same score (e.g. several empty piles) are equivalent, so they share one row.
    """
    groups = []  # [score, [moves]]
    for score, idx, use_hold in ranked:
        if groups and abs(groups[-1][0] - score) < 1e-6:
            groups[-1][1].append((idx, use_hold))
        else:
            groups.append([score, [(idx, use_hold)]])

    def label(moves):
        parts = []
        for prefix, stay in (("pile ", False), ("stay+", True)):
            piles = [str(i + 1) for i, h in moves if h == stay]
            if piles:
                parts.append(prefix + "/".join(piles))
        return " or ".join(parts)

    shown = groups[:8]
    rows = [(label(m), s, n == 0, m) for n, (s, m) in enumerate(shown)]
    return rows, sum(len(m) for _, m in groups[len(shown):]), chosen


# ---- animations (each a generator of (frame, seconds) for play())

def shuffle_frames(game):
    """A new game: the shoe counts up and the next-card chart grows into place."""
    steps = 12
    for k in range(1, steps + 1):
        t = ease(k / steps)
        msg = [sub("Shuffling a fresh shoe" + "." * (k % 4))]
        yield render(game, msg, None, fx=dict(shoe=round(DECK_SIZE * t), grow=t, tick=k)), 0.03


def sweep_frames(game, idx, msg):
    """A stay: the pile's cards fade out before it clears."""
    yield render(game, msg, None, idx, dict(sweep=idx, flash="Stays")), 0.16


def deal_frames(game, rank, idx, use_hold, call):
    """The card on its way: a stay clears the pile first, then the card slides in and glows."""
    msg = [call]
    if use_hold:
        yield from sweep_frames(game, idx, msg)
    for slide in (14, 8, 4, 1, 0):
        yield render(game, msg, None, idx, dict(landing=(idx, rank, use_hold, slide), toast=True)), 0.025
    yield render(game, msg, None, idx, dict(landing=(idx, rank, use_hold, 0), toast=True)), 0.12


def payout_frames(game, before, events, landed, message, ranking, focus, old_probs):
    """After the card: the score counts up, the next-card chart tweens to the new odds and the
    ranking bars grow in. A cleared pile glows gold (scored) or red (bust) before it empties;
    a bust also pulses the Busts tile."""
    gained = game.score - before
    bust = any("BUST" in e for e in events)
    cleared = (focus, landed, "bust" if bust else "scored") if landed else None
    steps = 10
    for k in range(1, steps + 1):
        t = ease(k / steps)
        fx = dict(tween=(old_probs, t), rgrow=t, toast=k < steps, cleared=cleared if k < steps else None,
                  flash="Busts" if bust else "Score" if gained else None, pulse=k // 2)
        if gained:
            fx["score"] = before + round(gained * t)
        yield render(game, message, ranking, focus, fx), 0.035


# ---- input: typed straight into the dashboard, with a live preview of each card

class Keys:
    """Raw key input (no echo, no line buffering) for the life of the session."""

    def __enter__(self):
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self, timeout):
        """The keys pressed within `timeout` seconds ('' if none)."""
        ready, _, _ = select.select([self.fd], [], [], timeout)
        return os.read(self.fd, 64).decode(errors="ignore") if ready else ""


def where(idx, use_hold):
    return f"stay, then pile {idx + 1}" if use_hold else f"pile {idx + 1}"


def preview(game, typed, cache, pick=0):
    """What the dashboard shows while a command is being typed: (fx, message, ranking, focus,
    chosen move), or None.

    A card shows where it would go: the advisor's choice, or with pick > 0 the pick-th next best
    legal move (chosen with the arrow keys). Moves are computed once per card and kept in `cache`,
    so pressing Enter places it without searching again. 'stay N' shows the pile it clears."""
    text = typed.strip().lower()
    rank = parse_card(text) if text else None
    if rank and not game.over() and game.seen[rank] < COPIES_PER_RANK:
        if rank not in cache:
            cache[rank] = rank_moves(game, rank)
        moves = cache[rank]
        best, (score, idx, use_hold) = moves[0], moves[min(pick, len(moves) - 1)]
        msg = [f"{gold(G['play'])} {chip(rank)} goes on {bold(gold(where(idx, use_hold)))}"]
        if (idx, use_hold) == (best[1], best[2]):
            msg.append(muted("The advisor's pick. Arrows choose another pile."))
        else:
            msg.append(f"{ink(G['warn'], ORANGE)} {sub('Your pick:')} {bold(f'{score - best[0]:+.0f}')} "
                       f"{sub('vs the advisor')}")
            msg.append(muted(f"Advisor: {where(best[1], best[2])}"))
        fx = dict(ghost=(idx, rank, use_hold), choosing=True)
        return fx, msg, analysis_for(moves, (idx, use_hold)), idx, (idx, use_hold)
    m = re.fullmatch(r"stay\s+(\d)", text)
    if m and 1 <= int(m[1]) <= NUM_PILES and game.holds and game.piles[int(m[1]) - 1]:
        i = int(m[1]) - 1
        msg = [f"{gold(G['stay'])} Stay clears pile {i + 1}", muted("Enter to confirm.")]
        return dict(sweep=i), msg, None, i, None
    return None


ARROWS = {"\033[A": -1, "\033[D": -1, "\033OA": -1, "\033OD": -1,   # up / left: a better move
          "\033[B": 1, "\033[C": 1, "\033OB": 1, "\033OC": 1}       # down / right: the next one


def read_command(game, keys, message, ranking, focus, cache):
    """Edit a command in the footer's input field, previewing it live. While a card is typed,
    the arrow keys step through the legal moves, best first. Returns (command, chosen move) on
    Enter; the chosen move is None unless a card preview was showing."""
    typed, pick, k = "", 0, 0
    while True:
        p = preview(game, typed, cache, pick)
        if p:
            fx, msg, rk, fc, chosen = p
            if "ghost" in fx:
                fx["ghost"] += (k,)
                pick = min(pick, len(cache[parse_card(typed)]) - 1)
            fx.update(typed=typed, tick=k)
            SCREEN.show(render(game, msg, rk, fc, fx))
        else:
            chosen = None
            SCREEN.show(render(game, message, ranking, focus, dict(typed=typed, tick=k)))
        got = keys.read(0.1 if ANIM else None)
        if not got:
            k += 1
            continue
        i = 0
        while i < len(got):
            seq = next((s for s in ARROWS if got.startswith(s, i)), None)
            if seq:
                if parse_card(typed):  # only a typed card has moves to choose between
                    pick = max(0, pick + ARROWS[seq])
                i += len(seq)
                continue
            if got.startswith("\033[", i) or got.startswith("\033O", i):  # other keys: skip them
                i += 3
                continue
            ch = got[i]
            i += 1
            if ch in "\r\n":  # the move as chosen now, including arrows that came with the Enter
                now = preview(game, typed, cache, pick)
                return typed, now[4] if now else None
            if ch in ("\x03", "\x04"):
                raise EOFError
            if ch in ("\x7f", "\b"):
                typed, pick = typed[:-1], 0
            elif ch == "\033":
                typed, pick = "", 0
            elif ch.isprintable() and len(typed) < 16:
                typed, pick = typed + ch, 0
        k = 0


def main():
    interactive = TTY
    if interactive:
        sys.stdout.write("\033[?1049h\033[?25l\033[2J")  # full screen, hidden cursor; restored on exit
    game = Game()
    try:
        if interactive:
            with Keys() as keys:
                play(shuffle_frames(game))
                game = session(game, keys)
        else:
            game = session(game, None)
    finally:
        if interactive:
            sys.stdout.write("\033[0m\033[?25h\033[?1049l")
            sys.stdout.flush()
    print(f"Twenty-One: final score {game.score:,} ({end_reason(game)}).")


def session(game, keys):
    global ANIM
    message, ranking, focus = None, None, None
    actions = []  # description of each undoable action, parallel to game.history
    confirm_new = False  # 'new' mid-game must be typed twice
    cache = {}  # rank -> ranked moves for the current position (filled by the live preview)

    while True:
        chosen = None  # a move the player picked with the arrow keys, if any
        try:
            if keys:
                line, chosen = read_command(game, keys, message, ranking, focus, cache)
                line = line.strip().lower()
            else:
                SCREEN.show(render(game, message, ranking, focus))
                line = input("> " if game.over() else f"Card {G['arrow']} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        pending_new, confirm_new = confirm_new, False
        message, ranking, focus = None, None, None

        if line in ("help", "?", "h"):
            message = ("Commands", help_lines())
            continue
        if line == "rules":
            message = ("Rules", rules_lines())
            continue
        if line == "anim":
            ANIM = TTY and not ANIM
            message = [say(f"Animations {'on' if ANIM else 'off'}.") if TTY
                       else error("Animations need a terminal.")]
            continue
        if line == "new":
            if game.over() or not game.history or pending_new:
                previous = game.score
                game, actions, cache = Game(), [], {}
                play(shuffle_frames(game))
                message = [say(f"New game. Last game scored {bold(gold(f'{previous:,}'))}." if previous
                               else "New game. Type the first card.")]
            else:
                confirm_new = True
                message = [error(f"Type new again to abandon this game (score {game.score:,}).")]
            continue
        if line == "undo":
            if game.undo():
                cache = {}
                message = [say(f"Undone: {actions.pop()}")]
            else:
                message = [error("Nothing to undo.")]
            continue
        if game.over() or line in ("quit", "exit"):
            break  # once the game has ended, anything but new/undo/help/rules exits
        if not line:
            continue

        if line.split()[0] == "stay":
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit() or not 1 <= int(parts[1]) <= NUM_PILES:
                message = [error(f"Usage: stay N (N = 1-{NUM_PILES})")]
            elif game.holds == 0:
                message = [error("No stays left.")]
            elif not game.piles[int(parts[1]) - 1]:
                message = [error(f"Pile {parts[1]} is already empty.")]
            else:
                focus = int(parts[1]) - 1
                message = [f"{gold(G['stay'])} Stay: pile {parts[1]} cleared"]
                play(sweep_frames(game, focus, message))
                actions.append(f"stay on pile {parts[1]}")
                game.snapshot()
                game.stay(focus)
                cache = {}
                message.append(detail(muted(f"{game.holds} stay{'s' if game.holds != 1 else ''} left, "
                                            "streak reset")))
            continue

        rank = parse_card(line)
        if rank is None:
            message = [error(f"'{line}' isn't a card or command. Type help.")]
            continue
        if game.seen[rank] >= COPIES_PER_RANK:
            message = [error(f"All {COPIES_PER_RANK} {pretty(rank)}s are already played.")]
            continue

        ranked = cache.get(rank) or rank_moves(game, rank)
        best_score, best_idx, best_hold = ranked[0]
        idx, use_hold = chosen if chosen in {(i, h) for _, i, h in ranked} else (best_idx, best_hold)
        overridden = (idx, use_hold) != (best_idx, best_hold)
        call = f"{gold(G['play'])} Place {chip(rank)} on {bold(gold(f'pile {idx + 1}'))}"
        play(deal_frames(game, rank, idx, use_hold, call))

        before, old_probs = game.score, shoe_probs(game)
        landed = ([] if use_hold else game.pile_cards[idx][:]) + [rank]
        actions.append(f"{pretty(rank)} placed on {describe_move(idx, use_hold)}")
        game.snapshot()
        events = game.place(rank, idx, use_hold)
        cache = {}
        focus = idx
        message = [call] + ([detail(f"{gold(G['stay'])} stay first: pile cleared")] if use_hold else []) \
            + [detail(event_line(e)) for e in events if not e.startswith("Stay")]
        if overridden:
            loss = next(s for s, i, h in ranked if (i, h) == (idx, use_hold)) - best_score
            message.append(detail(muted(f"Your pick ({loss:+.0f}); advisor: {where(best_idx, best_hold)}")))
        ranking = analysis_for(ranked, (idx, use_hold) if overridden else None) if len(ranked) > 1 else None
        cleared = landed if any(e.startswith(f"Pile {idx + 1}:") for e in events) else None
        play(payout_frames(game, before, events, cleared, message, ranking, focus, old_probs))

    return game


if __name__ == "__main__":
    main()
