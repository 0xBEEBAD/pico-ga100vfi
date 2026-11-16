"""Simulated cards for tuning `campaign --adaptive`: python -m ga100vfi.sim [cards].

Each scenario draws cards from a family; every card is run once with the adaptive
Planner, once at a fixed --v-dip 354, and once at its best code (the oracle, which
knows the card). Prints P(Goldilocks within 100 / 500 attempts) and mean attempts
(a miss counts as 500). No hardware is touched.

Card model: per code, P(dsov=0) = sig((c1 - v)/s1); among those, P(over) =
sig((c2 - v)/s2) and among overs P(lost link) = sig((c3 - v)/s3); a hit passes the
other checks with probability g(code). "ours" is fitted to the one measured card:
14 % Goldilocks per shot at 354 mV, over-glitched (CRC=0x00) at ~335 mV
(README "Measured waveform", docs/PROTOCOL.md). The other families are guesses at
how other cards could differ; their ranges are not measured.
"""
from __future__ import annotations

import math
import random
import sys
from multiprocessing import Pool

from .adaptive import LSB_MV, Planner, mv_to_code

MIN_V, MAX_V, FIXED_V, BUDGET = 300, 380, 354, 500
CODES = list(range(math.ceil(MIN_V / LSB_MV), math.floor(MAX_V / LSB_MV) + 1))


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


class SimCard:
    def __init__(self, c1, s1, c2, s2, c3, s3, g=lambda c: 1.0, drift_sd=0.0,
                 jump=None, nack=0.0, jitter=None):
        self.c1, self.s1, self.c2, self.s2, self.c3, self.s3 = c1, s1, c2, s2, c3, s3
        self.g, self.drift_sd, self.jump, self.nack = g, drift_sd, jump, nack
        self.jitter = jitter or {}  # code -> extra P(hit survives), a bumpy window
        self.t = 0

    def p_gold(self, c: int) -> float:
        v = c * LSB_MV
        return (_sig((self.c1 - v) / self.s1) * (1 - _sig((self.c2 - v) / self.s2))
                * self.g(c) * self.jitter.get(c, 1.0))

    def shoot(self, mv: int, rng: random.Random) -> str:
        self.t += 1
        shift = rng.gauss(0, self.drift_sd) if self.drift_sd else 0.0
        if self.jump and self.t == self.jump[0]:
            shift += self.jump[1]
        self.c1 += shift; self.c2 += shift; self.c3 += shift
        if rng.random() < self.nack:
            return "nack"
        c = mv_to_code(mv)
        v = c * LSB_MV
        if rng.random() >= _sig((self.c1 - v) / self.s1):
            return "under"
        if rng.random() < _sig((self.c2 - v) / self.s2):
            return "lost" if rng.random() < _sig((self.c3 - v) / self.s3) else "over"
        if rng.random() >= self.jitter.get(c, 1.0):
            return "under"
        return "goldilocks" if rng.random() < self.g(c) else "hit"


class Bimodal:
    """Two windows; the shallow one rarely passes, the deep one always does."""
    def __init__(self, shallow, deep, split_mv):
        self.shallow, self.deep, self.split = shallow, deep, split_mv

    def _side(self, c):
        return self.shallow if c * LSB_MV >= self.split else self.deep

    def p_gold(self, c):
        return self._side(c).p_gold(c)

    def shoot(self, mv, rng):
        return self._side(mv_to_code(mv)).shoot(mv, rng)


def _window(r, lo=318, hi=378):
    c1 = r.uniform(lo, hi)
    c2 = c1 - r.uniform(3, 20)
    return c1, r.uniform(2, 6), c2, r.uniform(1, 5), c2 - r.uniform(0, 15), r.uniform(1, 5)


def _card(scn: str, r: random.Random):
    if scn == "ours":        # fitted to the measured card (see module docstring)
        return SimCard(349, 3.8, 343, 1.5, 330, 3)
    if scn == "random":      # window anywhere in range, every hit passes
        return SimCard(*_window(r))
    if scn == "narrow":      # over starts right behind the glitch onset
        c1 = r.uniform(318, 378); c2 = c1 - r.uniform(0, 4)
        return SimCard(c1, r.uniform(3, 6), c2, r.uniform(2, 5), c2 - r.uniform(0, 10), 3)
    if scn == "edge":        # window at the ends of [min_v, max_v]
        c1 = r.uniform(300, 312) if r.random() < 0.5 else r.uniform(372, 384)
        c2 = c1 - r.uniform(3, 20)
        return SimCard(c1, r.uniform(2, 6), c2, r.uniform(1, 5), c2 - r.uniform(0, 15), 3)
    if scn == "drift":       # window wanders through the campaign
        return SimCard(*_window(r), drift_sd=0.15)
    if scn == "jump":        # window shifts 6-15 mV once, partway through
        return SimCard(*_window(r, 325, 370),
                       jump=(r.randint(50, 200), r.choice((-1, 1)) * r.uniform(6, 15)))
    if scn == "flaky":       # 5 % I2C NACKs, most overs lose the link
        w = list(_window(r)); w[4] = w[2] + 5
        return SimCard(*w, nack=0.05)
    if scn == "bumpy":       # hit rate jumps code to code, not a smooth window
        return SimCard(*_window(r), jitter={c: r.uniform(0.1, 1.0) for c in CODES})
    if scn == "low_g":       # 3-30 % of hits pass the other checks
        gv = r.uniform(0.03, 0.3)
        return SimCard(*_window(r), g=lambda c: gv)
    if scn == "rare":        # 1-8 % of hits pass
        gv = r.uniform(0.01, 0.08)
        return SimCard(*_window(r), g=lambda c: gv)
    if scn == "deep_g":      # deep hits fail the checks: best Goldilocks is shallower
        w = _window(r); cg = w[0] - r.uniform(0, 10); sg = r.uniform(1, 4)
        return SimCard(*w, g=lambda c: _sig((c * LSB_MV - cg) / sg))
    if scn == "dead_codes":  # half the codes hit but never pass
        dead = {c for c in CODES if r.random() < 0.5}
        return SimCard(*_window(r), g=lambda c: 0.0 if c in dead else 1.0)
    if scn == "dead_rare":   # dead codes, and the live ones pass 2-15 % of hits
        dead = {c for c in CODES if r.random() < 0.5}
        gv = r.uniform(0.02, 0.15)
        return SimCard(*_window(r), g=lambda c: 0.0 if c in dead else gv)
    if scn == "needle":      # one hitting code passes 30 % of hits, the rest never
        w = _window(r)
        hitting = [c for c in CODES if SimCard(*w).p_gold(c) > 0.02] or CODES
        good = r.choice(hitting)
        return SimCard(*w, g=lambda c: 0.3 if c == good else 0.0)
    if scn == "bimodal":
        c1 = r.uniform(345, 378); c1b = r.uniform(310, c1 - 18)
        shallow = SimCard(c1, r.uniform(2, 4), c1 - r.uniform(3, 8), 2, c1 - 30, 3,
                          g=lambda c: 0.15)
        deep = SimCard(c1b, r.uniform(2, 4), c1b - r.uniform(3, 8), 2, c1b - 10, 3)
        return Bimodal(shallow, deep, c1 - 12)
    raise KeyError(scn)


SCENARIOS = ("ours", "random", "narrow", "edge", "drift", "jump", "flaky", "bumpy",
             "low_g", "rare", "deep_g", "dead_codes", "dead_rare", "needle", "bimodal")


def _run(job):
    scn, seed = job
    out = {}
    for name in ("adaptive", "fixed", "oracle"):
        card = _card(scn, random.Random(seed))
        rng = random.Random(seed * 7919 + 1)
        planner = Planner(MIN_V, MAX_V, rng=random.Random(seed + 1)) if name == "adaptive" else None
        best = round(max(CODES, key=card.p_gold) * LSB_MV)
        out[name] = None
        for t in range(1, BUDGET + 1):
            mv = planner.next_mv() if planner else (FIXED_V if name == "fixed" else best)
            outcome = card.shoot(mv, rng)
            if outcome == "goldilocks":
                out[name] = t
                break
            if planner and outcome != "nack":
                planner.record(mv, outcome)
    return scn, out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cards = int(argv[0]) if argv else 200
    jobs = [(s, 1000 * i + 17) for s in SCENARIOS for i in range(cards)]
    with Pool() as pool:
        results = pool.map(_run, jobs, chunksize=4)
    runs = {}
    for scn, out in results:
        for name, t in out.items():
            runs.setdefault((scn, name), []).append(t)
    names = ("adaptive", "fixed", "oracle")
    print(f"{cards} cards per scenario, budget {BUDGET}; "
          "P(Goldilocks) % within 100 / 500, mean attempts\n")
    print("| scenario | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for scn in SCENARIOS:
        cells = []
        for name in names:
            ts = runs[(scn, name)]
            p100 = 100 * sum(t is not None and t <= 100 for t in ts) / len(ts)
            p500 = 100 * sum(t is not None for t in ts) / len(ts)
            mean = sum(BUDGET if t is None else t for t in ts) / len(ts)
            cells.append(f"{p100:.0f} / {p500:.0f}, {mean:.0f}")
        print(f"| {scn} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
