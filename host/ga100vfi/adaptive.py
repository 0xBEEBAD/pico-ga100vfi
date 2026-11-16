"""v_dip planner for `campaign --adaptive`: Thompson sampling over a card model.

Model of one card, per dip voltage v (mV):
  P(dsov=0)            = sig((c1 - v) / s1)   rises as the dip deepens
  P(over | dsov=0)     = sig((c2 - v) / s2)   rises deeper still (c2 <= c1)
  P(hit)               = P(dsov=0) * (1 - P(over | dsov=0))
A cloud of particles over (c1, s1, c2, s2) is reweighted by every outcome.

Goldilocks needs a hit plus the other checks. A pass ends the campaign, so every hit
recorded here failed them: P(Goldilocks | hit) at each code is Beta(1, 1 + failed
hits). Each shot draws one particle and one such g per code, and fires the code with
the highest P(hit) * g. Codes are explored in proportion to the chance they are the
best, so no single code is hammered while its hits keep failing.

`python -m ga100vfi.sim` compares this against fixed v_dip over simulated cards.
"""
from __future__ import annotations

import math
import random

# PMBus VOUT_COMMAND LSB; matches VID_LSB_MV / mv_to_code() in firmware/pwmvid.c.
LSB_MV = 6.25
# Particle count: 500-1500 scored alike in simulation; 1000 costs ~1.4 ms per shot.
PARTICLES = 1000


def mv_to_code(mv: int) -> int:
    return int(mv / LSB_MV + 0.5)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


class Planner:
    def __init__(self, min_mv: int, max_mv: int, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        # Every code whose voltage lies in [min_mv, max_mv]; round() maps back to it.
        self.codes = list(range(math.ceil(min_mv / LSB_MV),
                                math.floor(max_mv / LSB_MV) + 1))
        if not self.codes:
            raise ValueError(f"no PMBus code between {min_mv} and {max_mv} mV")
        r = self.rng
        self.particles = []
        for _ in range(PARTICLES):
            # Prior: glitch onset anywhere in range or just above it; over onset
            # 0-25 mV deeper. Ranges are from simulation, not measurement.
            c1 = r.uniform(min_mv, max_mv + 15)
            self.particles.append((c1, r.uniform(1.5, 7.0),
                                   c1 - r.uniform(0.0, 25.0), r.uniform(1.0, 6.0)))
        self.weights = [1.0 / PARTICLES] * PARTICLES
        self.failed_hits = dict.fromkeys(self.codes, 0)

    def next_mv(self) -> int:
        r = self.rng
        c1, s1, c2, s2 = r.choices(self.particles, weights=self.weights)[0]
        best, best_score = self.codes[0], -1.0
        for c in self.codes:
            v = c * LSB_MV
            score = (_sig((c1 - v) / s1) * (1.0 - _sig((c2 - v) / s2))
                     * r.betavariate(1.0, 1.0 + self.failed_hits[c]))
            if score > best_score:
                best, best_score = c, score
        return round(best * LSB_MV)

    def record(self, mv: int, outcome: str) -> None:
        """outcome: under, hit (failed the other checks), over, or lost (link down)."""
        c = mv_to_code(mv)
        v = c * LSB_MV
        if outcome == "hit":
            self.failed_hits[c] += 1
        for i, (c1, s1, c2, s2) in enumerate(self.particles):
            glitch = _sig((c1 - v) / s1)
            over = _sig((c2 - v) / s2)
            if outcome == "under":
                like = 1.0 - glitch
            elif outcome == "hit":
                like = glitch * (1.0 - over)
            else:
                like = glitch * over
            self.weights[i] *= max(like, 1e-12)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        # Resample once the cloud has collapsed onto a third of its particles. The
        # jitter keeps it able to follow a window that drifts during the campaign.
        if 1.0 / sum(w * w for w in self.weights) < len(self.particles) / 3:
            r = self.rng
            picked = r.choices(self.particles, weights=self.weights,
                               k=len(self.particles))
            self.particles = [(c1 + r.gauss(0, 0.7), max(1.0, s1 + r.gauss(0, 0.2)),
                               c2 + r.gauss(0, 0.7), max(0.8, s2 + r.gauss(0, 0.2)))
                              for c1, s1, c2, s2 in picked]
            self.weights = [1.0 / len(self.particles)] * len(self.particles)
