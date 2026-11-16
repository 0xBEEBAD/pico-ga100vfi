# Adaptive calibration (`campaign --adaptive`)

The best `v_dip` differs between boards (MP2888B VRM tolerance, die leakage). With
`--adaptive`, each attempt is classified and the next `v_dip` is picked from every
outcome so far (`host/ga100vfi/adaptive.py`):

| Outcome | Meaning |
|---|---|
| under | DSOV=1: dip too shallow |
| over | DSOV=0 with bad CRC, or PCIe link lost: dip too deep |
| hit | DSOV=0, CRC=0x55, but another Goldilocks check failed |

- A model of the card is fitted as a particle cloud: P(DSOV=0) and P(over | DSOV=0),
  each a sigmoid that rises as the dip deepens.
- Each PMBus code (6.25 mV) counts its failed hits; its P(Goldilocks | hit) is
  Beta(1, 1 + failed hits).
- Each shot fires the code with the highest P(hit) × P(Goldilocks | hit) under one
  random draw (Thompson sampling). A code whose hits keep failing loses share to its
  neighbours.

The output JSON adds `spots`: under/hit/over/lost/goldilocks counts per `v_dip` fired.

| Flag | Default | Meaning |
|---|---|---|
| `--adaptive` | off | Pick `v_dip` per shot; `--v-dip` is ignored |
| `--min-v` | 300 | Safety floor (mV) |
| `--max-v` | 380 | Safety ceiling (mV) |

The first shots can land anywhere in `--min-v`..`--max-v`, including the floor.

## Hardware run

2026-11-16: `campaign --adaptive --budget 200` → Goldilocks on attempt 8, `v_dip` 356,
3.2 s.

| `v_dip` (mV) | 325 | 331 | 344 | 350 | 356 |
|---|---|---|---|---|---|
| Outcome | lost | under | lost, lost | lost | under, under, Goldilocks |

The `ours` scenario below predicts hits at 344–350 mV; this card lost the link there.

## Simulation

`python -m ga100vfi.sim [cards]` runs the planner on simulated cards (no hardware).
`ours` is fitted to the measured card: 14 % Goldilocks per shot at 354 mV,
over-glitched at ~335 mV (`docs/PROTOCOL.md`). The other scenarios and the planner's
prior ranges are guesses, not measurements. 400 cards per scenario; each cell is
P(Goldilocks) % within 100 / 500 attempts, then mean attempts (a miss counts as 500).
`fixed` = `--v-dip 354`; `oracle` = each card's best code.

| Scenario | adaptive | fixed | oracle |
|---|---|---|---|
| ours | 100 / 100, 6 | 100 / 100, 7 | 100 / 100, 2 |
| random | 100 / 100, 4 | 72 / 81, 124 | 100 / 100, 2 |
| narrow | 100 / 100, 7 | 72 / 87, 115 | 100 / 100, 3 |
| edge | 100 / 100, 4 | 39 / 46, 296 | 100 / 100, 2 |
| drift | 100 / 100, 4 | 71 / 82, 125 | 100 / 100, 2 |
| jump | 100 / 100, 4 | 80 / 88, 85 | 100 / 100, 2 |
| flaky | 100 / 100, 4 | 69 / 80, 133 | 100 / 100, 2 |
| bumpy | 100 / 100, 7 | 66 / 74, 160 | 100 / 100, 3 |
| low_g | 95 / 100, 26 | 51 / 64, 223 | 99 / 100, 14 |
| rare | 75 / 98, 81 | 35 / 54, 290 | 88 / 100, 42 |
| deep_g | 100 / 100, 8 | 58 / 71, 184 | 100 / 100, 3 |
| dead_codes | 97 / 99, 18 | 33 / 38, 326 | 100 / 100, 5 |
| dead_rare | 66 / 88, 127 | 20 / 28, 387 | 88 / 97, 55 |
| needle | 65 / 91, 124 | 12 / 13, 440 | 94 / 100, 27 |
| bimodal | 99 / 100, 12 | 64 / 80, 151 | 100 / 100, 2 |
