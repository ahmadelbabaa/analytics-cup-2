# Where runners get free: the cost of marking handovers

Entry for the [PySport Analytics Cup 2.0](https://pysport.org/analytics-cup/editions/analytics-cup2/rules) (Football, Europe edition), using SkillCorner's A-League 2024/25 open data. Theme: **Defensive Positioning**. Deadline: **18 Dec 2026**.

> Work in progress. This README currently holds the project plan; it becomes the ≤1000-word submission write-up (max 2 figures/tables) before the deadline.

## Question

When an attacker runs across the pitch, marking responsibility has to pass from one defender to the next ("pass him on"). **Is the handover where runners get free, and how much space does the defence give away in it?**

## Evidence so far (kill tests, `kill_tests/`)

| Test | Result |
|---|---|
| Enough data? | 1,947 clean handovers during off-ball runs (runner + both defenders on camera), 69–120 per match, no single-team skew |
| Is separation concentrated in handovers? | The handover window holds **+15 pp** more of the runner's separation gain than a placebo window (95% match-bootstrap CI +13 to +17) |
| Is it just geometry (crossing zones)? | No: against a "hold your place in the block" baseline, geometry explains ≈0 of it; defender movement explains +15 pp |
| Headline | Defenders normally take **~1.0 m** of space from a runner by tracking him; during a handover that falls to **~0.3 m** |
| Does one costly handover predict a reception? | **Not shown** (p = 0.64). We make a mechanism claim, not a per-event predictor |

Ideas tested and dropped: option survival ("The Countdown": defender effect too flat) and extrapolation error from re-entry jumps (SkillCorner smooths re-entries, no signal).

## Method (reusable, `handover/`)

1. `load_match(dir)`: SkillCorner open-data match → positions, detection flags, possession.
2. `assign_markers(match)`: per frame, optimal 1-to-1 matching of outfield attackers to defenders (Hungarian), **sticky** by 2 m so pairings only change when another defender is clearly closer (A→B→A ping-pong 13% → 1%).
3. `detect_handovers(match, markers)`: marker changes between two pairings held ≥0.5 s, new marker within 5 m. ~10 per minute of possession (≈1 per attacker per minute); flags whether all three players were on camera.
4. `tracking_benefit(match, attacker, frames)`: actual separation vs a "hold your place in the block" baseline; `benefit` = metres of space the defence removed by moving.

## Plan

| Weeks | Work |
|---|---|
| 1–2 | Re-run the kill-test findings with the new detector (sticky Hungarian, not nearest-defender) on all 20 matches; robustness: leave-Auckland-out, detected-only, window size, stickiness |
| 3–4 | Break the handover cost down by situation (not team): block type, pitch zone, run type, defender roles (CB→FB, DM→CB…), handover direction |
| 5 | Try a better outcome link: runner separation at ball arrival / xThreat of the pass instead of "received" |
| 6 | Refine the baseline (per-line shift instead of rigid block); goal-side weighting in the marker cost |
| 7 | Figure 1: animated/frozen handover with the "hold" ghost defenders; Figure 2: benefit curve around the switch |
| 8 | Package polish (docstrings, tests, `pip install`), README write-up, 1-minute video |

Constraints: situation-level analysis only (no team rankings, Auckland is in 7 of 20 matches); match-clustered uncertainty everywhere; only players detected on camera in headline numbers.

## Setup

```bash
pip install -r requirements.txt
git lfs install
git clone --depth 1 https://github.com/SkillCorner/opendata.git data/skillcorner   # ~2.2 GB incl. LFS tracking
PYTHONPATH=. python tests/test_handover.py
```

```python
from handover import load_match, assign_markers, detect_handovers
m = load_match("data/skillcorner/data/matches/1874553")
handovers = detect_handovers(m, assign_markers(m))
```
