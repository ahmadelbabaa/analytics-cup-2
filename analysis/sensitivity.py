"""Does the headline survive the thresholds?

Headline: during a marking handover the defence protects less space than during ordinary marking.
Metric: space leaked = mean(actual - hold) separation over a 4 s window (see handover.tracking_benefit).
Handovers are compared with control moments: the same attackers, marked within max_distance by a held marker,
with no handover within 4 s. Grid over stickiness, marking distance and minimum hold; match-bootstrap CIs.

Run from the repo root:  PYTHONPATH=. python analysis/sensitivity.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from handover import assign_markers, detect_handovers, load_match, tracking_benefit
from handover.core import _held_blocks

DATA = Path("data/skillcorner/data/matches")
OUT = Path("data/sensitivity_events.csv")
W = 20
GRID = dict(sticky=[1.0, 2.0, 3.0], max_distance=[3.0, 5.0, 7.0], min_hold=[3, 5, 10])
rng = np.random.default_rng(0)


def leaked(match, attacker, centre):
    fs = list(range(centre - W, centre + W + 1))
    if not all(f in match.frames and attacker in match.frames[f] for f in fs):
        return np.nan
    b = tracking_benefit(match, attacker, fs)
    return -b.benefit.mean() if len(b) else np.nan


def controls(markers, handovers, n, max_distance, min_hold):
    """n random (attacker, frame) moments marked within max_distance by a held marker, no handover within 2W."""
    busy = {(a, f + k) for a, f in zip(handovers.attacker, handovers.frame) for k in range(-2 * W, 2 * W + 1)}
    cand = []
    for att, g in markers.groupby("attacker"):
        for blk in _held_blocks(g, min_hold).itertuples():
            cand += [(att, f) for f in range(blk.start + W, blk.end - W, W)]
    dist = markers.set_index(["attacker", "frame"]).distance
    cand = [c for c in cand if c not in busy and dist.get(c, np.inf) <= max_distance]
    idx = rng.choice(len(cand), min(n, len(cand)), replace=False) if cand else []
    return [cand[i] for i in idx]


def run():
    rows = []
    for d in sorted(DATA.iterdir()):
        m = load_match(d)
        for sticky in GRID["sticky"]:
            mk = assign_markers(m, sticky=sticky)
            for md, mh in itertools.product(GRID["max_distance"], GRID["min_hold"]):
                h = detect_handovers(m, mk, max_distance=md, min_hold=mh)
                h = h[h.detected]
                key = dict(match=m.match_id, sticky=sticky, max_distance=md, min_hold=mh)
                rows += [dict(key, kind="handover", leaked=leaked(m, a, f)) for a, f in zip(h.attacker, h.frame)]
                rows += [dict(key, kind="control", leaked=leaked(m, a, f))
                         for a, f in controls(mk, h, len(h), md, mh)]
        print(m.match_id, len(rows), flush=True)
    df = pd.DataFrame(rows).dropna()
    df.to_csv(OUT, index=False)
    return df


def summarise(df):
    out = []
    for key, g in df.groupby(["sticky", "max_distance", "min_hold"]):
        per = g.groupby(["match", "kind"]).leaked.agg(["sum", "count"]).unstack("kind")
        diffs = []
        for _ in range(1000):
            s = per.sample(len(per), replace=True, random_state=int(rng.integers(1e9)))
            diffs.append(s[("sum", "handover")].sum() / s[("count", "handover")].sum()
                         - s[("sum", "control")].sum() / s[("count", "control")].sum())
        h, c = g[g.kind == "handover"].leaked, g[g.kind == "control"].leaked
        out.append(dict(zip(["sticky", "max_distance", "min_hold"], key), n_handovers=len(h),
                        leaked_handover=h.mean(), leaked_control=c.mean(), diff=h.mean() - c.mean(),
                        ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5)))
    return pd.DataFrame(out).round(3)


if __name__ == "__main__":
    df = pd.read_csv(OUT) if "--cached" in sys.argv and OUT.exists() else run()
    s = summarise(df)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print(f"\ncombos where handovers leak more than control (CI above 0): {(s.ci_low > 0).sum()} / {len(s)}")
