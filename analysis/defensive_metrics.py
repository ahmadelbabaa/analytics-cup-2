"""Prototype: defensive quality of handovers across all 20 matches (detected handovers only).

- timing: early release by A / late pick-up by B, at a tight 3 m marking radius
- shape: is the zone B vacates covered? vacated gap vs the block's normal spacing
  (distance from each defender to his nearest outfield teammate at the same frames).

Run from the repo root:  PYTHONPATH=. python analysis/defensive_metrics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from handover import assign_markers, detect_handovers, handover_metrics, load_match

DATA = Path("data/skillcorner/data/matches")
RADIUS = 3.0


def normal_spacing(match, frames, team):
    """Median distance from each outfield defender to his nearest outfield teammate over the given frames."""
    out = []
    for f in frames:
        ps = np.array([match.frames[f][p][:2] for p in match.frames[f]
                       if match.team_of.get(p) not in (team, None) and p not in match.goalkeepers])
        if len(ps) > 1:
            dd = np.linalg.norm(ps[:, None] - ps[None], axis=2)
            np.fill_diagonal(dd, np.inf)
            out.append(np.median(dd.min(1)))
    return np.median(out) if out else np.nan


def main():
    rows = []
    for d in sorted(DATA.iterdir()):
        m = load_match(d)
        h = detect_handovers(m, assign_markers(m))
        q = handover_metrics(m, h[h.detected], radius=RADIUS)
        q["spacing_m"] = [normal_spacing(m, [f + 20], m.team_of[a]) for f, a in zip(q.frame, q.attacker)]
        rows.append(q)
        print(m.match_id, len(q), flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv("data/defensive_metrics.csv", index=False)

    print(f"\n{len(df)} detected handovers, {df.match_id.nunique()} matches")
    print("\ntiming at 3 m marking radius:")
    print(f"  runner unmarked at some point: {(df.free_s > 0).mean():.0%}, mean {df.free_s.mean():.2f} s")
    print(f"  A released early: {(df.early_release_s > 0).mean():.0%}  |  B picked up late: {(df.late_pickup_s > 0).mean():.0%}")
    both = (df.early_release_s > 0) & (df.late_pickup_s > 0)
    print(f"  both: {both.mean():.0%}  |  clean (neither): {((df.early_release_s == 0) & (df.late_pickup_s == 0)).mean():.0%}")
    print("  free time by fault: ", df.assign(
        fault=np.select([both, df.early_release_s > 0, df.late_pickup_s > 0], ["both", "early release", "late pick-up"], "clean")
    ).groupby("fault").agg(n=("free_s", "size"), free_s=("free_s", "mean"), leaked_m=("space_leaked_m", "mean")).round(2).to_string())

    ratio = df.vacated_gap_m / df.spacing_m
    print("\nshape: vacated gap vs normal nearest-teammate spacing")
    print(f"  median vacated gap {df.vacated_gap_m.median():.1f} m, median normal spacing {df.spacing_m.median():.1f} m")
    print(f"  zone left open (gap > 1.5x normal spacing): {(ratio > 1.5).mean():.0%}")
    print("  space leaked on the runner, by whether B's zone was covered:")
    print(df.assign(zone=np.where(ratio > 1.5, "left open", "covered")).groupby("zone").agg(
        n=("space_leaked_m", "size"), leaked_m=("space_leaked_m", "mean"), free_s=("free_s", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
