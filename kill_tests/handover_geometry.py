"""Geometry check for handover_mechanism.py: is the separation spike at a handover defending, or just geometry?

Crossing from one defender's zone to the next makes nearest-defender distance peak at the boundary even if
nobody makes a mistake. Baseline "hold" = every defender keeps his position relative to the block
(position at window start + the block centroid's shift since then). Geometry is whatever the runner gets
against that baseline; handover cost = actual peak separation - baseline peak separation.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import load_match, match_dirs
from handovers import MIN_HOLD, WINDOW, handovers, marker_sequence

rng = np.random.default_rng(0)


def separations(frames, runner, defenders, fs):
    """Actual and 'hold' nearest-defender distance for frames fs (defenders fixed to the block from fs[0])."""
    f0 = frames[fs[0]]
    ds = [d for d in defenders if d in f0 and all(d in frames[f] for f in fs)]
    p0 = np.array([f0[d][:2] for d in ds])
    act, hold = [], []
    for f in fs:
        p = np.array([frames[f][d][:2] for d in ds])
        r = np.array(frames[f][runner][:2])
        held = p0 + (p.mean(0) - p0.mean(0))
        act.append(np.linalg.norm(p - r, axis=1).min())
        hold.append(np.linalg.norm(held - r, axis=1).min())
    return np.array(act), np.array(hold)


def excess_gain(sep, lo, hi):
    gain = np.clip(np.diff(sep), 0, None)
    return gain[lo:hi - 1].sum() / gain.sum() - (hi - lo) / len(sep) if gain.sum() > 0 else np.nan


def main():
    rows, rel_pos, pool, results = [], [], [], []
    for d in match_dirs():
        mid, ev, team_of, frames = load_match(d)
        for r in ev[ev.event_type == "off_ball_run"].itertuples():
            runner = int(r.player_id)
            defenders = [p for p, t in team_of.items() if t != team_of.get(runner)]
            seq = marker_sequence(frames, runner, defenders, int(r.frame_start), int(r.frame_end))
            if len(seq) < 2 * MIN_HOLD + 2 * WINDOW:
                continue
            fs = [s[0] for s in seq]
            if fs[-1] - fs[0] != len(fs) - 1:           # need contiguous frames
                continue
            base = dict(match=mid, received=int(bool(r.received)), dangerous=r.dangerous)
            hs = handovers(seq)
            if hs:
                i = hs[0][2]
                lo, hi = max(0, i - WINDOW), min(len(fs), i + WINDOW + 1)
                if not all(s[3] and s[4] for s in seq[lo:hi]):
                    continue
                rows.append((base, "handover", fs, lo, hi, runner, defenders, frames))
                rel_pos.append(i / len(fs))
            elif all(s[3] and s[4] for s in seq):
                pool.append((base, fs, runner, defenders, frames))
        # evaluate now so frames of this match can be freed
        out = []
        for base, kind, fs, lo, hi, runner, defenders, fr in rows:
            out.append(evaluate(base, kind, fs, lo, hi, runner, defenders, fr))
        for base, fs, runner, defenders, fr in pool:
            i = int(rng.choice(rel_pos) * len(fs)) if rel_pos else len(fs) // 2
            lo, hi = max(0, i - WINDOW), min(len(fs), i + WINDOW + 1)
            out.append(evaluate(base, "placebo", fs, lo, hi, runner, defenders, fr))
        rows, pool = [], []
        results += out
        print(mid, len(results), flush=True)

    df = pd.DataFrame(results).dropna()
    df.to_csv("../data/kill_test_handover_geometry.csv", index=False)
    g = df.groupby("kind")
    print("\nexcess share of separation gain inside the window (actual vs hold baseline):")
    print(g[["excess_actual", "excess_hold"]].mean().round(3))
    h, p = df[df.kind == "handover"], df[df.kind == "placebo"]
    tot = (h.excess_actual.mean() - p.excess_actual.mean())
    geo = (h.excess_hold.mean() - p.excess_hold.mean())
    print(f"handover effect {tot:+.3f} = geometry {geo:+.3f} + defending {tot - geo:+.3f}")

    print("\npeak separation in window (m): actual, hold baseline, cost = actual - hold")
    print(g[["peak_actual", "peak_hold", "cost"]].describe().loc[:, (slice(None), ["mean", "50%"])].round(2))
    print(f"handovers with cost > 1 m: {(h.cost > 1).mean():.0%}, < -1 m (defenders beat holding): {(h.cost < -1).mean():.0%}")

    m = smf.logit("received ~ cost + peak_hold", data=h).fit(disp=0, cov_type="cluster", cov_kwds={"groups": h.match})
    print("\nlogit received ~ cost + peak_hold (handover runs, match-clustered SEs):")
    print(m.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(3))
    h = h.assign(cq=pd.qcut(h.cost, 4, labels=["defenders better", "q2", "q3", "costly"]))
    print(h.groupby("cq", observed=True).agg(cost=("cost", "median"), received=("received", "mean"), n=("received", "size")).round(3))


def evaluate(base, kind, fs, lo, hi, runner, defenders, frames):
    act, hold = separations(frames, runner, defenders, fs[lo:hi])
    full_act = np.array([min(np.hypot(frames[f][d][0] - frames[f][runner][0], frames[f][d][1] - frames[f][runner][1])
                             for d in defenders if d in frames[f]) for f in fs])
    full_hold = full_act.copy()
    full_hold[lo:hi] = hold
    return dict(base, kind=kind, excess_actual=excess_gain(full_act, lo, hi), excess_hold=excess_gain(full_hold, lo, hi),
                peak_actual=act.max(), peak_hold=hold.max(), cost=act.max() - hold.max())


if __name__ == "__main__":
    main()
