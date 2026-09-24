"""Kill test for "Marking handovers": during off-ball runs, how often does marking responsibility switch
between defenders, and does a sloppy switch (runner left unmarked) come before the runner being found?
"""
import numpy as np
import pandas as pd

from common import load_match, match_dirs

MIN_HOLD = 5   # frames a defender must stay "marker" to count (0.5 s), filters flicker
WINDOW = 10    # frames either side of the switch used to measure the coverage gap


def marker_sequence(frames, runner, defenders, f0, f1):
    """Per frame of the run: (frame, nearest defender, distance, runner detected, defender detected)."""
    seq = []
    for f in range(f0, f1 + 1):
        pos = frames.get(f)
        if not pos or runner not in pos:
            continue
        rx, ry, rdet = pos[runner]
        ds = [(np.hypot(pos[d][0] - rx, pos[d][1] - ry), d) for d in defenders if d in pos]
        if ds:
            dist, d = min(ds)
            seq.append((f, d, dist, rdet, pos[d][2]))
    return seq


def handovers(seq):
    """Switch points between two markers that each hold for >= MIN_HOLD frames (run-length smoothed)."""
    blocks = []  # [marker, first_idx, last_idx]
    for i, (_, d, *_rest) in enumerate(seq):
        if blocks and blocks[-1][0] == d:
            blocks[-1][2] = i
        else:
            blocks.append([d, i, i])
    blocks = [b for b in blocks if b[2] - b[1] + 1 >= MIN_HOLD]
    return [(a[0], b[0], b[1]) for a, b in zip(blocks, blocks[1:]) if a[0] != b[0]]


def main():
    rows = []
    for d in match_dirs():
        mid, ev, team_of, frames = load_match(d)
        runs = ev[ev.event_type == "off_ball_run"]
        for r in runs.itertuples():
            runner = int(r.player_id)
            defenders = [p for p, t in team_of.items() if t != team_of.get(runner)]
            seq = marker_sequence(frames, runner, defenders, int(r.frame_start), int(r.frame_end))
            if len(seq) < 2 * MIN_HOLD:
                continue
            hs = handovers(seq)
            row = dict(match=mid, subtype=r.event_subtype, targeted=r.targeted, received=r.received,
                       dangerous=r.dangerous, n_handovers=len(hs),
                       max_sep=max(s[2] for s in seq), clean_run=all(s[3] and s[4] for s in seq))
            if hs:
                old, new, i = hs[0]
                w = seq[max(0, i - WINDOW): i + WINDOW + 1]
                row.update(gap=max(s[2] for s in w) - seq[max(0, i - WINDOW)][2],  # separation opened during switch
                           gap_abs=max(s[2] for s in w),
                           clean_handover=all(s[3] and s[4] for s in w))
            rows.append(row)
        print(mid, len(rows), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("../data/kill_test_handovers.csv", index=False)
    h = df[df.n_handovers > 0]
    print(f"\nruns analysed: {len(df)}; with >=1 handover: {len(h)} ({len(h) / len(df):.0%}); "
          f"clean handovers (runner + both markers detected in window): {h.clean_handover.sum()}")
    print("\nreceived / targeted rate, handover vs none:")
    print(df.groupby(df.n_handovers > 0)[["targeted", "received", "dangerous"]].mean().round(3))

    c = h[h.clean_handover].copy()
    c["gap_q"] = pd.qcut(c.gap, 4, labels=["tight", "q2", "q3", "sloppy"])
    print("\nclean handovers by separation opened during the switch:")
    print(c.groupby("gap_q", observed=True).agg(gap_m=("gap", "median"), targeted=("targeted", "mean"),
                                                received=("received", "mean"), n=("received", "size")).round(3))
    # does the handover gap matter beyond how separated the runner gets overall?
    import statsmodels.formula.api as smf
    c["received_i"] = c.received.astype(int)
    m = smf.logit("received_i ~ gap + max_sep", data=c).fit(disp=0, cov_type="cluster", cov_kwds={"groups": c.match})
    print("\nlogit received ~ gap + max_sep (match-clustered SEs):")
    print(m.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(3))
    print("\nhandovers per match (clean):", c.groupby("match").size().describe().round(1).to_dict())


if __name__ == "__main__":
    main()
