"""Follow-up to handovers.py: is the handover WHERE runners get free?

For each off-ball run, separation s(t) = distance from runner to nearest defender. We ask how much of
the separation the runner gains (sum of positive steps of s) happens inside the handover window, compared
to what the window's share of the run's duration would give by chance. Placebo: runs without a handover,
with a window placed at the same relative position in the run.
"""
import numpy as np
import pandas as pd

from common import load_match, match_dirs
from handovers import MIN_HOLD, WINDOW, handovers, marker_sequence

rng = np.random.default_rng(0)


def window_stats(sep, lo, hi):
    """Share of separation gain and of time inside [lo, hi), and whether the run's peak separation is inside."""
    gain = np.clip(np.diff(sep), 0, None)          # gain[k] happens between frame k and k+1
    total = gain.sum()
    return dict(gain_share=gain[lo:hi - 1].sum() / total if total > 0 else np.nan,
                time_share=(hi - lo) / len(sep),
                peak_in=lo <= int(np.argmax(sep)) < hi)


def main():
    rows, rel_pos = [], []
    placebo_pool = []
    for d in match_dirs():
        mid, ev, team_of, frames = load_match(d)
        for r in ev[ev.event_type == "off_ball_run"].itertuples():
            runner = int(r.player_id)
            defenders = [p for p, t in team_of.items() if t != team_of.get(runner)]
            seq = marker_sequence(frames, runner, defenders, int(r.frame_start), int(r.frame_end))
            if len(seq) < 2 * MIN_HOLD + 2 * WINDOW:     # need room for a window inside the run
                continue
            sep = np.array([s[2] for s in seq])
            hs = handovers(seq)
            base = dict(match=mid, received=r.received, dangerous=r.dangerous, n=len(sep))
            if hs:
                i = hs[0][2]
                lo, hi = max(0, i - WINDOW), min(len(sep), i + WINDOW + 1)
                if not all(s[3] and s[4] for s in seq[lo:hi]):
                    continue
                rows.append(dict(base, kind="handover", **window_stats(sep, lo, hi)))
                rel_pos.append(i / len(sep))
            elif all(s[3] and s[4] for s in seq):
                placebo_pool.append((base, sep))
        print(mid, len(rows), len(placebo_pool), flush=True)

    # placebo windows at the relative positions handovers actually occur
    for base, sep in placebo_pool:
        i = int(rng.choice(rel_pos) * len(sep))
        lo, hi = max(0, i - WINDOW), min(len(sep), i + WINDOW + 1)
        rows.append(dict(base, kind="placebo", **window_stats(sep, lo, hi)))

    df = pd.DataFrame(rows).dropna(subset=["gain_share"])
    df["excess"] = df.gain_share - df.time_share
    df.to_csv("../data/kill_test_handover_mechanism.csv", index=False)
    print()
    print(df.groupby("kind").agg(n=("excess", "size"), time_share=("time_share", "mean"),
                                 gain_share=("gain_share", "mean"), peak_in_window=("peak_in", "mean"),
                                 excess=("excess", "mean")).round(3))

    h, p = df[df.kind == "handover"], df[df.kind == "placebo"]
    # match-level bootstrap of the handover-minus-placebo difference in excess gain share
    diffs = []
    matches = df.match.unique()
    for _ in range(2000):
        m = rng.choice(matches, len(matches))
        hb = pd.concat([h[h.match == x] for x in m]); pb = pd.concat([p[p.match == x] for x in m])
        diffs.append(hb.excess.mean() - pb.excess.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nhandover - placebo excess gain share: {h.excess.mean() - p.excess.mean():+.3f} "
          f"(95% match-bootstrap CI {lo:+.3f} to {hi:+.3f})")

    print("\nhandover runs: received rate by whether peak separation happened in the handover window")
    print(h.groupby("peak_in")[["received", "dangerous"]].agg(["mean", "size"]).round(3))


if __name__ == "__main__":
    main()
