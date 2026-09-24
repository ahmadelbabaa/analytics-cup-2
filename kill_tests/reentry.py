"""Kill test for "How real is the defensive shape?".

When a player comes back on camera, the jump between his last extrapolated position and his first
detected one is a direct measure of the extrapolation error. Does that error grow with time off camera?
"""
import numpy as np
import pandas as pd

from common import load_match, match_dirs


def reentries(frames):
    """One row per off-camera spell that ends with the player detected again."""
    out, unseen = [], {}  # unseen[pid] = frames spent extrapolated so far
    prev_f, prev = None, {}
    for f in sorted(frames):
        cur = frames[f]
        if prev_f is None or f != prev_f + 1:   # gap in data (stoppage): reset spells
            unseen = {}
        for pid, (x, y, det) in cur.items():
            if not det:
                unseen[pid] = unseen.get(pid, 0) + 1
            elif unseen.get(pid) and pid in prev:
                px, py, _ = prev[pid]
                out.append(dict(pid=pid, unseen_s=unseen[pid] / 10, jump=np.hypot(x - px, y - py)))
                unseen[pid] = 0
            else:
                unseen[pid] = 0
        prev_f, prev = f, cur
    return out


def normal_steps(frames):
    """Frame-to-frame displacement while detected on both frames (baseline: ~speed * 0.1 s)."""
    steps, fs = [], sorted(frames)
    for a, b in zip(fs[::50], [f + 1 for f in fs[::50]]):
        if b in frames:
            steps += [np.hypot(frames[b][p][0] - x, frames[b][p][1] - y)
                      for p, (x, y, d) in frames[a].items() if d and p in frames[b] and frames[b][p][2]]
    return steps


def main():
    rows, base = [], []
    for d in match_dirs():
        mid, _, _, frames = load_match(d)
        rows += [dict(r, match=mid) for r in reentries(frames)]
        base += normal_steps(frames)
        print(mid, len(rows), flush=True)
    df = pd.DataFrame(rows)
    print(f"\nre-entries: {len(df)}; baseline step while detected: median {np.median(base):.2f} m, "
          f"p95 {np.percentile(base, 95):.2f} m")
    bins = [0, 0.5, 1, 2, 5, 10, 20, np.inf]
    df["unseen"] = pd.cut(df.unseen_s, bins)
    print(df.groupby("unseen", observed=True).jump.describe(percentiles=[.5, .9])[["count", "mean", "50%", "90%"]].round(2))
    print("\nSpearman(unseen_s, jump) = %.2f" % df[["unseen_s", "jump"]].corr("spearman").iloc[0, 1])


if __name__ == "__main__":
    main()
